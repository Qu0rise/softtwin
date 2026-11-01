"""ROS 2 node for the real gripper: force-controlled grasp as an action, zero / release as services.

    python3 ros2/softtwin_grasp_node.py --ros-args -p actuator:=learned

interfaces (standard packages only; nothing to build)
  action   /softtwin/grasp        control_msgs/action/GripperCommand
           goal.command.max_effort = F* [N] (decided upstream from the handling table)
           feedback.effort = F_hat [N], feedback.position = mean bend [rad], feedback.reached_goal
  service  /softtwin/zero         std_srvs/Trigger   bend offsets from the unloaded fingers (call with the hand open)
  service  /softtwin/release      std_srvs/Trigger   vent, abort the running grasp
  sub      /real/pressure         std_msgs/Float32            kPa       (sensor QoS: best effort, depth 1)
  sub      /real/bend             std_msgs/Float32MultiArray  deg x 3   (one bend sensor per finger)
  sub      /real/skin             std_msgs/Float32MultiArray  N x 24    (fabric skin, finger-major)
  pub      /softtwin/pressure_cmd std_msgs/Float32            kPa, 50 Hz
  params   actuator (learned | greybox | static), rate_hz (50), p_max (80)

The control law is core/grip_controller.py, which calls sim_mujoco/twin_core.py: the same code the simulations ran.
tests/test_ros2_node.py runs this node with stand-ins for rclpy against the MuJoCo cell.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "core"))

import rclpy  # noqa: E402
from control_msgs.action import GripperCommand  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from std_msgs.msg import Float32, Float32MultiArray  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from grip_controller import GripController  # noqa: E402


class GraspNode(Node):
    def __init__(self):
        super().__init__("softtwin_grasp")
        self.declare_parameter("actuator", "learned")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("p_max", 80.0)
        self.ctl = GripController(self.get_parameter("actuator").value, p_max=self.get_parameter("p_max").value)
        self.p, self.bend, self.skin = 0.0, np.zeros(3), np.zeros(24)
        self.bend_offset = np.zeros(3)
        self.create_subscription(Float32, "/real/pressure", lambda m: setattr(self, "p", float(m.data)), qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/real/bend", lambda m: setattr(self, "bend", np.asarray(m.data, float)), qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/real/skin", lambda m: setattr(self, "skin", np.asarray(m.data, float)), qos_profile_sensor_data)
        self.pub = self.create_publisher(Float32, "/softtwin/pressure_cmd", 10)
        self.dt = 1.0 / self.get_parameter("rate_hz").value
        self.create_timer(self.dt, self.tick)
        self.create_service(Trigger, "/softtwin/zero", self.on_zero)
        self.create_service(Trigger, "/softtwin/release", self.on_release)
        # the action runs in its own (reentrant) group so that its waiting loop never stops the 50 Hz timer
        self.server = ActionServer(self, GripperCommand, "/softtwin/grasp", execute_callback=self.execute,
                                   goal_callback=lambda g: GoalResponse.ACCEPT, cancel_callback=lambda g: CancelResponse.ACCEPT,
                                   callback_group=ReentrantCallbackGroup())

    def tick(self):
        """50 Hz: the controller runs here, not in the action callback, so a slow client never delays the force loop"""
        cmd = self.ctl.step(self.p, self.bend - self.bend_offset, self.skin, self.dt)
        self.pub.publish(Float32(data=float(cmd)))

    def on_zero(self, req, res):
        self.bend_offset = np.array(self.bend, float)
        res.success, res.message = True, "bend offsets " + ", ".join(f"{v:.2f}" for v in self.bend_offset) + " deg"
        return res

    def on_release(self, req, res):
        self.ctl.release()
        res.success, res.message = True, "vented"
        return res

    def execute(self, handle):
        self.ctl.start(float(handle.request.command.max_effort))
        fb = GripperCommand.Feedback()
        t_end = time.monotonic() + 10.0
        while rclpy.ok() and time.monotonic() < t_end and not self.ctl.reached:
            if handle.is_cancel_requested:
                self.ctl.release()
                handle.canceled()
                return self._result()
            fb.effort, fb.position, fb.reached_goal, fb.stalled = self.ctl.F_hat, self._pos(), self.ctl.reached, False
            handle.publish_feedback(fb)
            time.sleep(self.dt)
        res = self._result()
        handle.succeed() if res.reached_goal else handle.abort()
        return res

    def _pos(self):
        return math.radians(float(np.mean(self.bend - self.bend_offset)))

    def _result(self):
        r = GripperCommand.Result()
        r.effort, r.position = self.ctl.F_hat, self._pos()
        r.reached_goal, r.stalled = self.ctl.reached, (not self.ctl.reached and self.ctl.contact)
        return r


def main():
    rclpy.init()
    node = GraspNode()
    from rclpy.executors import MultiThreadedExecutor
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    try:
        ex.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
