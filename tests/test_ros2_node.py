"""The ROS 2 grasp node, run without ROS: rclpy and the message packages are replaced by small stand-ins, and the
real gripper by the MuJoCo cell.  Checks the wiring (topics -> controller -> pressure command, services, action)."""
import sys
import threading
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "sim_mujoco"), str(REPO / "learning"), str(REPO / "core"), str(REPO / "ros2")]


class _Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _install_fake_ros():
    mods = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        mods[name] = m
        return m

    class Param:
        def __init__(self, v):
            self.value = v

    class Node:
        def __init__(self, name):
            self.params, self.subs, self.pubs, self.timers, self.services = {}, {}, {}, [], {}

        def declare_parameter(self, k, v):
            self.params[k] = Param(v)

        def get_parameter(self, k):
            return self.params[k]

        def create_subscription(self, typ, topic, cb, qos):
            self.subs[topic] = cb

        def create_publisher(self, typ, topic, qos):
            pub = types.SimpleNamespace(msgs=[])
            pub.publish = pub.msgs.append
            self.pubs[topic] = pub
            return pub

        def create_timer(self, period, cb):
            self.timers.append(cb)

        def create_service(self, typ, name, cb):
            self.services[name] = cb

    class GripperCommand:
        Feedback = Result = _Msg

    mod("rclpy", ok=lambda: True, init=lambda: None, shutdown=lambda: None)
    mod("rclpy.node", Node=Node)
    mod("rclpy.qos", qos_profile_sensor_data=None)
    mod("rclpy.callback_groups", ReentrantCallbackGroup=object)
    mod("rclpy.action", ActionServer=lambda *a, **k: None, CancelResponse=types.SimpleNamespace(ACCEPT=1),
        GoalResponse=types.SimpleNamespace(ACCEPT=1))
    mod("std_msgs"), mod("std_msgs.msg", Float32=_Msg, Float32MultiArray=_Msg)
    mod("std_srvs"), mod("std_srvs.srv", Trigger=types.SimpleNamespace(Request=_Msg, Response=_Msg))
    mod("control_msgs"), mod("control_msgs.action", GripperCommand=GripperCommand)
    sys.modules.update(mods)


@pytest.fixture(scope="module")
def node():
    _install_fake_ros()
    import softtwin_grasp_node as G
    return G.GraspNode()


def _plant(name, seed=3):
    """MuJoCo cell with the hand lowered to the planned grasp height, plus the stand-in muscle and the sensor models"""
    import softtwin_mj as S
    import twin_core as C
    cell = S.Cell(seed=seed, objects=(name,))
    o = cell.obj_pos(name)
    cell.move_palm([o[0], o[1], C.CFG["table_top"] + 0.22], 1.0)
    cell.move_palm([o[0], o[1], C.grasp_height(name, o[2])], 0.8)
    rig = C.make_rig(cell.inst, "learned", dt=cell.dt_ctrl)
    return cell, rig


def _drive(node, cell, rig, name, steps, stop=None):
    p_true = 0.0
    for _ in range(steps):
        s = rig.sensors
        node.subs["/real/pressure"](_Msg(data=s.pressure(p_true)))
        node.subs["/real/bend"](_Msg(data=list(s.bend(cell.bend_true()))))
        node.subs["/real/skin"](_Msg(data=list(s.skin(cell.taxels_true()))))
        node.timers[0]()
        cmd = node.pubs["/softtwin/pressure_cmd"].msgs[-1].data
        p_true = rig.supply.step(cmd, cell.dt_ctrl)
        cell.apply(rig.muscle.step(p_true, cell.dt_ctrl))
        cell.advance(1)
        if stop is not None and stop():
            break
    return cell.force_true(name)


def test_services(node):
    node.subs["/real/bend"](_Msg(data=[0.0, 0.0, 0.0]))
    res = node.services["/softtwin/zero"](_Msg(), _Msg())
    assert res.success and not node.bend_offset.any()
    assert node.services["/softtwin/release"](_Msg(), _Msg()).success and node.ctl.state == "idle"


def test_action_grasp_tomato(node):
    """goal F* = 2.2 N (the 'fragile' entry of the handling table): close, detect contact on the skin, regulate"""
    cell, rig = _plant("tomato")
    fb = []
    handle = types.SimpleNamespace(request=_Msg(command=_Msg(max_effort=2.2)), is_cancel_requested=False,
                                   publish_feedback=lambda f: fb.append(f.effort), succeed=lambda: None, abort=lambda: None)
    out = {}
    th = threading.Thread(target=lambda: out.update(res=node.execute(handle)))
    th.start()
    time.sleep(0.05)                       # execute() has started the controller
    f_true = _drive(node, cell, rig, "tomato", 600, stop=lambda: node.ctl.reached)
    th.join(timeout=15)
    res = out["res"]
    assert res.reached_goal, node.ctl.state
    assert abs(f_true - 2.2) < 0.4 and abs(res.effort - f_true) < 0.3
    assert fb
