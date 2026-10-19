"""Create the ROS 2 OmniGraph for the SoftTwin cell (run inside Isaac Sim after softtwin_scene.py).

Publishes: /clock, /softtwin/joint_states (gripper articulation), /softtwin/overhead/rgb (+camera_info)
Subscribes: /softtwin/joint_command (sensor_msgs/JointState, position targets) -> articulation controller
Requires the isaacsim.ros2.bridge extension (enabled by tools/launch_sim.bat).
"""
import omni.graph.core as og

GRAPH = "/World/ROS2Graph"
ART = "/World/Gripper/root"
CAM = "/World/Cameras/overhead"


def build(publish_camera=True):
    keys = og.Controller.Keys
    nodes = [
        ("tick", "omni.graph.action.OnPlaybackTick"),
        ("ctx", "isaacsim.ros2.bridge.ROS2Context"),
        ("simtime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("pub_js", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ("sub_js", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
        ("artctl", "isaacsim.core.nodes.IsaacArticulationController"),
    ]
    conns = [
        ("tick.outputs:tick", "clock.inputs:execIn"),
        ("tick.outputs:tick", "pub_js.inputs:execIn"),
        ("tick.outputs:tick", "sub_js.inputs:execIn"),
        ("tick.outputs:tick", "artctl.inputs:execIn"),
        ("ctx.outputs:context", "clock.inputs:context"),
        ("ctx.outputs:context", "pub_js.inputs:context"),
        ("ctx.outputs:context", "sub_js.inputs:context"),
        ("simtime.outputs:simulationTime", "clock.inputs:timeStamp"),
        ("simtime.outputs:simulationTime", "pub_js.inputs:timeStamp"),
        ("sub_js.outputs:jointNames", "artctl.inputs:jointNames"),
        ("sub_js.outputs:positionCommand", "artctl.inputs:positionCommand"),
    ]
    values = [
        ("pub_js.inputs:topicName", "/softtwin/joint_states"),
        ("pub_js.inputs:targetPrim", ART),
        ("sub_js.inputs:topicName", "/softtwin/joint_command"),
        ("artctl.inputs:targetPrim", ART),
    ]
    if publish_camera:
        nodes += [("rp", "isaacsim.core.nodes.IsaacCreateRenderProduct"), ("rgb", "isaacsim.ros2.bridge.ROS2CameraHelper"), ("info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper")]
        conns += [
            ("tick.outputs:tick", "rp.inputs:execIn"),
            ("rp.outputs:execOut", "rgb.inputs:execIn"),
            ("rp.outputs:execOut", "info.inputs:execIn"),
            ("rp.outputs:renderProductPath", "rgb.inputs:renderProductPath"),
            ("rp.outputs:renderProductPath", "info.inputs:renderProductPath"),
            ("ctx.outputs:context", "rgb.inputs:context"),
            ("ctx.outputs:context", "info.inputs:context"),
        ]
        values += [
            ("rp.inputs:cameraPrim", CAM),
            ("rp.inputs:width", 640),
            ("rp.inputs:height", 480),
            ("rgb.inputs:topicName", "/softtwin/overhead/rgb"),
            ("rgb.inputs:type", "rgb"),
            ("rgb.inputs:frameId", "overhead"),
            ("info.inputs:topicName", "/softtwin/overhead/camera_info"),
            ("info.inputs:frameId", "overhead"),
        ]
    og.Controller.edit(
        {"graph_path": GRAPH, "evaluator_name": "execution"},
        {keys.CREATE_NODES: nodes, keys.CONNECT: conns, keys.SET_VALUES: values},
    )
    return {"graph": GRAPH, "nodes": [n for n, _ in nodes]}


_result = build()
print("ROS 2 graph created:", _result)
