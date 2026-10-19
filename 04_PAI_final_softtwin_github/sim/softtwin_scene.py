"""Build the SoftTwin cell in the open Isaac Sim stage (run inside Kit via MCP: isaac_run_file('sim/softtwin_scene.py')).

Cell: table + 3-axis gantry + 3-finger soft pneumatic gripper + fragile/hard objects + tray + cameras.
The soft fingers are approximated as chains of rigid segments joined by compliant revolute joints:
  - the drive target of every joint = bend angle predicted by the actuator model from the chamber pressure
  - the drive stiffness = bending stiffness of the silicone, so contact makes the finger yield (passive compliance)
This is the classic "pseudo-rigid-body model" (PRBM) of a soft bending actuator.

Only pxr/USD + PhysX schemas are used so the script is stable across Isaac Sim versions.
Coordinates: metres, Z-up.
"""
from __future__ import annotations

import json
import math
import random

import omni.usd
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

CFG = {
    "table_top": 0.75,
    "gantry_home": [0.0, 0.0, 1.05],  # palm position (world) at joint zero
    "n_fingers": 3,
    "n_segments": 4,
    "seg_len": 0.026,
    "seg_w": 0.020,
    "seg_t": 0.012,
    "finger_radius": 0.034,
    "palm": [0.10, 0.10, 0.02],
    "bend_stiffness": 0.004,  # N*m / deg per joint (silicone bending stiffness, PRBM)
    "bend_damping": 0.0004,
    "slide_stiffness": 30000.0,
    "slide_damping": 3000.0,
    "seed": 7,
}

# object catalogue: name -> (shape, size, mass kg, color, max safe grip force N, class label)
OBJECTS = {
    "strawberry": ("sphere", [0.019], 0.018, (0.85, 0.08, 0.12), 3.0, "fragile"),
    "egg": ("ellipsoid", [0.021, 0.021, 0.028], 0.060, (0.95, 0.90, 0.80), 8.0, "fragile"),
    "tofu": ("cube", [0.045, 0.045, 0.030], 0.040, (0.97, 0.97, 0.94), 1.5, "very_fragile"),
    "tomato": ("sphere", [0.028], 0.090, (0.90, 0.25, 0.10), 6.0, "fragile"),
    "bolt": ("cylinder", [0.009, 0.055], 0.035, (0.45, 0.47, 0.50), 1000.0, "rigid"),
    # look-alikes: same appearance as a base object, different hidden physics (what the camera cannot see)
    "tomato_overripe": ("sphere", [0.028], 0.090, (0.90, 0.25, 0.10), 1.5, "soft"),
    "sample_strawberry": ("sphere", [0.019], 0.040, (0.85, 0.08, 0.12), 1000.0, "rigid"),
}
# hidden surface compliance per object: PhysX compliant contact (stiffness N/m, damping N s/m); None = rigid contact
COMPLIANCE = {"strawberry": (400.0, 4.0), "tofu": (600.0, 5.0), "tomato": (3000.0, 10.0), "tomato_overripe": (350.0, 4.0)}


def _stage():
    return omni.usd.get_context().get_stage()


def _mat(stage, path, rgb, rough=0.5):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _phys_mat(stage, path, sf, df, rest=0.0):
    m = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    api.CreateStaticFrictionAttr(sf)
    api.CreateDynamicFrictionAttr(df)
    api.CreateRestitutionAttr(rest)
    return m


def _bind_phys(prim, m):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(m, UsdShade.Tokens.weakerThanDescendants, "physics")


def _xform(stage, path, pos, quat=Gf.Quatd(1, 0, 0, 0)):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat)
    return x


def _rigid_link(stage, path, pos, quat, mass, geom="box", size=(0.01, 0.01, 0.01), rgb=(0.2, 0.2, 0.2), collide=True, mat=None):
    x = _xform(stage, path, pos, quat)
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
    if geom == "box":
        g = UsdGeom.Cube.Define(stage, path + "/geom")
        g.CreateSizeAttr(1.0)
        g.AddScaleOp().Set(Gf.Vec3f(*size))
    else:
        g = UsdGeom.Capsule.Define(stage, path + "/geom")
        g.CreateRadiusAttr(size[0])
        g.CreateHeightAttr(size[1])
        g.CreateAxisAttr("Z")
    g.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        if mat is not None:
            _bind_phys(g.GetPrim(), mat)
    return x


def _joint(stage, path, kind, body0, body1, p0, p1, r0=Gf.Quatf(1, 0, 0, 0), r1=Gf.Quatf(1, 0, 0, 0), axis="X", lower=None, upper=None):
    if kind == "prismatic":
        j = UsdPhysics.PrismaticJoint.Define(stage, path)
    elif kind == "revolute":
        j = UsdPhysics.RevoluteJoint.Define(stage, path)
    else:
        j = UsdPhysics.FixedJoint.Define(stage, path)
    if body0:
        j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateLocalPos0Attr(Gf.Vec3f(*p0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*p1))
    j.CreateLocalRot0Attr(r0)
    j.CreateLocalRot1Attr(r1)
    if kind in ("prismatic", "revolute"):
        j.CreateAxisAttr(axis)
        if lower is not None:
            j.CreateLowerLimitAttr(lower)
            j.CreateUpperLimitAttr(upper)
    return j


def _drive(joint, kind, stiffness, damping, target=0.0, max_force=1e6):
    d = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear" if kind == "prismatic" else "angular")
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(stiffness)
    d.CreateDampingAttr(damping)
    d.CreateTargetPositionAttr(target)
    d.CreateMaxForceAttr(max_force)
    return d


def quat_z(deg):
    h = math.radians(deg) / 2
    return Gf.Quatd(math.cos(h), 0, 0, math.sin(h))


def build(seed: int | None = None, objects=None):
    stage = _stage()
    seed = CFG["seed"] if seed is None else seed
    rng = random.Random(seed)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for p in ("/World", "/Looks", "/PhysicsMaterials"):
        if stage.GetPrimAtPath(p):
            stage.RemovePrim(p)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    ps = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    ps.CreateTimeStepsPerSecondAttr(240)
    ps.CreateEnableGPUDynamicsAttr(False)

    # lights
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(900)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    sun.CreateIntensityAttr(2500)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40, 20, 0))

    silicone = _phys_mat(stage, "/PhysicsMaterials/silicone", 1.2, 1.0)
    wood = _phys_mat(stage, "/PhysicsMaterials/wood", 0.6, 0.5)
    produce = _phys_mat(stage, "/PhysicsMaterials/produce", 0.9, 0.8)

    # ground + table + tray
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    s = 5.0
    ground.CreatePointsAttr([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.32, 0.33, 0.35)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    top = CFG["table_top"]

    def static_box(path, center, size, rgb, mat=None):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*center))
        c.AddScaleOp().Set(Gf.Vec3f(*size))
        c.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if mat is not None:
            _bind_phys(c.GetPrim(), mat)
        return c

    static_box("/World/Table/top", (0, 0, top - 0.02), (1.0, 0.7, 0.04), (0.62, 0.50, 0.38), wood)
    for i, (x, y) in enumerate([(-0.46, -0.31), (0.46, -0.31), (-0.46, 0.31), (0.46, 0.31)]):
        static_box(f"/World/Table/leg{i}", (x, y, (top - 0.04) / 2), (0.04, 0.04, top - 0.04), (0.35, 0.28, 0.22))
    tray = (0.30, 0.20)
    static_box("/World/Tray/base", (tray[0], tray[1], top + 0.005), (0.18, 0.14, 0.01), (0.15, 0.35, 0.65))
    for n, c, sz in [
        ("wx0", (tray[0] - 0.09, tray[1], top + 0.025), (0.01, 0.14, 0.04)),
        ("wx1", (tray[0] + 0.09, tray[1], top + 0.025), (0.01, 0.14, 0.04)),
        ("wy0", (tray[0], tray[1] - 0.07, top + 0.025), (0.18, 0.01, 0.04)),
        ("wy1", (tray[0], tray[1] + 0.07, top + 0.025), (0.18, 0.01, 0.04)),
    ]:
        static_box(f"/World/Tray/{n}", c, sz, (0.15, 0.35, 0.65))

    # ---------------- gripper articulation ----------------
    G = "/World/Gripper"
    UsdGeom.Xform.Define(stage, G)
    hx, hy, hz = CFG["gantry_home"]
    root = _rigid_link(stage, f"{G}/root", (hx, hy, hz + 0.25), Gf.Quatd(1, 0, 0, 0), 2.0, size=(0.06, 0.06, 0.06), rgb=(0.2, 0.2, 0.22), collide=False)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim()).CreateEnabledSelfCollisionsAttr(False)
    _joint(stage, f"{G}/joints/fix_world", "fixed", None, f"{G}/root", (hx, hy, hz + 0.25), (0, 0, 0))
    cx = _rigid_link(stage, f"{G}/carriage_x", (hx, hy, hz + 0.25), Gf.Quatd(1, 0, 0, 0), 0.5, size=(0.05, 0.05, 0.05), rgb=(0.25, 0.25, 0.28), collide=False)
    cy = _rigid_link(stage, f"{G}/carriage_y", (hx, hy, hz + 0.25), Gf.Quatd(1, 0, 0, 0), 0.5, size=(0.04, 0.04, 0.04), rgb=(0.25, 0.25, 0.28), collide=False)
    palm_size = CFG["palm"]
    palm = _rigid_link(stage, f"{G}/palm", (hx, hy, hz), Gf.Quatd(1, 0, 0, 0), 0.3, size=palm_size, rgb=(0.85, 0.85, 0.88), collide=True)
    # mast so the Z slide reads visually
    mast = UsdGeom.Cylinder.Define(stage, f"{G}/palm/mast")
    mast.CreateRadiusAttr(0.012)
    mast.CreateHeightAttr(0.25)
    mast.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.135))
    mast.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.6, 0.62)])

    ks, kd = CFG["slide_stiffness"], CFG["slide_damping"]
    jx = _joint(stage, f"{G}/joints/slide_x", "prismatic", f"{G}/root", f"{G}/carriage_x", (0, 0, 0), (0, 0, 0), axis="X", lower=-0.45, upper=0.45)
    jy = _joint(stage, f"{G}/joints/slide_y", "prismatic", f"{G}/carriage_x", f"{G}/carriage_y", (0, 0, 0), (0, 0, 0), axis="Y", lower=-0.32, upper=0.32)
    jz = _joint(stage, f"{G}/joints/slide_z", "prismatic", f"{G}/carriage_y", f"{G}/palm", (0, 0, -0.25), (0, 0, 0), axis="Z", lower=-0.30, upper=0.10)
    for j in (jx, jy, jz):
        _drive(j, "prismatic", ks, kd)

    n, L = CFG["n_segments"], CFG["seg_len"]
    w, t, r = CFG["seg_w"], CFG["seg_t"], CFG["finger_radius"]
    kb, db = CFG["bend_stiffness"], CFG["bend_damping"]
    finger_joints = []
    for f in range(CFG["n_fingers"]):
        th = 90 + f * 360.0 / CFG["n_fingers"]
        qd = quat_z(th)
        qf = Gf.Quatf(qd.GetReal(), *[float(v) for v in qd.GetImaginary()])
        ux, uy = math.cos(math.radians(th)), math.sin(math.radians(th))
        base = (hx + r * ux, hy + r * uy, hz - palm_size[2] / 2)
        prev = f"{G}/palm"
        for k in range(n):
            path = f"{G}/f{f}_s{k}"
            c = (base[0], base[1], base[2] - (k + 0.5) * L)
            shade = 0.55 + 0.08 * k
            _rigid_link(stage, path, c, qd, 0.008, size=(t, w, L * 0.96), rgb=(0.95, shade * 0.6, 0.25), collide=True, mat=silicone)
            # tactile skin: every segment reports its contacts (sim/tactile_isaac.py bins them into 2 taxels per segment)
            cr = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path))
            cr.CreateThresholdAttr(0.0)
            if k == 0:
                p0, r0 = (r * ux, r * uy, -palm_size[2] / 2), qf
            else:
                p0, r0 = (0, 0, -L / 2), Gf.Quatf(1, 0, 0, 0)
            j = _joint(stage, f"{G}/joints/f{f}_j{k}", "revolute", prev, path, p0, (0, 0, L / 2), r0=r0, axis="Y", lower=-10.0, upper=75.0)
            _drive(j, "revolute", kb, db, 0.0, max_force=5.0)
            finger_joints.append(str(j.GetPath()))
            prev = path

    # ---------------- objects ----------------
    names = objects or ["strawberry", "egg", "tofu", "tomato", "bolt"]
    spots = [(-0.30, -0.18), (-0.12, -0.20), (0.05, -0.18), (-0.25, 0.05), (-0.05, 0.08), (0.12, -0.02)]
    rng.shuffle(spots)
    UsdGeom.Xform.Define(stage, "/World/Objects")
    meta = {}
    for i, name in enumerate(names):
        shape, size, mass, rgb, fmax, cls = OBJECTS[name]
        x, y = spots[i]
        x += rng.uniform(-0.02, 0.02)
        y += rng.uniform(-0.02, 0.02)
        path = f"/World/Objects/{name}"
        if shape == "sphere":
            z = top + size[0] + 0.002
        elif shape == "ellipsoid":
            z = top + size[2] + 0.002
        elif shape == "cube":
            z = top + size[2] / 2 + 0.002
        else:
            z = top + size[1] / 2 + 0.002
        xf = _xform(stage, path, (x, y, z))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        if shape in ("sphere", "ellipsoid"):
            g = UsdGeom.Sphere.Define(stage, path + "/geom")
            g.CreateRadiusAttr(1.0)
            sc = (size[0], size[0], size[0]) if shape == "sphere" else tuple(size)
            g.AddScaleOp().Set(Gf.Vec3f(*sc))
        elif shape == "cube":
            g = UsdGeom.Cube.Define(stage, path + "/geom")
            g.CreateSizeAttr(1.0)
            g.AddScaleOp().Set(Gf.Vec3f(*size))
        else:
            g = UsdGeom.Cylinder.Define(stage, path + "/geom")
            g.CreateRadiusAttr(size[0])
            g.CreateHeightAttr(size[1])
            g.CreateAxisAttr("Z")
        g.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        if name in COMPLIANCE:   # soft flesh: compliant contact material for this object only
            ks_c, kd_c = COMPLIANCE[name]
            soft = _phys_mat(stage, f"/World/Materials/soft_{name}", 0.9, 0.8)
            pm = PhysxSchema.PhysxMaterialAPI.Apply(stage.GetPrimAtPath(f"/World/Materials/soft_{name}"))
            pm.CreateCompliantContactStiffnessAttr(ks_c)
            pm.CreateCompliantContactDampingAttr(kd_c)
            _bind_phys(g.GetPrim(), soft)
        else:
            _bind_phys(g.GetPrim(), produce)
        for k, v in {"softtwin:max_force": float(fmax), "softtwin:class": cls, "softtwin:name": name}.items():
            prim.CreateAttribute(k, Sdf.ValueTypeNames.Float if isinstance(v, float) else Sdf.ValueTypeNames.String).Set(v)
        meta[name] = {"path": path, "pos": [x, y, z], "max_force": fmax, "class": cls, "mass": mass}

    # ---------------- cameras ----------------
    wrist = UsdGeom.Camera.Define(stage, f"{G}/palm/wrist_cam")   # looks straight down (-Z of the palm)
    wrist.CreateFocalLengthAttr(12.0)
    wrist.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10))
    wrist.AddTranslateOp().Set(Gf.Vec3d(0, 0, -palm_size[2] / 2 - 0.002))
    cams = {
        "/World/Cameras/overhead": ((0.0, 0.0, 1.9), (0.0, 0.0, top)),
        "/World/Cameras/hero": ((0.85, -0.85, 1.25), (0.0, 0.0, top + 0.1)),
        "/World/Cameras/closeup": ((0.30, -0.35, 0.95), (-0.05, -0.05, top + 0.03)),
    }
    for path, (eye, target) in cams.items():
        cam = UsdGeom.Camera.Define(stage, path)
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
        UsdGeom.Xformable(cam).AddTransformOp().Set(m)

    info = {"seed": seed, "objects": meta, "finger_joints": finger_joints, "cfg": CFG, "tray": [tray[0], tray[1], top]}
    stage.GetPrimAtPath("/World").CreateAttribute("softtwin:scene_info", Sdf.ValueTypeNames.String).Set(json.dumps(info))
    return info


if __name__ in ("__main__", "__mcp__"):
    _result = build()
    print(f"SoftTwin cell built: {len(_result['objects'])} objects, {len(_result['finger_joints'])} finger joints")
