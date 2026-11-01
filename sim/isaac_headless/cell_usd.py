"""The SoftTwin cell in USD with the geometry of sim_mujoco/softtwin_mj.build_xml (palm, 3 fingers x 4 segments, joint
stiffness k_b, segment mass, fingertip capsule, table, tray, one object instance from twin_core.sample_instance).
Used by q1_isaac.py (PhysX runs) and render_isaac.py (RTX renders of recorded runs).  Needs Kit (pxr, omni.usd)."""
import math

import numpy as np
import omni.usd

import twin_core as C

CFG = C.CFG
NF, NS, L = C.NF, C.NS, C.L
HOME = np.array(CFG["home"])
DT_PHYS = 0.001
M_FINGER = 0.0539           # effective finger mass of a contact (as identified in MuJoCo), for the contact damping


def quat_z(deg):
    from pxr import Gf
    h = math.radians(deg) / 2
    return Gf.Quatf(math.cos(h), 0.0, 0.0, math.sin(h))


def build(seed, name, k_scale):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    st = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.Xform.Define(st, "/World")
    scene = UsdPhysics.Scene.Define(st, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    ps = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    ps.CreateTimeStepsPerSecondAttr(int(round(1 / DT_PHYS)))
    ps.CreateEnableGPUDynamicsAttr(False)
    ps.CreateSolverTypeAttr("TGS")

    def material(path, mu, combine=None):
        m = UsdShade.Material.Define(st, path)
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr(float(mu))
        api.CreateDynamicFrictionAttr(float(mu))
        api.CreateRestitutionAttr(0.0)
        px = PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim())
        if combine:
            px.CreateFrictionCombineModeAttr(combine)
        return m, px

    def bind(prim, m):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(m, UsdShade.Tokens.weakerThanDescendants, "physics")

    def collider(prim, mat=None):
        UsdPhysics.CollisionAPI.Apply(prim)
        c = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        c.CreateContactOffsetAttr(0.002)
        c.CreateRestOffsetAttr(0.0)
        if mat is not None:
            bind(prim, mat)

    def box(path, center, half, mat=None, collide=True, rgb=None):
        c = UsdGeom.Cube.Define(st, path)
        if rgb is not None:
            c.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*center))
        c.AddScaleOp().Set(Gf.Vec3f(*[2 * h for h in half]))
        if collide:
            collider(c.GetPrim(), mat)
        return c

    def link(path, pos, rot, mass, inertia=None):
        x = UsdGeom.Xform.Define(st, path)
        x.AddTranslateOp().Set(Gf.Vec3d(*pos))
        x.AddOrientOp().Set(rot)
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        m = UsdPhysics.MassAPI.Apply(x.GetPrim())
        m.CreateMassAttr(float(mass))
        if inertia is not None:
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        return x

    def joint(path, kind, b0, b1, p0, r0, p1=(0, 0, 0), r1=None, axis="X", lo=None, hi=None):
        J = {"fixed": UsdPhysics.FixedJoint, "prismatic": UsdPhysics.PrismaticJoint, "revolute": UsdPhysics.RevoluteJoint}[kind].Define(st, path)
        if b0:
            J.CreateBody0Rel().SetTargets([b0])
        J.CreateBody1Rel().SetTargets([b1])
        J.CreateLocalPos0Attr(Gf.Vec3f(*p0))
        J.CreateLocalRot0Attr(r0)
        J.CreateLocalPos1Attr(Gf.Vec3f(*p1))
        J.CreateLocalRot1Attr(r1 if r1 is not None else Gf.Quatf(1, 0, 0, 0))
        if kind != "fixed":
            J.CreateAxisAttr(axis)
            if lo is not None:
                J.CreateLowerLimitAttr(lo)
                J.CreateUpperLimitAttr(hi)
        return J

    def drive(J, kind, k, d):
        dr = UsdPhysics.DriveAPI.Apply(J.GetPrim(), "angular" if kind == "revolute" else "linear")
        dr.CreateTypeAttr("force")
        dr.CreateStiffnessAttr(float(k))
        dr.CreateDampingAttr(float(d))
        dr.CreateTargetPositionAttr(0.0)
        dr.CreateMaxForceAttr(1e6)

    top = CFG["table_top"]
    finger_mat, _ = material("/World/Materials/silicone", 1.0)
    wood, _ = material("/World/Materials/wood", 0.8)
    box("/World/Table", (0, 0, top - 0.02), (0.5, 0.35, 0.02), wood, rgb=(0.62, 0.50, 0.38))
    tx, ty = CFG["tray"]
    for i, (h, c) in enumerate([((0.09, 0.07, 0.005), (tx, ty, top + 0.005)), ((0.005, 0.07, 0.02), (tx - 0.09, ty, top + 0.02)),
                                ((0.005, 0.07, 0.02), (tx + 0.09, ty, top + 0.02)), ((0.09, 0.005, 0.02), (tx, ty - 0.07, top + 0.02)),
                                ((0.09, 0.005, 0.02), (tx, ty + 0.07, top + 0.02))]):
        box(f"/World/Tray/w{i}", c, h, wood, rgb=(0.15, 0.35, 0.65))

    # gripper: fixed root -> slide x -> slide y -> slide z -> palm (gravity compensated, like MuJoCo gravcomp) -> fingers
    G = "/World/Gripper"
    hx, hy, hz = HOME
    I0 = Gf.Quatf(1, 0, 0, 0)
    root = link(f"{G}/root", (hx, hy, hz), I0, 1.0, (1e-3, 1e-3, 1e-3))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(True)                 # fingertips can touch each other (as in MuJoCo)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    joint(f"{G}/fix", "fixed", None, f"{G}/root", (hx, hy, hz), I0)
    prev = f"{G}/root"
    for ax, nm, kp, kv, lo, hi in [("X", "slide_x", 4000, 240, -0.45, 0.45), ("Y", "slide_y", 4000, 240, -0.35, 0.35), ("Z", "slide_z", 6000, 290, -0.3, 0.2)]:
        body = f"{G}/{nm}_body" if nm != "slide_z" else f"{G}/palm"
        b = link(body, (hx, hy, hz), I0, 0.5 if nm == "slide_z" else 0.01, (0.002, 0.002, 0.002) if nm == "slide_z" else (1e-5, 1e-5, 1e-5))
        PhysxSchema.PhysxRigidBodyAPI.Apply(b.GetPrim()).CreateDisableGravityAttr(True)
        J = joint(f"{G}/{nm}", "prismatic", prev, body, (0, 0, 0), I0, axis=ax, lo=lo, hi=hi)
        drive(J, "prismatic", kp, kv)                          # kv = MuJoCo kv + joint damping 40
        prev = body
    t, w = CFG["seg_t"], CFG["seg_w"]
    kb_deg, db_deg = CFG["k_b"] * math.pi / 180.0, CFG["d_b"] * math.pi / 180.0   # USD angular drives are per degree
    for f in range(NF):
        th = 90 + f * 360.0 / NF
        q = quat_z(th)
        ux, uy = math.cos(math.radians(th)), math.sin(math.radians(th))
        base = np.array([hx + CFG["finger_r"] * ux, hy + CFG["finger_r"] * uy, hz - CFG["palm"][2]])
        parent = f"{G}/palm"
        for k in range(NS):
            path = f"{G}/f{f}_s{k}"
            x = link(path, tuple(base - np.array([0, 0, k * L])), q, CFG["seg_mass"] + (0.001 if k == NS - 1 else 0.0))
            PhysxSchema.PhysxContactReportAPI.Apply(x.GetPrim()).CreateThresholdAttr(0.0)
            g = box(f"{path}/geom", (0, 0, -L / 2), (t / 2, w / 2, L * 0.48), finger_mat, rgb=(0.95, 0.45 + 0.08 * k, 0.20))
            if k == NS - 1:
                cap = UsdGeom.Capsule.Define(st, f"{path}/tip")
                cap.CreateRadiusAttr(0.004)
                cap.CreateHeightAttr(w * 0.8)
                cap.CreateAxisAttr("Y")
                cap.AddTranslateOp().Set(Gf.Vec3d(-t / 2 - 0.002, 0, -L + 0.004))
                cap.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.69, 0.20)])
                collider(cap.GetPrim(), finger_mat)
            if k == 0:
                J = joint(f"{G}/f{f}_j{k}", "revolute", parent, path, (CFG["finger_r"] * ux, CFG["finger_r"] * uy, -CFG["palm"][2]), q,
                          axis="Y", lo=-10.0, hi=80.0)
            else:
                J = joint(f"{G}/f{f}_j{k}", "revolute", parent, path, (0, 0, -L), I0, axis="Y", lo=-10.0, hi=80.0)
            drive(J, "revolute", kb_deg, db_deg)
            PhysxSchema.PhysxJointAPI.Apply(J.GetPrim()).CreateArmatureAttr(0.0002)
            parent = path
            del x, g

    # the object: this seed's instance (same as MuJoCo: twin_core.sample_instance)
    o, inst = C.OBJECTS[name], C.sample_instance(seed, name)
    sx, sy = inst["xy"]
    P = f"/World/Objects/{name}"
    UsdGeom.Xform.Define(st, "/World/Objects")
    x = link(P, (sx, sy, top + C.half_height(name) + 0.001), quat_z(inst["yaw_deg"]), inst["mass"])
    PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim()).CreateSolverPositionIterationCountAttr(32)
    PhysxSchema.PhysxContactReportAPI.Apply(x.GetPrim()).CreateThresholdAttr(0.0)
    mat, px = material(f"/World/Materials/{name}", inst["mu"], combine="min")     # MuJoCo: the object's friction wins (priority)
    if inst["k"] is not None:
        k_set = inst["k"] * k_scale
        m_eff = 1.0 / (1.0 / M_FINGER + 1.0 / inst["mass"])
        px.CreateCompliantContactStiffnessAttr(float(k_set))
        px.CreateCompliantContactDampingAttr(float(2.0 * math.sqrt(k_set * m_eff)))    # critical, as solref "tc 1"
    if o.shape == "sphere":
        g = UsdGeom.Sphere.Define(st, f"{P}/geom")
        g.CreateRadiusAttr(o.size[0])
        g.CreateDisplayColorAttr([Gf.Vec3f(*o.rgba[:3])])
        collider(g.GetPrim(), mat)
    elif o.shape == "box":
        box(f"{P}/geom", (0, 0, 0), o.size, mat, rgb=o.rgba[:3])
    elif o.shape == "cylinder":
        g = UsdGeom.Cylinder.Define(st, f"{P}/geom")
        g.CreateRadiusAttr(o.size[0])
        g.CreateHeightAttr(2 * o.size[1])
        g.CreateAxisAttr("Z")
        g.CreateDisplayColorAttr([Gf.Vec3f(*o.rgba[:3])])
        collider(g.GetPrim(), mat)
    else:                                                          # ellipsoid: convex hull of a fine mesh
        a, b, c = o.size
        pts = [(0.0, 0.0, c)]
        for i in range(1, 15):
            th_ = math.pi * i / 15
            for j in range(16):
                ph = 2 * math.pi * j / 16
                pts.append((a * math.sin(th_) * math.cos(ph), b * math.sin(th_) * math.sin(ph), c * math.cos(th_)))
        pts.append((0.0, 0.0, -c))
        faces, idx = [], []
        for j in range(16):
            faces.append(3)
            idx += [0, 1 + j, 1 + (j + 1) % 16]
        for i in range(13):
            for j in range(16):
                faces.append(4)
                a0, a1 = 1 + i * 16 + j, 1 + i * 16 + (j + 1) % 16
                idx += [a0, a0 + 16, a1 + 16, a1]
        last = len(pts) - 1
        for j in range(16):
            faces.append(3)
            idx += [1 + 13 * 16 + j, last, 1 + 13 * 16 + (j + 1) % 16]
        g = UsdGeom.Mesh.Define(st, f"{P}/geom")
        g.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        g.CreateFaceVertexCountsAttr(faces)
        g.CreateFaceVertexIndicesAttr(idx)
        g.CreateDisplayColorAttr([Gf.Vec3f(*o.rgba[:3])])
        g.CreateSubdivisionSchemeAttr("none")
        collider(g.GetPrim(), mat)
        UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("convexHull")
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(g.GetPrim()).CreateHullVertexLimitAttr(255)
    return inst
