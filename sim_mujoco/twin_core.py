"""SoftTwin core: everything except the physics engine.

Object specs, the random instance of each object per seed, the stand-in muscle, the actuator models, the sensors
(pressure, bend, fabric skin), the force estimator, the force controller, the pick-and-place script and the outcome
labels.  MuJoCo (sim_mujoco/softtwin_mj.py) and Isaac Sim / PhysX (sim/isaac_headless/q1_isaac.py) both run this
file unchanged; each only provides an `engine` with the small interface below, so a difference between the two
engines can only come from the physics.

engine interface (one gripper, one object):
    dt_ctrl                      control period [s] (0.02)
    t                            simulated time [s]
    obj_pos(name)                object centre, world frame [m]
    apply(bend_deg)              set the free bend of every finger (what the muscle does) [deg]
    advance(n_ctrl=1)            simulate n control periods with the current palm target and bend
    set_palm(xyz)                palm target, world frame [m] (the gantry follows it with a stiff servo)
    bend_true()                  (3,) total bend of each finger = sum of its joint angles [deg]
    taxels_true()                (24,) normal force on each taxel area of the inner finger surface [N]
    force_true(name)             sum of the normal contact forces between the fingers and the object [N]
    on_frame(tag, ep)            camera hook (optional)

No numpy-free dependencies: this file must also import inside Isaac Sim's Kit python.
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "learning"))
from actuator_net import PFilter, SyntheticMuscle  # noqa: E402

# ------------------------------------------------------------------ cell geometry (shared by both engines)
CFG = dict(table_top=0.75, home=(0.0, 0.0, 1.00), n_fingers=3, n_seg=4, seg_len=0.026, seg_w=0.020, seg_t=0.012, seg_mass=0.006,
           finger_r=0.045, palm=(0.06, 0.06, 0.01), k_b=0.5, d_b=0.006, tray=(0.30, 0.20), dt_ctrl=0.02)
NF, NS, L = CFG["n_fingers"], CFG["n_seg"], CFG["seg_len"]
LEVER = np.array([(NS - k) * L for k in range(NS)])          # lever arm of a fingertip force about each joint [m]
TIP_COMPLIANCE = float(LEVER @ LEVER) / CFG["k_b"]           # fingertip compliance of one finger [m/N] (0.041 m/N)
SPOTS = [(-0.28, -0.16), (-0.12, -0.18), (0.04, -0.16), (-0.24, 0.06), (-0.06, 0.08), (0.10, -0.02)]

# ------------------------------------------------------------------ objects
# Sizes, masses, contact stiffness, friction and allowable force come from the literature (docs/physical_parameters.md).
#   size : MuJoCo half sizes (ellipsoid / box), radius (sphere), radius + half length (cylinder) [m]
#   k    : stiffness of one finger-pad contact [N/m]; None = practically rigid (egg shell 1e5 N/m and up)
#   fmax : allowable total grip force (sum over the three fingers) before damage [N]
#   cls  : handling class (what the right grip is), see HANDLING
Obj = namedtuple("Obj", "shape size mass rgba fmax cls mu k label")
OBJECTS = {
    "strawberry": Obj("ellipsoid", (0.020, 0.017, 0.0145), 0.020, (0.85, 0.08, 0.12, 1), 3.0, "fragile", 0.6, 1000.0, "いちご"),
    "egg": Obj("ellipsoid", (0.028, 0.021, 0.0205), 0.056, (0.95, 0.90, 0.80, 1), 25.0, "sturdy", 0.5, None, "卵"),
    "tofu": Obj("box", (0.017, 0.017, 0.015), 0.036, (0.97, 0.97, 0.92, 1), 4.0, "fragile", 0.4, 500.0, "木綿豆腐"),
    "tomato": Obj("sphere", (0.022,), 0.043, (0.90, 0.25, 0.10, 1), 5.0, "fragile", 0.6, 1500.0, "中玉トマト"),
    "bolt": Obj("cylinder", (0.006, 0.020), 0.035, (0.55, 0.57, 0.60, 1), 1e9, "sturdy", 0.4, None, "ボルト"),
    # look-alikes: same appearance as a base object, different hidden physics (what a camera cannot see)
    "tomato_overripe": Obj("sphere", (0.022,), 0.043, (0.90, 0.25, 0.10, 1), 2.0, "soft", 0.6, 600.0, "過熟中玉トマト"),
    "strawberry_overripe": Obj("ellipsoid", (0.020, 0.017, 0.0145), 0.020, (0.85, 0.08, 0.12, 1), 1.5, "very_fragile", 0.6, 400.0, "過熟いちご"),
}
BASE_OBJECTS = ("strawberry", "egg", "tofu", "tomato", "bolt")
LOOKALIKES = ("tomato_overripe", "strawberry_overripe")
ALL_OBJECTS = BASE_OBJECTS + LOOKALIKES
VISUAL_ID = {"tomato_overripe": "tomato", "strawberry_overripe": "strawberry"}
CLASSES = ("very_fragile", "soft", "fragile", "sturdy")
# handling table of the line: target grip force per class, and the class's allowable force (the plant's spec, not learned)
HANDLING = {"very_fragile": dict(target=1.2, limit=1.5), "soft": dict(target=1.55, limit=2.0),
            "fragile": dict(target=2.2, limit=3.0), "sturdy": dict(target=3.0, limit=25.0)}
TARGET_FORCE = {c: h["target"] for c, h in HANDLING.items()}

# weight-aware floor on the target (used by the size test and the Q2 variant 'plunger_clf_mass'):
#   F* = min(max(target of the class, FLOOR_SAFETY * m_hat (g + A_MAX) / mu_hat), FLOOR_CAP * limit of the class)
# m_hat, mu_hat: nominal mass and friction of the object TYPE the camera recognises (scaled by the measured size^3);
# the individual's mass and friction are not known.  FLOOR_SAFETY is read off nominal-size objects only (see size_shift.py).
FLOOR_SAFETY, A_MAX, FLOOR_CAP = 2.5, 0.9, 0.85


def weight_floor_target(type_name, cls, scale=1.0):
    o, h = OBJECTS[type_name], HANDLING[cls]
    f_min = FLOOR_SAFETY * o.mass * scale ** 3 * (9.81 + A_MAX) / o.mu
    return round(float(min(max(h["target"], f_min), FLOOR_CAP * h["limit"])), 3)
VISUAL_CLASS = {n: OBJECTS[n].cls for n in BASE_OBJECTS}
RANDOMISE = dict(mass=0.15, mu=0.20, k=0.20, xy=0.015)    # +- fractions (xy: +- metres)


def half_height(name):
    o = OBJECTS[name]
    return o.size[0] if o.shape == "sphere" else (o.size[1] if o.shape == "cylinder" else o.size[2])


def finger_tip(bend_deg):
    """inner face of a fingertip for a uniform bend: (radius from the palm axis, drop below the finger base) [m]"""
    phi, R, D = math.radians(bend_deg) / NS, CFG["finger_r"], 0.0
    for k in range(NS):
        a = (k + 1) * phi
        R, D = R - L * math.sin(a), D + L * math.cos(a)
    a = NS * phi
    return R - CFG["seg_t"] / 2 * math.cos(a), D + CFG["seg_t"] / 2 * math.sin(a)


def grasp_height(name, obj_z, F_star=None):
    """palm height for the grasp, from the object's size (known from the camera).

    On a rounded object the fingertips meet it below the equator, so the contact normals point a little upward and carry
    part of the weight (cradle); a deeper cradle holds better while the hand accelerates.  Too deep, and a grip much
    larger than the weight squeezes the object out upward: with three contacts at angle alpha below the equator this
    happens when F* (sin(alpha) - mu cos(alpha)) > m g.  The planner takes the deepest alpha (at most asin 0.6) that keeps
    F* (sin(alpha) - mu_w cos(alpha)) <= 0.5 m g, with m and mu the nominal mass and friction of what the object looks
    like, mu_w = 0.8 mu (the low end of the spread) and F* the class target.  Boxes and upright cylinders: half way down.
    The bend at which a tip reaches the surface comes from the finger kinematics; the tip's rise at that bend is added."""
    o, h, top = OBJECTS[name], half_height(name), CFG["table_top"]
    believed = OBJECTS[VISUAL_ID.get(name, name)]
    F = F_star if F_star else HANDLING[believed.cls]["target"]
    depth = 0.5
    if o.shape in ("sphere", "ellipsoid"):
        mu_w, w = 0.8 * believed.mu, believed.mass * 9.81
        ok = [a for a in np.radians(np.arange(5.0, 36.9, 0.5)) if F * (math.sin(a) - mu_w * math.cos(a)) <= 0.5 * w]
        depth = math.sin(ok[-1]) if ok else 0.1
    tip_z = max(top + 0.003, obj_z - depth * h)
    dz = (tip_z - obj_z) / h
    if o.shape == "sphere":
        w = o.size[0] * math.sqrt(max(0.0, 1 - dz * dz))
    elif o.shape == "ellipsoid":
        w = min(o.size[0], o.size[1]) * math.sqrt(max(0.0, 1 - dz * dz))
    else:                                                  # box faces, upright cylinder
        w = min(o.size[0], o.size[1]) if o.shape == "box" else o.size[0]
    bends = np.arange(0.0, 90.0, 0.5)
    B = next((b for b in bends if finger_tip(b)[0] <= w), bends[-1])
    return tip_z + CFG["palm"][2] + finger_tip(B)[1]


def sample_instance(seed, name):
    """the random instance of `name` under `seed`: identical in every engine (and independent of the other objects)"""
    o, r = OBJECTS[name], RANDOMISE
    rng = np.random.default_rng([int(seed), zlib.crc32(name.encode())])
    inst = dict(name=name, mass=o.mass * rng.uniform(1 - r["mass"], 1 + r["mass"]), mu=o.mu * rng.uniform(1 - r["mu"], 1 + r["mu"]))
    kf = rng.uniform(1 - r["k"], 1 + r["k"])
    inst["k"] = None if o.k is None else o.k * kf
    sx, sy = SPOTS[int(rng.integers(len(SPOTS)))]
    inst["xy"] = (sx + rng.uniform(-r["xy"], r["xy"]), sy + rng.uniform(-r["xy"], r["xy"]))
    inst["yaw_deg"] = float(rng.uniform(0.0, 120.0)) if o.shape in ("box", "ellipsoid") else 0.0
    inst["sensor_seed"] = int(rng.integers(2 ** 31))
    return inst


# ------------------------------------------------------------------ actuator side
class Supply:
    """pump + valve: first-order pressure response (tau [s])"""

    def __init__(self, tau=0.25):
        self.p, self.tau = 0.0, tau

    def step(self, cmd, dt):
        self.p += (cmd - self.p) * (1 - math.exp(-dt / self.tau))
        return self.p


class Actuator:
    """model of the muscle: pressure [kPa] -> free bend of one finger [deg].
    learned  : MLP [p, dp/dt, previous output] -> bend (learning/actuator_net.py), run autoregressively
    affine   : best straight line, bend = c (p - p0)+  (least squares on the identification data; p0 came out 0)
    static   : the identified steady-state curve (dead band + saturation) without dynamics, i.e. a lookup table
    greybox  : dead band + saturating curve + first-order lag, 5 identified parameters (scripts/fit_greybox.py)
    greybox_play : the same + one play operator of identified width w (6 parameters).  Note that the stand-in muscle
               itself is built from exactly these elements, so this model has the right structure by construction.
    A residual table from an in-situ dry stroke can be added (calibrate())."""

    def __init__(self, kind="learned"):
        d = json.load(open(REPO / "learning" / "actuator_net.json"))
        self.kind = kind
        self.W, self.b = [np.array(w) for w in d["W"]], [np.array(b) for b in d["b"]]
        self.xm, self.xs, self.ym, self.ys = np.array(d["x_mean"]), np.array(d["x_std"]), d["y_mean"], d["y_std"]
        self.aff = d.get("affine", {"c": d["linear_c"], "p0": 0.0})
        self.p_tau = d.get("p_filter_tau", 1e-6)
        if kind in ("greybox", "static"):
            self.gb = json.load(open(REPO / "learning" / "actuator_greybox.json"))["params"]
        if kind == "greybox_play":
            self.gb = json.load(open(REPO / "learning" / "actuator_greybox_play.json"))["params"]
        self.cal_P, self.cal_R = None, None
        self.reset()

    def reset(self):
        self.prev_p, self.prev_a, self.play = 0.0, 0.0, None
        self.flt = PFilter(self.p_tau)

    def raw(self, p, dt):
        if self.kind == "affine":
            return self.aff["c"] * max(0.0, p - self.aff["p0"])
        if self.kind in ("greybox", "static", "greybox_play"):
            g = self.gb
            x = max(0.0, p - g["p0"])
            target = g["A"] * (1 - math.exp(-x / g["P"])) + g["B"] * x
            if self.kind == "greybox_play":
                self.play = target if self.play is None else min(max(self.play, target - g["w"] / 2), target + g["w"] / 2)
                target = self.play
            if self.kind == "static":                       # steady-state curve only (a measured lookup table, no lag)
                return target
            self.prev_a += (target - self.prev_a) * (1 - math.exp(-dt / g["tau"]))
            return max(0.0, self.prev_a)
        p = self.flt(p, dt)                                 # same pressure pre-filter as in identification
        h = (np.array([p, (p - self.prev_p) / dt, self.prev_a]) - self.xm) / self.xs
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        a = float(h[0] * self.ys + self.ym)
        self.prev_p, self.prev_a = p, a
        return max(0.0, a)

    def __call__(self, p, dt):
        a = self.raw(p, dt)
        if self.cal_P is not None:
            a = max(0.0, a + float(np.interp(p, self.cal_P, self.cal_R)))
        return a


# ------------------------------------------------------------------ sensors
@dataclass
class SensorSpec:
    """what the real gripper would measure (all values are assumptions, see docs/physical_parameters.md)"""
    p_noise: float = 0.2        # pressure sensor noise [kPa, 1 sigma]
    bend_noise: float = 0.3     # bend sensor noise per finger [deg, 1 sigma]
    bend_res: float = 0.1       # bend sensor resolution [deg]
    bend_tau: float = 0.02      # bend sensor lag [s]
    skin_dead: float = 0.08     # taxel dead zone [N]
    skin_noise: float = 0.02    # taxel noise [N, 1 sigma]
    skin_gain_sd: float = 0.10  # taxel gain spread
    skin_xt: float = 0.10       # crosstalk to the neighbouring taxel
    skin_tau: float = 0.03      # skin lag [s]
    skin_sat: float = 8.0       # taxel full scale [N]
    skin_bits: int = 12
    contact_thr: float = 0.10   # skin sum that counts as contact [N]


@dataclass
class PlungerSpec:
    """firmness plunger: an 8 mm flat tip on a small linear axis at the palm centre, with a load cell and an encoder
    (like a bench-top fruit firmness tester, but non-destructive: it stops at F_probe)"""
    F_probe: float = 0.3        # stop force [N]
    v_fast: float = 0.05        # extension speed until 4 mm above the top expected from the camera [m/s]
    v_seek: float = 0.002       # then, until contact [m/s]
    v_probe: float = 0.001      # extension speed while measuring [m/s]
    max_indent: float = 0.003   # stop after this much travel past contact [m]
    enc_noise: float = 2e-6     # encoder noise [m, 1 sigma]
    enc_res: float = 1e-6       # encoder resolution [m]
    load_noise: float = 0.005   # load cell noise [N, 1 sigma]
    load_res: float = 0.001     # load cell resolution [N]
    contact_thr: float = 0.02   # load that counts as contact [N]


def probe_plunger(engine, ep, spec: PlungerSpec | None = None, rng_seed=0, record_truth=True, top_expected=None):
    """press the plunger on the top of the object and measure its stiffness (non-destructive).
    Returns dict(k=N/mm slope of force vs indentation, top=contact height [m], samples).  Logged as phase 'plunger'."""
    sp = spec or PlungerSpec()
    rng = np.random.default_rng(rng_seed)
    dt = engine.dt_ctrl
    ext, e0, rows = 0.0, None, []
    tip0 = float(np.asarray(engine.palm_target)[2]) - CFG["palm"][2] - 0.006  # retracted tip face height (6 mm tip)
    fast_to = 0.0 if top_expected is None else max(0.0, tip0 - top_expected - 0.004)
    t0 = engine.t
    while engine.t - t0 < 10.0:
        ext += (sp.v_fast if ext < fast_to else sp.v_seek if e0 is None else sp.v_probe) * dt
        engine.plunger_set(ext)
        engine.advance(1)
        x_true, f_true = engine.plunger_pos(), engine.plunger_force()
        x = np.round((x_true + rng.normal(0, sp.enc_noise)) / sp.enc_res) * sp.enc_res
        f = max(0.0, np.round((f_true + rng.normal(0, sp.load_noise)) / sp.load_res) * sp.load_res)
        if e0 is None and f > sp.contact_thr:
            e0 = x
        if e0 is not None:
            rows.append((x - e0, f, x_true, f_true))
            if f >= sp.F_probe or x - e0 > sp.max_indent:
                break
        ep.log.append(dict(t=round(engine.t, 3), phase="plunger", ext=round(x, 6), load=round(f, 4), load_true=round(f_true, 4)))
    while engine.plunger_pos() > 1e-3:                        # retract
        ext = max(0.0, ext - 0.05 * dt)
        engine.plunger_set(ext)
        engine.advance(1)
    out = dict(k=0.0, top=0.0, n=len(rows), indent_end=0.0)
    if e0 is not None and len(rows) >= 4:
        a = np.array(rows)
        m = (a[:, 1] >= 0.05) & (a[:, 1] <= sp.F_probe)
        if m.sum() >= 3:
            out["k"] = float(np.polyfit(a[m, 0] * 1e3, a[m, 1], 1)[0])     # N/mm
        out["top"] = float(tip0 - e0)
        out["indent_end"] = float(a[-1, 0] * 1e3)
        if record_truth:
            out["samples"] = a.round(7).tolist()
    return out


class Sensors:
    """pressure sensor, one bend sensor per finger, fabric skin of 24 taxels; sampled once per control period"""

    def __init__(self, seed, spec: SensorSpec | None = None, dt=0.02):
        self.s = spec or SensorSpec()
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        n = NF * NS * 2
        self.gain = 1.0 + self.rng.normal(0, self.s.skin_gain_sd, n)
        self.A = np.eye(n)
        for f in range(NF):
            for i in range(NS * 2):
                for d in (-1, 1):
                    if 0 <= i + d < NS * 2:
                        self.A[f * NS * 2 + i, f * NS * 2 + i + d] = self.s.skin_xt
        self.y_skin = np.zeros(n)
        self.y_bend = None

    def pressure(self, p):
        return float(p + self.rng.normal(0, self.s.p_noise))

    def bend(self, bend_true):
        b = np.asarray(bend_true, float)
        a = 1 - math.exp(-self.dt / self.s.bend_tau)
        self.y_bend = b.copy() if self.y_bend is None else self.y_bend + a * (b - self.y_bend)
        v = self.y_bend + self.rng.normal(0, self.s.bend_noise, NF)
        return np.round(v / self.s.bend_res) * self.s.bend_res

    def skin(self, tax_true):
        a = 1 - math.exp(-self.dt / self.s.skin_tau)
        self.y_skin += a * (self.A @ np.asarray(tax_true) * self.gain - self.y_skin)
        v = self.y_skin + self.rng.normal(0, self.s.skin_noise, len(self.y_skin))
        v = np.where(v < self.s.skin_dead, 0.0, v)
        lsb = self.s.skin_sat / 2 ** self.s.skin_bits
        return np.round(np.clip(v, 0, self.s.skin_sat) / lsb) * lsb


TAXEL_S = np.array([(k + 0.25 + 0.5 * m) * L for k in range(NS) for m in range(2)])   # taxel centre, distance from the finger base [m]
JOINT_S = np.arange(NS) * L                                                          # joint positions along the finger [m]


def lever_sum(s):
    """sum over the joints of the lever arm of a normal force applied at distance s from the finger base: sum_j (s - s_j)+"""
    return float(np.clip(s - JOINT_S, 0.0, None).sum())


TAXEL_LAMBDA = np.array([lever_sum(s) for s in TAXEL_S])                             # lever sum of a force at each taxel


def estimate_force(bend_model_deg, bend_meas_deg, k_b=CFG["k_b"], skin=None, min_skin=0.05):
    """proprioceptive grip force [N].  A normal force F_i at distance s_i along a finger bends joint j back by
    F_i (s_i - s_j)+ / k_b, so the missing total bend of the finger (model's free bend - measured bend) is
    sum_i F_i Lambda(s_i) / k_b = F * Lambda_bar / k_b, with Lambda(s) = sum_j (s - s_j)+ and Lambda_bar the force-weighted
    mean of Lambda over the contact points.  The weights come from the skin (taxel readings).  Lambda is convex in s, so
    Lambda at the centroid would underestimate Lambda_bar (and overestimate the force) when the load is spread out.
    Without skin, or when a finger's skin reads less than min_skin, the fingertip is assumed (Lambda = sum_j l_j = 0.26 m).
    Sum over the fingers, negative parts clipped."""
    miss = np.radians(np.asarray(bend_model_deg, float) - np.asarray(bend_meas_deg, float))
    lam = np.full(NF, LEVER.sum())
    if skin is not None:
        sk = np.clip(np.asarray(skin, float).reshape(NF, NS * 2), 0.0, None)
        for f in range(NF):
            w = sk[f].sum()
            if w >= min_skin:
                lam[f] = max(float(sk[f] @ TAXEL_LAMBDA / w), 0.25 * L)
    return float(np.clip(k_b * miss / lam, 0, None).sum())


# ------------------------------------------------------------------ episode
@dataclass
class Episode:
    name: str
    mode: str
    actuator: str
    log: list = field(default_factory=list)
    retries_left: int = 0
    F_star_chosen: float | None = None


@dataclass
class Rig:
    """the gripper's 'software + muscle' for one episode: plant, model, supply, sensors"""
    actuator: Actuator
    supply: Supply
    muscle: object
    sensors: Sensors
    k_b_model: float = CFG["k_b"]
    lever: str = "skin"          # where the force acts: 'skin' (taxel centroid) or 'tip' (always the fingertip)


def make_rig(inst, actuator="learned", plant=None, spec=None, dt=0.02, lever="skin"):
    """plant: None = nominal stand-in muscle (noise-free); else a muscle object with .step(p, dt)"""
    return Rig(Actuator(actuator), Supply(), plant if plant is not None else SyntheticMuscle(seed=0, noise=0.0),
               Sensors(inst["sensor_seed"], spec, dt), lever=lever)


class Controller:
    """grip-force controller (closed loop) or fixed-pressure grip (open loop), one call per control period.

    closed loop:  free closing at +12 kPa/s until the skin reports contact; at contact the command is reset to the
    measured pressure (removes the supply-lag overshoot); then integral action on the force error,
    dp_cmd/dt = clip(Ki (F* - F_fb), -15, +3) kPa/s with Ki = 4 kPa/(N s).  F_fb is the proprioceptive estimate
    ('est'), or the true force ('ideal', an ideal force sensor: upper bound).  contact='est' detects contact from the
    estimate instead of the skin (ablation)."""

    def __init__(self, mode, F_star, p_fixed=0.0, feedback="est", contact="skin", Ki=4.0, thr=SensorSpec.contact_thr):
        self.mode, self.F, self.p_fixed, self.fb, self.contact_src, self.Ki, self.thr = mode, F_star, p_fixed, feedback, contact, Ki, thr
        self.p_cmd, self.contact, self.hold = 0.0, False, 0

    def step(self, f_fb, p_meas, skin_sum, dt):
        if self.mode == "open_loop":
            self.p_cmd = self.p_fixed
            return self.p_cmd
        e = self.F - f_fb
        touched = skin_sum > self.thr if self.contact_src == "skin" else f_fb >= 0.15 * self.F
        if not self.contact and not touched:
            dp = 12.0
        else:
            if not self.contact:
                self.contact, self.p_cmd = True, p_meas
            dp = float(np.clip(self.Ki * e, -15.0, 3.0))
        self.p_cmd = float(np.clip(self.p_cmd + dp * dt, 0.0, 80.0))
        self.hold = self.hold + 1 if (self.contact and abs(e) < 0.15 * self.F) else 0
        return self.p_cmd


def _measure(engine, name, rig, ctrl, state):
    """one sample of every sensor + the truths, and the model's free bend for the current pressure"""
    s = rig.sensors
    tax_true = engine.taxels_true()
    state["tax"] = s.skin(tax_true)
    state["bend_meas"] = s.bend(engine.bend_true())
    state["f_est"] = estimate_force(state["bend_model"], state["bend_meas"], rig.k_b_model,
                                    skin=state["tax"] if rig.lever == "skin" else None)
    state["f_true"] = engine.force_true(name)
    state["tax_true"] = tax_true


def _actuate(engine, rig, state, p_cmd, dt):
    """pressure command -> supply -> (sensor) -> model free bend; muscle -> fingers"""
    p = rig.supply.step(p_cmd, dt)
    state["p"] = rig.sensors.pressure(p)
    state["bend_model"] = rig.actuator(state["p"], dt)
    state["bend_free"] = rig.muscle.step(p, dt)
    engine.apply(state["bend_free"])


def _record(ep, engine, state, phase, **extra):
    ep.log.append(dict(t=round(engine.t, 3), phase=phase, p_cmd=round(state.get("p_cmd", 0.0), 2), p=round(state.get("p", 0.0), 2),
                       bend_model=round(state.get("bend_model", 0.0), 2), bend_free=round(state.get("bend_free", 0.0), 2), bend_meas=np.round(state["bend_meas"], 2).tolist(),
                       bend_true=np.round(engine.bend_true(), 3).tolist(), f_est=round(state["f_est"], 3), f_true=round(state["f_true"], 3),
                       tax=np.round(state["tax"], 4).tolist(), tax_true=np.round(state["tax_true"], 4).tolist(), **extra))


def _move(engine, ep, rig, state, target, seconds, phase, ctrl=None, vent=False, guard=None):
    """palm moves with a smooth-step profile; every control period the grip is updated (ctrl) and everything logged.
    guard: stop the palm (hold position) as soon as the skin total exceeds this value [N] (guarded placing)"""
    dt = engine.dt_ctrl
    start = np.array(engine.palm_target, float)
    n = max(1, int(round(seconds / dt)))
    stopped = False
    for i in range(n):
        a = (i + 1) / n
        if guard is not None and float(np.sum(state["tax"])) > guard:
            stopped = True
        if not stopped:
            engine.set_palm(start + a * a * (3 - 2 * a) * (np.asarray(target) - start))
        p_cmd = 0.0 if vent else (ctrl.step(state["f_fb"], state["p"], float(np.sum(state["tax"])), dt) if ctrl else state.get("p_cmd", 0.0))
        state["p_cmd"] = p_cmd
        _actuate(engine, rig, state, p_cmd, dt)
        engine.advance(1)
        _measure(engine, ep.name, rig, ctrl, state)
        state["f_fb"] = state["f_true"] if (ctrl is not None and ctrl.fb == "ideal") else state["f_est"]
        _record(ep, engine, state, phase)
        if ctrl is not None and state.get("skin_hold"):
            # lost the object: the force estimate collapses while the hand keeps closing -> stop squeezing and abort
            # (the skin alone is not enough: when the hand closes on nothing, the fingertips press on each other)
            lost = state["f_est"] < 0.5 * ctrl.F
            state["lost_n"] = state.get("lost_n", 0) + 1 if lost else 0
            if state["lost_n"] >= 3:
                state["lost"] = True
                return


def calibrate_dry_stroke(engine, rig, levels=(8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0), settle=1.2, bend_max=24.0):
    """in-situ calibration with nothing in the hand: hold each pressure until the supply and the muscle have settled,
    average the measured bend over the last 0.2 s, and store measured - model as an additive table (vented after).
    The fingertips meet at about 28 deg of bend in air, so the staircase stops once the measured bend passes bend_max;
    above the last level the residual is held constant."""
    dt = engine.dt_ctrl
    state = {}
    P, R = [], []
    rig.actuator.cal_P = None
    for lv in levels:
        meas, mod = [], []
        for k in range(int(settle / dt)):
            _actuate(engine, rig, state, lv, dt)
            engine.advance(1)
            b = rig.sensors.bend(engine.bend_true())
            if k >= int(settle / dt) - 10:
                meas.append(b.mean())
                mod.append(state["bend_model"])
        if np.mean(meas) > bend_max:
            break
        P.append(state["p"])
        R.append(float(np.mean(meas) - np.mean(mod)))
    for _ in range(int(0.8 / dt)):
        _actuate(engine, rig, state, 0.0, dt)
        engine.advance(1)
    rig.actuator.reset()
    rig.supply.p = 0.0
    if P:
        rig.actuator.cal_P, rig.actuator.cal_R = np.array([0.0] + P), np.array([R[0]] + R)
    return dict(P=[round(x, 2) for x in P], residual_deg=[round(x, 2) for x in R])


def attempt(engine, ep, rig, F_star, p_fixed=0.0, feedback="est", contact="skin", probe_fn=None, F_probe=0.8, slot=(0.0, 0.0),
            probe="finger", plunger_spec=None):
    """approach -> descend -> [probe] -> grip (open loop: fixed pressure; closed loop: force control) -> lift -> (slip check)
    -> transport -> lower -> release.  Returns 'slip' or 'done'.
    probe='finger': squeeze gently to F_probe with the fingers, then probe_fn(engine, ep) chooses F* (fabric skin + bend);
    probe='plunger': press the firmness plunger on the top first, then probe_fn(engine, ep) chooses F* (ep.plunger)."""
    name, top, dt = ep.name, CFG["table_top"], engine.dt_ctrl
    h = half_height(name)
    state = dict(p=0.0, p_cmd=0.0, bend_model=0.0, f_fb=0.0)
    rig.actuator.reset()
    rig.supply.p = 0.0
    _measure(engine, name, rig, None, state)
    o = engine.obj_pos(name)
    _move(engine, ep, rig, state, [o[0], o[1], top + 0.22], 1.0, "approach", vent=True)
    o = engine.obj_pos(name)                                   # perception update right before descending
    _move(engine, ep, rig, state, [o[0], o[1], top + 0.22], 0.3, "align", vent=True)
    engine.on_frame("pregrasp", ep)
    grasp_z = grasp_height(name, o[2], F_star)
    _move(engine, ep, rig, state, [o[0], o[1], grasp_z], 0.8, "descend", vent=True)
    if probe_fn is not None and probe in ("plunger", "both"):
        ep.plunger = probe_plunger(engine, ep, plunger_spec, rng_seed=engine.inst["sensor_seed"] + 1,
                                   top_expected=o[2] + h)
        if probe == "plunger":
            F_star = float(probe_fn(engine, ep))
            ep.F_star_chosen = F_star
            probe_fn = None
            z = grasp_height(name, o[2], F_star)           # the depth of the grasp depends on F*: re-plan it
            if abs(z - grasp_z) > 5e-4:
                grasp_z = z
                _move(engine, ep, rig, state, [o[0], o[1], grasp_z], 0.4, "descend", vent=True)

    probing = ep.mode == "closed_loop" and probe_fn is not None
    ctrl = Controller(ep.mode, F_probe if probing else F_star, p_fixed, feedback, contact, thr=rig.sensors.s.contact_thr)
    state["f_fb"] = state["f_est"]
    for k in range(int((8.0 if probing else 5.0) / dt)):
        p_cmd = ctrl.step(state["f_fb"], state["p"], float(np.sum(state["tax"])), dt)
        state["p_cmd"] = p_cmd
        _actuate(engine, rig, state, p_cmd, dt)
        engine.advance(1)
        _measure(engine, name, rig, ctrl, state)
        state["f_fb"] = state["f_true"] if feedback == "ideal" else state["f_est"]
        _record(ep, engine, state, "grip", probing=probing, contact=ctrl.contact)
        if probing and ctrl.hold > 10:
            F_star = float(probe_fn(engine, ep))
            ep.F_star_chosen = F_star
            ctrl.F, ctrl.hold, probing = F_star, 0, False
        if ep.mode == "closed_loop" and not probing and ctrl.hold > 20:
            break
        if ep.mode == "open_loop" and k * dt > 2.0:
            break
    engine.on_frame("grasp", ep)
    ep.F_last = ctrl.F
    tx, ty = CFG["tray"][0] + slot[0], CFG["tray"][1] + slot[1]
    place_z = (top + 0.010 + 0.012 + h) - (o[2] - grasp_z)    # object's bottom ~12 mm above the tray floor
    state["skin_hold"] = max(float(np.sum(state["tax"])), 0.1)
    _move(engine, ep, rig, state, [o[0], o[1], top + 0.20], 1.0, "lift", ctrl)
    engine.on_frame("lift", ep)
    if engine.obj_pos(name)[2] < top + 0.08:                    # the object stayed on the table: slip
        _move(engine, ep, rig, state, [o[0], o[1], top + 0.20], 0.4, "regrasp", vent=True)
        return "slip"
    for tgt, sec, ph, guard in [([tx, ty, top + 0.20], 2.0, "transport", None),           # peak 0.9 m/s^2, 0.45 m/s
                                ([tx, ty, place_z], 0.8, "lower", "skin")]:
        if state.get("lost"):
            break
        # guarded placing: if the object slid down in the hand and meets the tray early, the skin load jumps -> stop there
        _move(engine, ep, rig, state, tgt, sec, ph, ctrl, guard=float(np.sum(state["tax"])) + 0.6 if guard else None)
    if state.get("lost"):                                       # dropped in transport: open the hand, move away
        _move(engine, ep, rig, state, list(engine.palm_target), 0.6, "release", vent=True)
        _move(engine, ep, rig, state, [tx, ty, top + 0.22], 0.6, "retract", vent=True)
        return "done"
    _move(engine, ep, rig, state, list(engine.palm_target), 0.8, "release", vent=True)
    _move(engine, ep, rig, state, [tx, ty, top + 0.22], 0.6, "retract", vent=True)
    return "done"


HELD = ("grip", "lift", "transport", "lower")


def outcome(engine, ep):
    """labels from the simulator (no human): success = lifted, in the tray and never crushed.
    crushed: the true grip force while the object is held, smoothed over 3 control samples (60 ms), ever exceeded the
    allowable force (any attempt).  no_contact: the fingers never touched the object (max true force < 0.05 N)."""
    name = ep.name
    pos = engine.obj_pos(name)
    top, (tx, ty) = CFG["table_top"], CFG["tray"]
    in_tray = abs(pos[0] - tx) < 0.09 and abs(pos[1] - ty) < 0.07 and pos[2] < top + 0.07
    held = np.array([e["f_true"] for e in ep.log if e["phase"] in HELD])
    held_f = np.convolve(held, np.ones(3) / 3, mode="same") if len(held) >= 3 else held
    peak = float(held_f.max()) if len(held_f) else 0.0
    est = np.array([e["f_est"] for e in ep.log if e["phase"] in HELD])
    fmax = OBJECTS[name].fmax
    touched = max([e.get("f_true", 0.0) for e in ep.log] or [0.0]) >= 0.05
    lifted = getattr(ep, "lifted", False)
    crushed = peak > fmax
    # dropped: after the lift the object left the fingers before the release (5 samples = 0.1 s without contact)
    carried = np.array([e["f_true"] for e in ep.log if e["phase"] in ("transport", "lower")])
    free = np.convolve((carried < 0.02).astype(float), np.ones(5), mode="valid") if len(carried) >= 5 else np.zeros(1)
    fell = bool(lifted and free.max() >= 5)
    in_tray = in_tray and not fell
    # failure mode (one label, in order of what happened first): no contact / slip at lift / dropped in transport / crushed
    mode = ("no_contact" if not touched else "slip" if not lifted else "dropped" if not in_tray else "crushed" if crushed else "ok")
    return dict(object=name, cls=OBJECTS[name].cls, mode=ep.mode, actuator=ep.actuator, lifted=bool(lifted), in_tray=bool(in_tray),
                crushed=bool(crushed), no_contact=not touched, failure=mode, success=bool(lifted and in_tray and not crushed),
                peak_force_true=round(peak, 3), peak_force_est=round(float(est.max()) if len(est) else 0.0, 3), max_safe=fmax,
                F_star_final=round(getattr(ep, "F_last", 0.0), 3))


def run_episode(engine, name, mode="closed_loop", actuator="learned", F_target=None, p_fixed=None, max_retries=3, feedback="est",
                contact="skin", plant=None, spec=None, probe_fn=None, escalate=None, F_probe=0.8, calibrate=False, slot=(0.0, 0.0), seed=None,
                probe="finger", plunger_spec=None, lever="skin"):
    """one pick-and-place with up to `max_retries` regrasps after a slip (open loop gets the same budget).
    closed loop: after a slip F* grows x1.3, capped at the allowable force of the class the planner believes in
    (escalate(F, ep) replaces that rule).  open loop: the pressure grows x1.3."""
    inst = engine.inst
    ep = Episode(name, mode, actuator)
    rig = make_rig(inst, actuator, plant, spec, engine.dt_ctrl, lever)
    if calibrate:
        ep.calibration = calibrate_dry_stroke(engine, rig)
    believed = OBJECTS[VISUAL_ID.get(name, name)].cls
    F = HANDLING[believed]["target"] if F_target is None else F_target
    F_target_used = F
    p = p_fixed if p_fixed is not None else 0.0
    attempts = 0
    while True:
        attempts += 1
        res = attempt(engine, ep, rig, F, p, feedback, contact, probe_fn if attempts == 1 else None, F_probe, slot, probe, plunger_spec)
        ep.lifted = res == "done"
        if res == "done" or attempts > max_retries:
            break
        if ep.F_star_chosen is not None and attempts == 1:
            F = ep.F_star_chosen
        if mode == "open_loop":
            p = min(80.0, p * 1.3)
        elif escalate is not None:
            F = float(escalate(F, ep))
        else:
            F = max(F, min(F * 1.3, 0.8 * HANDLING[believed]["limit"]))   # never lower the force on a re-grasp
    r = outcome(engine, ep)
    t_end = ep.log[-1]["t"] if ep.log else 0.0
    r.update(attempts=attempts, p_fixed=p_fixed if mode == "open_loop" else None, feedback=feedback, contact_src=contact,
             cycle_time_s=round(t_end - (ep.log[0]["t"] if ep.log else 0.0), 2), mass=round(inst["mass"], 4), mu=round(inst["mu"], 3),
             k_contact=None if inst["k"] is None else round(inst["k"], 1))
    grip = [e for e in ep.log if e["phase"] in HELD]
    if grip:
        ft, fe = np.array([e["f_true"] for e in grip]), np.array([e["f_est"] for e in grip])
        on = ft > 0.05
        r["est_mae_contact"] = round(float(np.mean(np.abs(ft[on] - fe[on]))), 4) if on.any() else None
        r["est_peak_err"] = round(float(fe.max() - ft.max()), 4)
        held = ft > 0.5                                     # bias of the estimate while the object is really held
        r["est_ratio_held"] = round(float(np.median(fe[held] / ft[held])), 4) if held.any() else None
        bm = np.array([np.mean(e["bend_model"]) for e in grip])        # muscle model's free bend - the plant's (truth)
        bf = np.array([np.mean(e["bend_free"]) for e in grip])
        r["free_bend_err_held"] = round(float(np.median(bm[held] - bf[held])), 3) if held.any() else None
        tr = [e["f_true"] for e in ep.log if e["phase"] == "transport"]
        F_hold = getattr(ep, "F_last", None) or F_target_used   # the target the loop was tracking while carrying
        r["hold_over_target"] = round(float(np.median(tr)) / F_hold, 4) if tr and F_hold else None
    if seed is not None:
        r["seed"] = seed
    return r, ep
