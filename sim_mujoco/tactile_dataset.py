"""Mass-produce auto-annotated multimodal grasp data (vision + tactile skin + pressure + proprioception) in MuJoCo,
train vision / touch / fusion classifiers on it, and run the look-alike benchmark.

  python sim_mujoco/tactile_dataset.py generate --per-object 40      # -> data/tactile_dataset/
  python sim_mujoco/tactile_dataset.py train                         # -> learning/tactile_classifier.json, results/tactile_classifier.json
  python sim_mujoco/tactile_dataset.py bench --seeds 10              # -> results/benchmark_lookalike.json

Every label is produced by the simulator itself (no human annotation): object identity and hidden physics,
segmentation masks / boxes / depth for each image, per-taxel contact force ground truth next to the noisy skin
reading, true grip force next to the estimate, and the outcome (success / crushed / slipped) of the grasp.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import mujoco  # noqa: E402

import softtwin_mj as S  # noqa: E402
import tactile as T  # noqa: E402
import twin_core as C  # noqa: E402

REPO = S.REPO
DATA = REPO / "data" / "tactile_dataset"
ALL_OBJECTS = S.ALL_OBJECTS
IMG_H, IMG_W = 144, 192
PHASES = ("approach", "align", "descend", "inflate", "lift", "transport", "lower", "regrasp", "release", "retract")


class Cameras:
    """renders wrist (rgb, depth, segmentation) and scene (rgb, segmentation) views at the episode's key moments"""

    def __init__(self, cell: S.Cell, name: str, renderer=None, tags=("pregrasp", "grasp", "lift")):
        self.c, self.name, self.tags = cell, name, tags
        # one OSMesa context per process: reuse a caller's renderer (e.g. the video) instead of opening a second one
        self.r = renderer if renderer is not None else mujoco.Renderer(cell.m, IMG_H, IMG_W)
        self.gid = cell.m.geom(f"{name}_g").id
        self.shots, self.fallbacks = {}, 0
        cell.frame_cb = self

    def _render(self, cam, depth=False, seg=False):
        r = self.r
        r.update_scene(self.c.d, camera=cam)
        if depth:
            r.enable_depth_rendering()
            out = r.render().copy()
            r.disable_depth_rendering()
            return out
        if seg:
            fl = r.scene.flags
            keep = {f: fl[f] for f in (mujoco.mjtRndFlag.mjRND_REFLECTION, mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_SKYBOX, mujoco.mjtRndFlag.mjRND_FOG)}
            for f in keep:
                fl[f] = 0   # reflections / shadows blend colours and corrupt the id image
            r.enable_segmentation_rendering()
            try:
                out = r.render().copy()
            except IndexError:   # rare corrupted id pixel under OSMesa -> fall back to a difference mask
                out = None
            finally:
                r.disable_segmentation_rendering()
                for f, v in keep.items():
                    fl[f] = v
            if out is None:
                self.fallbacks += 1
                rgb = r.render().astype(int)
                a0 = self.c.m.geom_rgba[self.gid, 3]
                self.c.m.geom_rgba[self.gid, 3] = 0.0
                r.update_scene(self.c.d, camera=cam)
                hid = r.render().astype(int)
                self.c.m.geom_rgba[self.gid, 3] = a0
                return np.abs(rgb - hid).sum(-1) > 60
            return (out[..., 0] == self.gid) & (out[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        return r.render().copy()

    def __call__(self, cell, tag, ep):
        if tag in self.shots or tag not in self.tags:      # keep the first attempt's views
            return
        cam = "wrist" if tag == "pregrasp" else "close"
        rgb, mask = self._render(cam), self._render(cam, seg=True)
        shot = dict(rgb=rgb, mask=mask)
        if cam == "wrist":
            shot["depth"] = self._render(cam, depth=True).astype(np.float16)
        self.shots[tag] = shot


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)] if len(xs) else None


def hidden_physics(cell, name):
    inst = cell.inst
    g = cell.m.geom(f"{name}_g")
    return dict(mass=round(float(inst["mass"]), 4), mu=round(float(inst["mu"]), 3), k_N_per_m=None if inst["k"] is None else round(inst["k"], 1),
                solref_tc=round(float(g.solref[0]), 5), yaw_deg=round(inst["yaw_deg"], 1))


def vision_feat(cams):
    s = cams.shots.get("pregrasp")
    return T.vision_features(s["rgb"], s["mask"]) if s else np.zeros(len(T.VISION_FEATS))


# ------------------------------------------------------------------ one annotated episode
def run_annotated(args):
    idx, name, seed, F_rand, out_dir = args
    cell = S.Cell(seed=seed, objects=(name,))
    cams = Cameras(cell, name)
    # dataset episodes: plunger probe, finger probe, then a grasp with a random target force (covers crush and slip),
    # no regrasp (each episode is one labelled grasp)
    r, ep = S.run_episode(cell, name, probe="both", probe_fn=lambda c, e: F_rand, max_retries=0)
    feats_t, feats_p, feats_v = T.probe_features(ep), T.plunger_features(ep), vision_feat(cams)
    log = [e for e in ep.log if e["phase"] in ("grip", "lift", "transport", "lower")]
    pl = [e for e in ep.log if e["phase"] == "plunger"]
    ts = dict(
        t=np.array([e["t"] for e in log], np.float32), phase=np.array([e["phase"] for e in log]),
        probing=np.array([bool(e.get("probing", False)) for e in log], bool),
        p_cmd=np.array([e["p_cmd"] for e in log], np.float32), p=np.array([e["p"] for e in log], np.float32),
        bend_model=np.array([e["bend_model"] for e in log], np.float32), bend_meas=np.array([e["bend_meas"] for e in log], np.float32),
        bend_true=np.array([e["bend_true"] for e in log], np.float32),
        f_est=np.array([e["f_est"] for e in log], np.float32), f_true=np.array([e["f_true"] for e in log], np.float32),
        skin=np.array([e["tax"] for e in log], np.float32), skin_true=np.array([e["tax_true"] for e in log], np.float32),
        plunger_ext=np.array([e["ext"] for e in pl], np.float32), plunger_load=np.array([e["load"] for e in pl], np.float32),
        plunger_load_true=np.array([e["load_true"] for e in pl], np.float32),
    )
    imgs = {}
    for tag, sh in cams.shots.items():
        imgs[f"{tag}_rgb"] = sh["rgb"]
        imgs[f"{tag}_mask"] = sh["mask"]
        if "depth" in sh:
            imgs[f"{tag}_depth"] = sh["depth"]
    cams.r.close()
    fn = f"ep_{idx:05d}.npz"
    np.savez_compressed(Path(out_dir) / "episodes" / fn, **ts, **imgs, probe_features=feats_t, plunger_features=feats_p, vision_features=feats_v)
    meta = dict(
        id=idx, file=f"episodes/{fn}", seed=seed, object=name, looks_like=S.VISUAL_ID.get(name, name),
        true_class=S.OBJECTS[name].cls, prior_class=S.VISUAL_CLASS[S.VISUAL_ID.get(name, name)], max_safe_force=S.OBJECTS[name].fmax,
        physics=hidden_physics(cell, name), F_probe=0.8, plunger_F=C.PlungerSpec.F_probe, F_target=round(F_rand, 3),
        outcome=dict(success=r["success"], crushed=r["crushed"], lifted=r["lifted"], slipped=(r["failure"] == "slip"), dropped=(r["failure"] == "dropped"),
                     no_contact=r["no_contact"], in_tray=r["in_tray"], failure=r["failure"],
                     peak_force_true=r["peak_force_true"], peak_force_est=r["peak_force_est"]),
        boxes={tag: bbox(sh["mask"]) for tag, sh in cams.shots.items()},
        probe_features=dict(zip(T.PROBE_FEATS, np.round(feats_t, 4).tolist())),
        plunger_features=dict(zip(T.PLUNGER_FEATS, np.round(feats_p, 5).tolist())),
        vision_features=dict(zip(T.VISION_FEATS, np.round(feats_v, 4).tolist())),
        n_steps=len(log), mask_fallbacks=cams.fallbacks,
    )
    return meta


def _safe_job(job):
    """run one episode; the meta is written next to the npz so generation can resume after a crashed worker"""
    out = Path(job[-1]) / "episodes" / f"ep_{job[0]:05d}.json"
    if out.exists():
        return json.loads(out.read_text())
    m = run_annotated(job)
    out.write_text(json.dumps(m))
    return m


def generate(per_object, out_dir=DATA, workers=2, seed0=1000, timeout=120):
    out_dir = Path(out_dir)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed0)
    jobs = []
    for i in range(per_object):
        for j, name in enumerate(ALL_OBJECTS):
            idx = i * len(ALL_OBJECTS) + j
            jobs.append((idx, name, seed0 + idx, float(rng.uniform(0.5, 4.0)), str(out_dir)))   # seeds 1000+: disjoint from the benchmarks (0-29)
    t0 = time.time()
    metas, failed, todo = {}, [], list(jobs)
    while todo:
        pool = Pool(workers, maxtasksperchild=20)
        pend = [(jb, pool.apply_async(_safe_job, (jb,))) for jb in todo]
        todo, broken = [], False
        for jb, ar in pend:
            if broken:
                todo.append(jb)
                continue
            try:
                metas[jb[0]] = ar.get(timeout=timeout)
            except Exception as e:  # crashed / hung worker (OSMesa) -> skip this episode, restart the pool for the rest
                failed.append((jb[0], jb[1], repr(e)[:80]))
                broken = True
            if len(metas) % 50 == 0:
                print(f"{len(metas)}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
        pool.terminate()
    metas = [metas[k] for k in sorted(metas)]
    with open(out_dir / "index.jsonl", "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    import pandas as pd
    flat = pd.json_normalize(metas)
    flat = flat.drop(columns=[c for c in flat.columns if c.startswith("boxes.")])
    flat.to_parquet(out_dir / "index.parquet")
    card = dataset_card(metas, time.time() - t0) + f"- skipped episodes (renderer crash): {len(failed)}\n"
    (out_dir / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    print(card, failed)
    return metas


def dataset_card(metas, sec):
    n = len(metas)
    steps = sum(m["n_steps"] for m in metas)
    o = {k: np.mean([m["outcome"][k] for m in metas]) for k in ("success", "crushed", "slipped", "dropped", "no_contact")}
    return f"""# SoftTwin tactile grasp dataset (synthetic, auto-annotated)

- episodes: {n} ({len(ALL_OBJECTS)} object types incl. 2 look-alikes), generated in {sec/60:.1f} min on 2 CPU cores; seeds 1000 and up
  (disjoint from the benchmark seeds 0-29)
- every episode: firmness plunger probe (load + encoder, 50 Hz), finger probe to 0.8 N, then one grasp with a random target force
  F* ~ U(0.5, 4.0) N and no regrasp
- control-rate samples (50 Hz, grip to lower): {steps}; each has pressure command / pressure / model free bend / measured and true
  bend of the 3 fingers / estimated and true grip force / 24 skin taxels (sensor model) / 24 true taxel forces
- images per episode: wrist RGB + depth + segmentation mask (pre-grasp), scene RGB + mask (grasp, lift); {IMG_W}x{IMG_H}
- labels per episode: object, looks-like, true handling class, visual prior class, allowable force, hidden physics
  (mass, friction, contact stiffness [N/m], yaw), target force, outcome (success {o['success']:.0%}, crushed {o['crushed']:.0%},
  slipped {o['slipped']:.0%}, dropped {o['dropped']:.0%}, no contact {o['no_contact']:.0%}), bounding boxes, feature vectors
- files: `index.jsonl` / `index.parquet` (one row per episode), `episodes/ep_XXXXX.npz`
- all annotations come from the simulator; sensor values pass through the sensor models of `sim_mujoco/twin_core.py`
  (assumptions for a textile piezoresistive skin, a bend sensor and a small load cell + encoder, not calibrated devices)
"""


# ------------------------------------------------------------------ training
def load_index(d=DATA):
    return [json.loads(line) for line in open(Path(d) / "index.jsonl")]


FEATURE_SETS = {"vision": ("vision",), "finger": ("vision", "probe"), "plunger": ("vision", "plunger"), "all": ("vision", "probe", "plunger")}


def feature_sets(metas):
    blocks = {"vision": (T.VISION_FEATS, "vision_features"), "probe": (T.PROBE_FEATS, "probe_features"),
              "plunger": (T.PLUNGER_FEATS, "plunger_features")}
    B = {k: np.array([[m[key][f] for f in names] for m in metas]) for k, (names, key) in blocks.items()}
    X = {fs: np.concatenate([B[b] for b in parts], 1) for fs, parts in FEATURE_SETS.items()}
    y = np.array([T.CLASSES.index(m["true_class"]) for m in metas])
    return X, y


def feature_names(fs):
    names = {"vision": T.VISION_FEATS, "probe": T.PROBE_FEATS, "plunger": T.PLUNGER_FEATS}
    return [n for b in FEATURE_SETS[fs] for n in names[b]]


def _auc(y, s):
    """ROC AUC via the rank-sum statistic"""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    r = np.empty(len(s))
    r[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def train(d=DATA):
    """classifiers (vision / +finger / +plunger / all) and the learned force window on ALL dataset episodes (the evaluation
    is the online benchmark on other seeds); 5-fold CV accuracies are in results/metrics.json (analysis_stats.py)"""
    metas = load_index(d)
    X, y = feature_sets(metas)
    models, res = {}, {"n_train": len(metas), "classes": list(T.CLASSES), "models": {}}
    for k, Xk in X.items():
        models[k] = T.Softmax().fit(Xk, y).to_json()
        res["models"][k] = dict(features=feature_names(k))
    Fd = np.array([m["F_target"] for m in metas])
    cr = np.array([m["outcome"]["crushed"] for m in metas])
    sl = np.array([m["outcome"]["slipped"] for m in metas])
    fw = T.ForceWindow().fit(X["plunger"], Fd, cr, sl)
    fw.features = feature_names("plunger")
    (REPO / "learning").mkdir(exist_ok=True)
    json.dump(fw.to_json(), open(REPO / "learning" / "force_window.json", "w"))
    json.dump(models, open(REPO / "learning" / "tactile_classifier.json", "w"))
    json.dump(res, open(REPO / "results" / "tactile_classifier.json", "w"), indent=1)
    return res


def train_loo(d=DATA):
    """leave-one-object-out force windows: for each object, a window fitted without any episode of that object"""
    metas = load_index(d)
    X, _ = feature_sets(metas)
    Fd = np.array([m["F_target"] for m in metas])
    cr = np.array([m["outcome"]["crushed"] for m in metas])
    sl = np.array([m["outcome"]["slipped"] for m in metas])
    objs = np.array([m["object"] for m in metas])
    out = {o: T.ForceWindow().fit(X["plunger"][objs != o], Fd[objs != o], cr[objs != o], sl[objs != o]).to_json() for o in ALL_OBJECTS}
    json.dump(out, open(REPO / "learning" / "force_window_loo.json", "w"))
    return out


# ------------------------------------------------------------------ look-alike benchmark (closed loop, online)
POLICIES = ("visual_prior", "soft_start", "finger_clf", "plunger_clf", "plunger_clf_tip", "plunger_clf_mass", "force_window", "force_window_loo",
            "oracle")
SOFT_START = min(C.TARGET_FORCE.values())     # N: start every object at the lowest class target, raise x1.3 after each slip
CAL = True           # the deployed gripper runs its dry-stroke calibration (see twin_core.calibrate_dry_stroke) before grasping


def _cap(cls):
    return 0.8 * C.HANDLING[cls]["limit"]


def _bench_job(args):
    name, seed, policy = args
    models = {k: T.Softmax.from_json(v) for k, v in json.load(open(REPO / "learning" / "tactile_classifier.json")).items()}
    fw = T.ForceWindow.from_json(json.load(open(REPO / "learning" / "force_window.json")))
    cell = S.Cell(seed=seed, objects=(name,))
    cams = Cameras(cell, name)
    prior = S.VISUAL_CLASS[S.VISUAL_ID.get(name, name)]
    info = {}
    if policy == "visual_prior":
        r, ep = S.run_episode(cell, name, F_target=C.TARGET_FORCE[prior], calibrate=CAL)
    elif policy == "soft_start":
        r, ep = S.run_episode(cell, name, F_target=SOFT_START, escalate=lambda F, e: min(F * 1.3, _cap(prior)), calibrate=CAL)
    elif policy == "oracle":
        tc = S.OBJECTS[name].cls
        r, ep = S.run_episode(cell, name, F_target=C.TARGET_FORCE[tc], escalate=lambda F, e: min(F * 1.3, _cap(tc)), calibrate=CAL)
    elif policy.startswith("force_window"):       # learned window: F* straight from p(crush), p(slip); no classes
        if policy == "force_window_loo":           # window that never saw this object
            fw = T.ForceWindow.from_json(json.load(open(REPO / "learning" / "force_window_loo.json"))[name])

        def choose(c, e):
            info["x"] = np.r_[vision_feat(cams), T.plunger_features(e)]
            info["F_chosen"] = fw.choose(info["x"])
            return info["F_chosen"]
        r, ep = S.run_episode(cell, name, probe="plunger", probe_fn=choose,
                              escalate=lambda F, e: fw.escalate(info["x"], F) if "x" in info else min(F * 1.3, _cap(prior)), calibrate=CAL)
        info.pop("x", None)
    else:
        kind = {"finger_clf": "finger", "plunger_clf": "plunger", "plunger_clf_tip": "plunger", "plunger_clf_mass": "plunger"}[policy]
        vis_type = S.VISUAL_ID.get(name, name)        # the type the camera recognises (its nominal mass and friction)

        def choose(c, e):
            x = {"finger": np.r_[vision_feat(cams), T.probe_features(e)], "plunger": np.r_[vision_feat(cams), T.plunger_features(e)]}[kind]
            pr = models[kind].predict_proba(x)[0]
            cls = T.CLASSES[int(pr.argmax())]
            info.update(pred=cls, conf=round(float(pr.max()), 3))
            if policy == "plunger_clf_mass":
                return C.weight_floor_target(vis_type, cls)
            return C.TARGET_FORCE[cls]
        # after a slip, raise F* only up to 80 % of the allowable force of the class touch measured (handling table)
        r, ep = S.run_episode(cell, name, F_target=C.TARGET_FORCE[prior], probe="plunger" if kind == "plunger" else "finger", probe_fn=choose,
                              escalate=lambda F, e: max(F, min(F * 1.3, _cap(info.get("pred", prior)))),
                              lever="tip" if policy.endswith("_tip") else "skin", calibrate=CAL)
    cams.r.close()
    return dict(r, policy=policy, prior=prior, **info)


def bench(seeds, workers=2, policies=POLICIES, out=REPO / "results" / "benchmark_lookalike.json"):
    jobs = [(n, s, p) for s in seeds for n in ALL_OBJECTS for p in policies]
    rows, todo, tries = [], list(jobs), {}
    while todo:
        pool = Pool(workers, maxtasksperchild=20)
        pend = [(jb, pool.apply_async(_bench_job, (jb,))) for jb in todo]
        todo, broken = [], False
        for jb, ar in pend:
            if broken:
                todo.append(jb)
                continue
            try:
                rows.append(ar.get(timeout=180))
            except Exception as e:  # crashed renderer -> retry this job in a fresh pool (at most twice)
                tries[jb] = tries.get(jb, 0) + 1
                print("retry", jb, repr(e)[:60], flush=True)
                if tries[jb] > 2:
                    raise
                todo.append(jb)
                broken = True
        pool.terminate()
    summ = {}
    for p in policies:
        rs = [r for r in rows if r["policy"] == p]
        la = [r for r in rs if r["object"] in S.VISUAL_ID]
        summ[p] = dict(n=len(rs), success=int(sum(r["success"] for r in rs)), crushed=int(sum(r["crushed"] for r in rs)),
                       failure={f: int(sum(r["failure"] == f for r in rs)) for f in ("no_contact", "slip", "dropped", "crushed")},
                       attempts=round(float(np.mean([r["attempts"] for r in rs])), 2), cycle_time_s=round(float(np.mean([r["cycle_time_s"] for r in rs])), 2),
                       lookalike_success=int(sum(r["success"] for r in la)), lookalike_n=len(la), lookalike_crushed=int(sum(r["crushed"] for r in la)),
                       by_object={o: [int(sum(r["success"] for r in rs if r["object"] == o)), int(sum(r["crushed"] for r in rs if r["object"] == o))] for o in ALL_OBJECTS})
        print(p, json.dumps(summ[p]), flush=True)
    res = dict(engine=f"MuJoCo {mujoco.__version__}", seeds=list(seeds), summary=summ, rows=rows)
    json.dump(res, open(out, "w"), indent=1, default=str)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    g = sp.add_parser("generate")
    g.add_argument("--per-object", type=int, default=40)
    g.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    sp.add_parser("train")
    sp.add_parser("train-loo")
    b = sp.add_parser("bench")
    b.add_argument("--seeds", type=int, default=10)
    b.add_argument("--seed0", type=int, default=0, help="first seed (30: the held-out confirmation run)")
    b.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    b.add_argument("--out", default=str(REPO / "results" / "benchmark_lookalike.json"))
    b.add_argument("--policies", default=",".join(POLICIES))
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.per_object, workers=a.workers)
    elif a.cmd == "train":
        train()
    elif a.cmd == "train-loo":
        train_loo()
    elif a.cmd == "bench":
        bench(range(a.seed0, a.seed0 + a.seeds), workers=a.workers, out=a.out, policies=tuple(a.policies.split(",")))
