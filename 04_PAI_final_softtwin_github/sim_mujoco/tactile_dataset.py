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

REPO = S.REPO
DATA = REPO / "data" / "tactile_dataset"
ALL_OBJECTS = tuple(S.BASE_OBJECTS) + ("tomato_overripe", "sample_strawberry")
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
    g = cell.m.geom(f"{name}_g")
    return dict(mass=round(float(cell.m.body(name).mass[0]), 4), mu=round(float(g.friction[0]), 3),
                tc=round(float(g.solref[0]), 4), dmin=round(float(g.solimp[0]), 3), width=round(float(g.solimp[2]), 4))


def vision_feat(cams):
    s = cams.shots.get("pregrasp")
    return T.vision_features(s["rgb"], s["mask"]) if s else np.zeros(len(T.VISION_FEATS))


# ------------------------------------------------------------------ one annotated episode
def run_annotated(args):
    idx, name, seed, F_rand, out_dir = args
    cell = S.Cell(seed=seed, objects=(name,))
    T.FabricSkin(cell, seed=seed)
    cams = Cameras(cell, name)
    # dataset episodes: probe, then grasp with a random target force (so that crush / slip outcomes are covered too)
    r, ep = S.run_episode(cell, name, probe_fn=lambda c, e: F_rand, max_retries=0)
    feats_t = T.probe_features(ep)
    feats_v = vision_feat(cams)
    log = [e for e in ep.log if e["phase"] == "inflate"]
    ts = dict(
        t=np.array([e["t"] for e in log], np.float32),
        probing=np.array([e["probing"] for e in log], bool),
        p_cmd=np.array([e["p_cmd"] for e in log], np.float32), p=np.array([e["p"] for e in log], np.float32),
        bend_model=np.array([e["bend"] for e in log], np.float32), q_finger=np.array([e["q_f"] for e in log], np.float32),
        f_est=np.array([e["f_est"] for e in log], np.float32), f_true=np.array([e["f_true"] for e in log], np.float32),
        skin=np.array([e["tax"] for e in log], np.float32), skin_true=np.array([e["tax_true"] for e in log], np.float32),
    )
    imgs = {}
    for tag, s in cams.shots.items():
        imgs[f"{tag}_rgb"] = s["rgb"]
        imgs[f"{tag}_mask"] = s["mask"]
        if "depth" in s:
            imgs[f"{tag}_depth"] = s["depth"]
    cams.r.close()
    fn = f"ep_{idx:05d}.npz"
    np.savez_compressed(Path(out_dir) / "episodes" / fn, **ts, **imgs, probe_features=feats_t, vision_features=feats_v)
    meta = dict(
        id=idx, file=f"episodes/{fn}", seed=seed, object=name, looks_like=S.VISUAL_ID.get(name, name),
        true_class=S.OBJECTS[name][5], prior_class=S.VISUAL_CLASS[S.VISUAL_ID.get(name, name)], max_safe_force=S.OBJECTS[name][4],
        physics=hidden_physics(cell, name), F_probe=0.4, F_target=round(F_rand, 3),
        outcome=dict(success=r["success"], crushed=r["crushed"], lifted=r["lifted"], slipped=not r["lifted"], in_tray=r["in_tray"],
                     peak_force_true=r["peak_force_true"], peak_force_est=r["peak_force_est"]),
        boxes={tag: bbox(s["mask"]) for tag, s in cams.shots.items()},
        probe_features=dict(zip(T.PROBE_FEATS, np.round(feats_t, 4).tolist())),
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
            jobs.append((idx, name, seed0 + idx, float(rng.uniform(0.5, 4.0)), str(out_dir)))
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
    sr = np.mean([m["outcome"]["success"] for m in metas])
    cr = np.mean([m["outcome"]["crushed"] for m in metas])
    sl = np.mean([m["outcome"]["slipped"] for m in metas])
    return f"""# SoftTwin tactile grasp dataset (synthetic, auto-annotated)

- episodes: {n} ({len(ALL_OBJECTS)} object types incl. 2 look-alikes), generated in {sec/60:.1f} min on 2 CPU cores
- control-rate samples (50 Hz, grasp phase): {steps}; each has pressure command / pressure / model bend / 3 finger angles /
  estimated and true grip force / 24 skin taxels (sensor model) / 24 true taxel forces
- images per episode: wrist RGB + depth + segmentation mask (pre-grasp), scene RGB + mask (grasp, lift); {IMG_W}x{IMG_H}
- labels per episode: object, looks-like, true handling class, visual prior class, max safe force, hidden physics
  (mass, friction, contact time constant, solimp), random target force, outcome (success {sr:.0%}, crushed {cr:.0%}, slipped {sl:.0%}),
  bounding boxes, probe and vision feature vectors
- files: `index.jsonl` / `index.parquet` (one row per episode), `episodes/ep_XXXXX.npz`
- all annotations come from the simulator; skin values pass through `tactile.FabricSkin` (crosstalk, gain error, lag, noise,
  dead zone, saturation, 12-bit ADC). Numbers are assumptions for a textile piezoresistive skin, not a calibrated device.
"""


# ------------------------------------------------------------------ training
def load_index(d=DATA):
    return [json.loads(line) for line in open(Path(d) / "index.jsonl")]


def feature_sets(metas):
    V = np.array([[m["vision_features"][k] for k in T.VISION_FEATS] for m in metas])
    P = np.array([[m["probe_features"][k] for k in T.PROBE_FEATS] for m in metas])
    y = np.array([T.CLASSES.index(m["true_class"]) for m in metas])
    return {"vision": V, "touch": P, "fusion": np.c_[V, P]}, y


def train(d=DATA):
    metas = load_index(d)
    X, y = feature_sets(metas)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(metas))
    n_tr = int(0.7 * len(metas))
    tr, te = perm[:n_tr], perm[n_tr:]
    res, models = {"n_train": int(n_tr), "n_test": int(len(te)), "classes": list(T.CLASSES), "models": {}}, {}
    objs = [m["object"] for m in metas]
    for k, Xk in X.items():
        clf = T.Softmax().fit(Xk[tr], y[tr])
        pred = clf.predict(Xk[te])
        cm = np.zeros((len(T.CLASSES),) * 2, int)
        for a, b in zip(y[te], pred):
            cm[a, b] += 1
        per_obj = {}
        for o in ALL_OBJECTS:
            ii = [i for i, j in enumerate(te) if objs[j] == o]
            per_obj[o] = round(float(np.mean(pred[ii] == y[te][ii])), 3) if ii else None
        res["models"][k] = dict(accuracy=round(float(np.mean(pred == y[te])), 3), per_object=per_obj, confusion=cm.tolist(),
                                features=list(T.VISION_FEATS if k == "vision" else T.PROBE_FEATS if k == "touch" else T.VISION_FEATS + T.PROBE_FEATS))
        models[k] = clf.to_json()
        print(k, res["models"][k]["accuracy"], per_obj)
    # grip-force window from the outcome annotations: success / crush / slip rate vs target force per true class
    win = {}
    edges = np.linspace(0.5, 4.0, 8)
    for c in T.CLASSES:
        ms = [m for m in metas if m["true_class"] == c]
        rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            mm = [m for m in ms if a <= m["F_target"] < b]
            if mm:
                rows.append(dict(F=round((a + b) / 2, 2), n=len(mm), success=round(np.mean([m["outcome"]["success"] for m in mm]), 3),
                                 crushed=round(np.mean([m["outcome"]["crushed"] for m in mm]), 3),
                                 slipped=round(np.mean([m["outcome"]["slipped"] for m in mm]), 3)))
        win[c] = rows
    res["force_window"] = win
    (REPO / "learning").mkdir(exist_ok=True)
    json.dump(models, open(REPO / "learning" / "tactile_classifier.json", "w"))
    json.dump(res, open(REPO / "results" / "tactile_classifier.json", "w"), indent=1)
    return res


# ------------------------------------------------------------------ look-alike benchmark (closed loop, online)
def _bench_job(args):
    name, seed, policy = args
    models = {k: T.Softmax.from_json(v) for k, v in json.load(open(REPO / "learning" / "tactile_classifier.json")).items()}
    cell = S.Cell(seed=seed, objects=(name,))
    T.FabricSkin(cell, seed=seed)
    cams = Cameras(cell, name)
    prior = S.VISUAL_CLASS[S.VISUAL_ID.get(name, name)]
    info = {}
    if policy == "vlm_prior":
        r, ep = S.run_episode(cell, name, F_target=S.TARGET_FORCE[prior])
    elif policy == "oracle":
        r, ep = S.run_episode(cell, name, F_target=S.TARGET_FORCE[S.OBJECTS[name][5]])
    else:
        kind = policy.split("_")[0]              # vision_clf / touch_clf / fusion_clf

        def choose(c, e):
            fv, ft = vision_feat(cams), T.probe_features(e)
            x = {"vision": fv, "touch": ft, "fusion": np.r_[fv, ft]}[kind]
            pr = models[kind].predict_proba(x)[0]
            cls = T.CLASSES[int(pr.argmax())]
            info.update(pred=cls, conf=round(float(pr.max()), 3))
            return S.TARGET_FORCE[cls]
        r, ep = S.run_episode(cell, name, F_target=S.TARGET_FORCE[prior], probe_fn=choose)
    cams.r.close()
    t_grasp = max(e["t"] for e in ep.log)
    return dict(r, policy=policy, seed=seed, prior=prior, cycle_time_s=round(t_grasp, 2), **info)


def bench(seeds, workers=2):
    policies = ("vlm_prior", "vision_clf", "touch_clf", "fusion_clf", "oracle")
    jobs = [(n, s, p) for s in seeds for n in ALL_OBJECTS for p in policies]
    rows, todo = [], list(jobs)
    while todo:
        pool = Pool(workers, maxtasksperchild=20)
        pend = [(jb, pool.apply_async(_bench_job, (jb,))) for jb in todo]
        todo, broken = [], False
        for jb, ar in pend:
            if broken:
                todo.append(jb)
                continue
            try:
                rows.append(ar.get(timeout=120))
            except Exception as e:  # crashed renderer -> retry this job once in a fresh pool
                print("retry", jb, repr(e)[:60], flush=True)
                todo.append(jb)
                broken = True
        pool.terminate()
    summ = {}
    for p in policies:
        rs = [r for r in rows if r["policy"] == p]
        la = [r for r in rs if r["object"] in S.VISUAL_ID]
        summ[p] = dict(n=len(rs), success=round(np.mean([r["success"] for r in rs]), 3), crushed=round(np.mean([r["crushed"] for r in rs]), 3),
                       dropped=round(np.mean([not r["lifted"] for r in rs]), 3), attempts=round(float(np.mean([r["attempts"] for r in rs])), 2),
                       cycle_time_s=round(float(np.mean([r["cycle_time_s"] for r in rs])), 2),
                       lookalike_success=round(np.mean([r["success"] for r in la]), 3), lookalike_crushed=round(np.mean([r["crushed"] for r in la]), 3),
                       by_object={o: round(np.mean([r["success"] for r in rs if r["object"] == o]), 3) for o in ALL_OBJECTS})
        print(p, json.dumps(summ[p]))
    res = dict(engine=f"MuJoCo {mujoco.__version__}", seeds=list(seeds), summary=summ, rows=rows)
    json.dump(res, open(REPO / "results" / "benchmark_lookalike.json", "w"), indent=1, default=str)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    g = sp.add_parser("generate")
    g.add_argument("--per-object", type=int, default=40)
    g.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    sp.add_parser("train")
    b = sp.add_parser("bench")
    b.add_argument("--seeds", type=int, default=10)
    b.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.per_object, workers=a.workers)
    elif a.cmd == "train":
        train()
    elif a.cmd == "bench":
        bench(range(a.seeds), workers=a.workers)
