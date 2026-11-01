"""Touch features, classifiers and the learned grip-force window (numpy only).

* probe_features   : the finger probe - squeeze to 0.8 N, read the fabric skin and the bend sensors (twin_core.Sensors)
* plunger_features : the firmness plunger at the palm centre (twin_core.probe_plunger)
* vision_features  : what a camera can see (mean colour, area and shape of the segmented object)

The look-alikes (overripe tomato / strawberry) share the appearance of a base object, not its firmness, so vision alone
cannot pick a safe grip force.  Whether touch can depends on the probe: see scripts/probe_sensitivity.py.
"""
from __future__ import annotations

import numpy as np

import twin_core as C

N_TAX = C.NF * C.NS * 2
CLASSES = C.CLASSES


# ------------------------------------------------------------------ features
PROBE_FEATS = ("compliance", "compliance_sd", "bend_contact", "p_contact", "dp_probe", "tax_sum", "n_active", "seg_centroid",
               "tip_ratio", "settle_steps")


def probe_features(ep):
    """features of the probe squeeze, from the skin's first contact to the end of the probe (sensor readings only).

    compliance: per finger, least-squares slope of the measured bend against that finger's skin force while the probe
    force builds up [deg/N], median over the fingers.  Once a fingertip rests on the object, the finger can only bend
    further by indenting the object, so a softer object gives a larger slope (finger kinematics: 1 deg of total bend
    ~ 1.4 mm at the tip).  The other features describe where and how much the skin is loaded."""
    pr = [e for e in ep.log if e.get("phase") == "grip" and e.get("probing")]
    if not pr:
        return np.zeros(len(PROBE_FEATS))
    nf, ns = C.NF, C.NS
    tax = np.array([e["tax"] for e in pr]).reshape(len(pr), nf, ns * 2)
    bend = np.array([e["bend_meas"] for e in pr])
    sf = tax.sum(2)
    s = sf.sum(1)
    on = np.nonzero(s > C.SensorSpec.contact_thr)[0]
    i0 = int(on[0]) if len(on) else len(pr) - 1
    slopes = []
    for f in range(nf):
        idx = np.arange(i0, len(pr))
        idx = idx[sf[idx, f] > 0.0]
        if len(idx) >= 5 and np.ptp(sf[idx, f]) > 0.05:
            slopes.append(float(np.polyfit(sf[idx, f], bend[idx, f], 1)[0]))
    end = tax[-1].reshape(nf, ns, 2)
    seg_load = end.sum((0, 2))
    return np.array([
        float(np.median(slopes)) if slopes else 0.0, float(np.std(slopes)) if len(slopes) > 1 else 0.0,
        float(bend[i0].mean()), pr[i0]["p"], pr[-1]["p"] - pr[i0]["p"],
        s[-1], float((tax[-1] > 0.02).sum()),
        float(seg_load @ np.arange(ns) / max(seg_load.sum(), 1e-6)),
        float(end[:, -1, 1].sum() / max(end.sum(), 1e-6)),
        float(len(pr) - i0),
    ])


PLUNGER_FEATS = ("k_plunger", "top_height", "indent_end")


def plunger_features(ep):
    """firmness plunger: slope of load vs indentation [N/mm], contact height above the table [m], indentation at the end
    of the probe [mm] (twin_core.probe_plunger)"""
    p = getattr(ep, "plunger", None)
    if not p or p.get("n", 0) < 4:
        return np.zeros(len(PLUNGER_FEATS))
    return np.array([p["k"], p["top"] - C.CFG["table_top"], p["indent_end"]])


VISION_FEATS = ("r", "g", "b", "area", "aspect")


def vision_features(rgb, seg_obj_mask):
    m = seg_obj_mask
    if m.sum() < 3:
        return np.zeros(len(VISION_FEATS))
    col = rgb[m].mean(0) / 255.0
    ys, xs = np.nonzero(m)
    h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    return np.array([col[0], col[1], col[2], float(m.sum()) / m.size, float(max(h, w) / max(min(h, w), 1))])


# ------------------------------------------------------------------ tiny multinomial logistic regression (numpy)
class Softmax:
    def __init__(self, n_cls=len(CLASSES), l2=1e-2, epochs=3000, lr=0.3):
        self.k, self.l2, self.epochs, self.lr = n_cls, l2, epochs, lr

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        Y = np.eye(self.k)[y]
        self.W = np.zeros((Z.shape[1], self.k))
        for _ in range(self.epochs):
            P = self._p(Z)
            g = Z.T @ (P - Y) / len(Z) + self.l2 * np.r_[self.W[:-1], np.zeros((1, self.k))]
            self.W -= self.lr * g
        return self

    def _p(self, Z):
        a = Z @ self.W
        a -= a.max(1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(1, keepdims=True)

    def predict_proba(self, X):
        return self._p(np.c_[(np.atleast_2d(X) - self.mu) / self.sd, np.ones(len(np.atleast_2d(X)))])

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def to_json(self):
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(), W=self.W.tolist(), classes=list(CLASSES))

    @classmethod
    def from_json(cls, d):
        m = cls()
        m.mu, m.sd, m.W = np.array(d["mu"]), np.array(d["sd"]), np.array(d["W"])
        return m


class Logistic:
    """binary logistic regression with L2 (numpy, full-batch gradient descent)"""

    def __init__(self, l2=1e-2, epochs=4000, lr=0.3):
        self.l2, self.epochs, self.lr = l2, epochs, lr

    def fit(self, Z, y):
        Z = np.c_[Z, np.ones(len(Z))]
        self.w = np.zeros(Z.shape[1])
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-Z @ self.w))
            self.w -= self.lr * (Z.T @ (p - y) / len(Z) + self.l2 * np.r_[self.w[:-1], 0.0])
        return self

    def p(self, Z):
        return 1 / (1 + np.exp(-np.c_[Z, np.ones(len(Z))] @ self.w))


class ForceWindow:
    """learned grip-force window: p(crush | x, F) and p(slip | x, F) from the auto-annotated outcomes alone.

    No class labels and no hand-set target forces: every dataset episode was grasped with a random F*, and the
    outcome annotation (crushed / slipped) is the label.  phi(x, F) = [z, F, z*F] lets the observation shift and
    tilt both boundaries.  The policy picks the F that minimises p_crush + p_slip; after a slip it may only go up
    while p_crush stays below `crush_max`."""
    GRID = np.round(np.arange(0.4, 4.01, 0.05), 3)  # N

    def __init__(self, l2=3e-3, crush_max=0.2):
        self.l2, self.crush_max = l2, crush_max

    def _phi(self, X, F):
        Z = (np.atleast_2d(X) - self.mu) / self.sd
        F = np.asarray(F, float).reshape(-1, 1)
        if len(Z) == 1 and len(F) > 1:
            Z = np.repeat(Z, len(F), 0)
        return np.c_[Z, F, Z * F]

    def fit(self, X, F, crushed, slipped):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        P = self._phi(X, F)
        self.crush = Logistic(self.l2).fit(P, np.asarray(crushed, float))
        self.slip = Logistic(self.l2).fit(P, np.asarray(slipped, float))
        return self

    def curves(self, x):
        """p(crush) and p(slip) on the force grid.  Shape constraint: squeezing harder can only make a crush more
        likely and a slip less likely, so the fitted curves are projected onto non-decreasing / non-increasing ones
        (the z*F term can otherwise bend them where the data has no crushes at all, e.g. the ripe tomato)."""
        P = self._phi(x, self.GRID)
        return self.GRID, np.maximum.accumulate(self.crush.p(P)), np.minimum.accumulate(self.slip.p(P))

    def choose(self, x):
        g, pc, ps = self.curves(x)
        return float(g[int(np.argmin(pc + ps))])

    def escalate(self, x, F):
        g, pc, ps = self.curves(x)
        ok = (g > F) & (pc < self.crush_max)
        return float(g[ok][int(np.argmin(ps[ok]))]) if ok.any() else F

    def to_json(self):
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(), w_crush=self.crush.w.tolist(), w_slip=self.slip.w.tolist(),
                    crush_max=self.crush_max, features=list(getattr(self, "features", [])))

    @classmethod
    def from_json(cls, d):
        m = cls(crush_max=d["crush_max"])
        m.mu, m.sd = np.array(d["mu"]), np.array(d["sd"])
        m.crush, m.slip = Logistic(), Logistic()
        m.crush.w, m.slip.w = np.array(d["w_crush"]), np.array(d["w_slip"])
        return m


def true_class(name):
    return CLASSES.index(C.OBJECTS[name].cls)
