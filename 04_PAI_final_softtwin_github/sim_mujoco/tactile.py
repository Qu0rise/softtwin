"""Tactile skin + haptic probing for the SoftTwin gripper (MuJoCo).

* FabricSkin   : sensor model of a cloth-like piezoresistive skin glued on the inner silicone surface
                 (24 taxels = 3 fingers x 4 segments x 2).  Raw MuJoCo touch forces -> crosstalk, per-taxel gain
                 error, first-order lag, noise, dead zone, saturation and ADC quantisation.
* probe_features: features from the gentle probe squeeze (F_probe = 0.4 N) - skin + pressure + proprioception.
* vision_features: what a camera can see (mean colour, area and shape of the segmented object).

The look-alike objects (overripe tomato, resin sample strawberry) share appearance with a base object but not
its hidden physics, so vision alone cannot pick a safe grip force; touch can.
"""
from __future__ import annotations

import numpy as np

import softtwin_mj as S

N_TAX = S.CFG["n_fingers"] * S.CFG["n_seg"] * 2
CLASSES = ("very_fragile", "soft", "fragile", "rigid")


class FabricSkin:
    """cloth-like piezoresistive skin on the silicone fingers (all numbers are assumptions of a typical textile sensor)"""

    def __init__(self, cell, seed=0, tau=0.015, crosstalk=0.10, gain_sd=0.08, noise=0.006, dead=0.02, sat=8.0, bits=12):
        self.c, self.rng = cell, np.random.default_rng(seed + 7)
        self.tau, self.xt, self.noise, self.dead, self.sat = tau, crosstalk, noise, dead, sat
        self.gain = 1.0 + self.rng.normal(0, gain_sd, N_TAX)
        self.lsb = sat / (2 ** bits)
        self.y = np.zeros(N_TAX)
        # neighbours along each finger (8 taxels in a row per finger)
        self.A = np.eye(N_TAX)
        for f in range(S.CFG["n_fingers"]):
            for i in range(8):
                j = f * 8 + i
                for d in (-1, 1):
                    if 0 <= i + d < 8:
                        self.A[j, j + d] = self.xt
        dt = cell.m.opt.timestep
        self.a = 1 - np.exp(-dt / tau)
        orig = cell._step

        def step():
            orig()
            self.y += self.a * (self.A @ cell.taxels() * self.gain - self.y)

        cell._step = step
        cell.skin = self

    def read(self):
        v = self.y + self.rng.normal(0, self.noise, N_TAX)
        v = np.where(v < self.dead, 0.0, v)
        return np.round(np.clip(v, 0, self.sat) / self.lsb) * self.lsb


# ------------------------------------------------------------------ features
PROBE_FEATS = ("q_contact", "dq", "stiff", "p_contact", "dp", "tax_sum", "n_active", "seg_centroid", "tip_ratio", "settle_steps")


def probe_features(ep):
    """features of the probe squeeze, from first skin contact to the end of the probe.
    stiffness is measured per finger (skin force rise / that finger's extra bend after its own first contact), then the median."""
    pr = [e for e in ep.log if e.get("phase") == "inflate" and e.get("probing")]
    if not pr:
        return np.zeros(len(PROBE_FEATS))
    nf, ns = S.CFG["n_fingers"], S.CFG["n_seg"]
    tax = np.array([e["tax"] for e in pr]).reshape(len(pr), nf, ns * 2)
    qf = np.array([e["q_f"] for e in pr])
    sf = tax.sum(2)                                   # skin force per finger over time
    s = sf.sum(1)
    i0 = int(np.argmax(s > 0.08)) if (s > 0.08).any() else len(pr) - 1
    stiff, dqs = [], []
    for f in range(nf):
        on = np.nonzero(sf[:, f] > 0.03)[0]
        if len(on) and len(pr) - on[0] > 3:
            dq = qf[-1, f] - qf[on[0], f]
            stiff.append((sf[-1, f] - sf[on[0], f]) / max(dq, 0.1))
            dqs.append(dq)
    end = tax[-1].reshape(nf, ns, 2)
    seg_load = end.sum((0, 2))
    return np.array([
        float(qf[i0].mean()), float(np.median(dqs)) if dqs else 0.0, float(np.median(stiff)) if stiff else 0.0,
        pr[i0]["p"], pr[-1]["p"] - pr[i0]["p"],
        s[-1], float((tax[-1] > 0.02).sum()),
        float(seg_load @ np.arange(ns) / max(seg_load.sum(), 1e-6)),
        float(end[:, -1, 1].sum() / max(end.sum(), 1e-6)),
        float(len(pr) - i0),
    ])


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


def true_class(name):
    return CLASSES.index(S.OBJECTS[name][5])
