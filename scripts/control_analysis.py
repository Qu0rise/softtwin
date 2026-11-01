"""Loop numbers quoted in the report (Sec. 6.3): static gain dF/dp after contact, rise time, hold error.

    python scripts/control_analysis.py        -> results/control_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_mujoco"))
import softtwin_mj as S  # noqa: E402
import twin_core as C  # noqa: E402

K_I = 4.0


def main(seeds=range(5)):
    out = {}
    for n in C.BASE_OBJECTS:
        F = C.TARGET_FORCE[C.OBJECTS[n].cls]
        gains, rise, hold = [], [], []
        for s in seeds:
            cell = S.Cell(seed=s, objects=(n,))
            r, ep = S.run_episode(cell, n, max_retries=0)
            g = [e for e in ep.log if e["phase"] == "grip" and e.get("contact")]
            p, f, t = (np.array([e[k] for e in g]) for k in ("p", "f_true", "t"))
            m = (f > 0.2) & (f < 0.8 * F)
            if m.sum() > 5:
                gains.append(float(np.polyfit(p[m], f[m], 1)[0]))
            hit = np.nonzero(f >= 0.8 * F)[0]
            if len(hit):
                rise.append(float(t[hit[0]] - t[0]))
            tr = [e for e in ep.log if e["phase"] == "transport"]
            if tr:
                hold.append(float(np.mean([e["f_true"] for e in tr]) / F))
        out[n] = dict(target_N=F, dF_dp_N_per_kPa=round(float(np.mean(gains)), 3),
                      rise_to_80pct_s=round(float(np.mean(rise)), 2) if rise else None,
                      transport_force_over_target=round(float(np.mean(hold)), 3) if hold else None)
    g = float(np.mean([v["dF_dp_N_per_kPa"] for v in out.values()]))
    res = dict(per_object=out, dF_dp_mean=round(g, 3), K_i=K_I, tau_closed_loop_s=round(1.0 / (K_I * g), 2),
               supply_tau_s=0.25, rate_limit_up_kPa_s=3.0, force_slew_N_s=round(3.0 * g, 2), seeds=list(seeds))
    json.dump(res, open(REPO / "results" / "control_analysis.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
