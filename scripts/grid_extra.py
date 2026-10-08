"""Checks added after the planned tests ran. NOT in docs/test-plan.md; reported as such.

Why: several planned lead tests failed (cells near posts were already moving differently from the
controls before the troops came). Two checks of whether that changes the answer:
  recent baseline   the same stacked model, but the pre-period is only the 24 weeks before each
                    start, so an older divergence does not count as an effect
  pre-trend slope   the slope of the event-study leads, to tell a drift from noise around a single
                    unusual reference period

  .venv/bin/python scripts/grid_extra.py   ->  out/grid_extra.json
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grid as G  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    cells, events, _ = G.load()
    ev = pd.read_csv(ROOT / "out/grid_event.csv")
    out = {"note": "Not in the test plan; added after the planned tests ran."}
    for k in [G.PRIMARY] + G.SECONDARY:
        st = G.stacked(events, k, cells)
        recent = []
        for w, df in st.items():
            s = (pd.Timestamp(G.START[w]) - G.P0).days // 28
            recent.append(df[df.p >= s - 6])
        r = G.fit(pd.concat(recent, ignore_index=True))
        e = ev[(ev.outcome == k) & (ev.rel <= -2)]
        slope = np.polyfit(e.rel, np.log(e.rr), 1)[0]
        lead_mean = float(np.exp(np.log(e.rr).mean()))
        post = ev[(ev.outcome == k) & (ev.rel >= 0)]
        out[k] = {"recent_baseline": r["treat"], "recent_baseline_ring": r.get("ring"), "verdict_recent": r["verdict"],
                  "lead_slope_pct_per_4wk": round(100 * (np.exp(slope) - 1), 1),
                  "lead_mean_rr": round(lead_mean, 2), "lag_mean_rr": round(float(np.exp(np.log(post.rr).mean())), 2)}
        print(k, out[k])
    (ROOT / "out/grid_extra.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
