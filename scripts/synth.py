"""Test 4 of docs/test-plan.md: synthetic DC from other large police agencies.

This estimates what the whole federal surge did (Guard, federal agents, federal control of MPD),
not the Guard alone.

  .venv/bin/python scripts/synth.py   ->  out/synth.json, out/synth_series.csv
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRE = ("2017-01-01", "2025-07-01")
POST = ("2025-09-01", "2026-06-01")
BASE = ("2017-01-01", "2024-12-01")
OUTCOMES = {"murder": ["murder_total"], "robbery": ["robbery_total"],
            "property": ["burglary_total", "theft_total", "motor_total"]}


def weights(y, X):
    """Non-negative weights summing to 1 that make X @ w track y."""
    k = X.shape[1]
    obj = lambda w: np.sum((y - X @ w) ** 2)
    grad = lambda w: -2 * X.T @ (y - X @ w)
    res = minimize(obj, np.full(k, 1 / k), jac=grad, method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def run(panel, unit):
    y = panel[unit]
    X = panel.drop(columns=unit)
    pre = (panel.index >= PRE[0]) & (panel.index <= PRE[1])
    post = (panel.index >= POST[0]) & (panel.index <= POST[1])
    w = weights(y[pre].values, X[pre].values)
    synth = X.values @ w
    gap = y.values - synth
    pre_rmspe = np.sqrt(np.mean(gap[pre] ** 2))
    post_rmspe = np.sqrt(np.mean(gap[post] ** 2))
    effect = y[post].sum() / synth[post].sum() - 1
    return {"w": pd.Series(w, index=X.columns), "synth": synth, "pre_rmspe": pre_rmspe, "ratio": post_rmspe / pre_rmspe,
            "effect": effect}


def main():
    r = pd.read_csv(ROOT / "data/external/rtci_Jan17_Jun26.csv", low_memory=False)
    # agencies only: lowercase ids are state and region totals (the "dc" total duplicates DC itself)
    a = r[(r["size"] == "all") & (r.agencies == 1) & r.id.str.match(r"^[A-Z]{2}[A-Z0-9]{7}$")].copy()
    a["m"] = pd.to_datetime(a.year.astype(str) + "-" + a.month.astype(str) + "-01")
    a = a[(a.m >= "2017-01-01") & (a.m <= "2026-06-01")]
    names = {"DCMPD0000": "Washington, DC"}
    out, series = {}, []
    for k, cols in OUTCOMES.items():
        a[k] = a[cols].sum(axis=1, min_count=len(cols))
        wide = a.pivot_table(index="m", columns="id", values=k, aggfunc="first")
        big = a.groupby("id").population.max()
        donors = [c for c in wide.columns if c != "DCMPD0000" and big.get(c, 0) >= 250_000
                  and wide[c].notna().all() and len(wide[c]) == 114]
        panel = wide[["DCMPD0000"] + donors].astype(float)
        base = panel.loc[BASE[0]:BASE[1]].mean()
        panel = panel[[c for c in panel.columns if base[c] > 0]] / base[[c for c in panel.columns if base[c] > 0]]
        dc = run(panel, "DCMPD0000")
        plac = {}
        for d in [c for c in panel.columns if c != "DCMPD0000"]:
            p = run(panel.drop(columns="DCMPD0000"), d)
            plac[d] = p
        ratios = np.array([p["ratio"] for p in plac.values()])
        pre_r = np.array([p["pre_rmspe"] for p in plac.values()])
        keep = pre_r <= 5 * dc["pre_rmspe"]
        top = dc["w"].sort_values(ascending=False).head(6)
        out[k] = {"donors": len(panel.columns) - 1,
                  "effect_pct": round(100 * dc["effect"], 1),
                  "pre_rmspe": round(float(dc["pre_rmspe"]), 3),
                  "post_pre_ratio": round(float(dc["ratio"]), 2),
                  "rank_all": int((ratios >= dc["ratio"]).sum() + 1), "p_all": round(float(((ratios >= dc["ratio"]).sum() + 1) / (len(ratios) + 1)), 3),
                  "rank_good_fit": int((ratios[keep] >= dc["ratio"]).sum() + 1), "n_good_fit": int(keep.sum()),
                  "p_good_fit": round(float(((ratios[keep] >= dc["ratio"]).sum() + 1) / (keep.sum() + 1)), 3),
                  "placebo_effects_pct_quantiles": {str(q): round(100 * float(np.quantile([p["effect"] for p in plac.values()], q)), 1)
                                                    for q in [0.025, 0.5, 0.975]},
                  "top_weights": {i: round(float(v), 3) for i, v in top.items() if v > 0.001}}
        series.append(pd.DataFrame({"month": panel.index.strftime("%Y-%m"), "outcome": k,
                                    "dc": panel["DCMPD0000"].round(4).values, "synthetic": np.round(dc["synth"], 4)}))
        print(k, {x: out[k][x] for x in ["donors", "effect_pct", "pre_rmspe", "post_pre_ratio", "rank_all", "p_all", "rank_good_fit", "n_good_fit", "p_good_fit"]}, out[k]["placebo_effects_pct_quantiles"])
    (ROOT / "out/synth.json").write_text(json.dumps(out, indent=1))
    pd.concat(series).to_csv(ROOT / "out/synth_series.csv", index=False)
    print("wrote out/synth.json, out/synth_series.csv")


if __name__ == "__main__":
    main()
