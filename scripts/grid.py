"""Tests 1 and 2 of docs/test-plan.md: stacked event study on 500 m cells, and placebo posts.

  .venv/bin/python scripts/grid.py            ->  out/grid.json, out/grid_event.csv
  .venv/bin/python scripts/grid.py --ri 1000  ->  also out/grid_ri.json (placebo posts)

Effects are rate ratios: 0.85 means 15% fewer events near posts than the comparison cells
predict. Smallest effect that matters, fixed in the plan: a 15% reduction.
"""
import argparse
import re
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
P0 = pd.Timestamp("2025-08-11")            # a 4-week period starts here
LAT0, LON0 = 38.79, -77.12
KX, KY = 111_320 * np.cos(np.radians(38.9)), 110_540
CELL = 500
SESOI = 0.85                               # rate ratio for a 15% reduction

W1_STATIONS = ["Foggy Bottom-GWU", "Smithsonian", "Eastern Market", "Stadium Armory", "Waterfront", "McPherson Sq",
               "L'Enfant Plaza", "Gallery Pl-Chinatown", "Metro Center", "NoMa - Gallaudet U", "Union Station", "Navy Yard - Ballpark"]
MALL = {"Washington Monument": (38.8895, -77.0353), "Lincoln Memorial": (38.8893, -77.0502)}
W2 = {"H St NE at 4th": (38.9001, -76.9999), "H St NE at 8th": (38.9001, -76.9951), "H St NE at 12th": (38.9001, -76.9906),
      "Dupont Circle": (38.9097, -77.0435), "14th St NW at P": (38.9096, -77.0319), "14th St NW at U": (38.9169, -77.0320),
      "Georgetown, M St at Wisconsin": (38.9050, -77.0629)}
W3 = {"MLK Jr Ave at Good Hope Rd SE": (38.8628, -76.9856)}        # plus Anacostia station
W4 = {"Starburst Plaza": (38.9003, -76.9812), "Trinidad, Wheatley Education Campus": (38.9046, -76.9846)}
# --updated: sites in the DC Guard's Sept 12, 2025 daily update (DC v. Trump, ECF 83-1, pp. 10-11),
# found after the plan was written. They join Wave 2, which then starts Sept 1, 2025, and Metro
# stations within 500 m of any known site leave the placebo pool. Reported as an update, not as planned.
W2_UPDATED = {"Lincoln Park": (38.8899, -76.9903), "Seward Square": (38.8868, -76.9985), "Garfield Park": (38.8835, -77.0040),
              "Fields at RFK": (38.8897, -76.9718), "City Center": (38.9008, -77.0244)}
UPDATED = False
START = {1: "2025-08-11", 2: "2025-11-20", 3: "2026-01-01"}
START_SENS = {1: "2025-08-11", 2: "2025-09-01", 3: "2025-12-01"}
END = {"gunshots": "2026-06-30"}
END_DEFAULT = "2026-09-30"
VIOLENT = ["HOMICIDE", "SEX ABUSE", "ASSAULT W/DANGEROUS WEAPON", "ROBBERY"]
PRIMARY = "gunshots"
SECONDARY = ["violent_ex_adw", "gun_violent", "property", "theft_other", "theft_auto"]
LABELS = {"gunshots": "Gunshots detected", "violent_ex_adw": "Homicide, robbery and sex abuse", "gun_violent": "Reported gun crime",
          "property": "Property crime", "theft_other": "Other theft", "theft_auto": "Theft from cars"}


def to_xy(lat, lon):
    return (np.asarray(lon) - LON0) * KX, (np.asarray(lat) - LAT0) * KY


def load():
    st = json.loads((ROOT / "data/external/metro_stations.geojson").read_text())["features"]
    stations = {f["properties"]["NAME"]: (f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in st}
    posts = {1: {n: stations[n] for n in W1_STATIONS} | MALL, 2: W2 | (W2_UPDATED if UPDATED else {}),
             3: W3 | {"Anacostia": stations["Anacostia"]}, 4: W4}
    other_stations = {n: v for n, v in stations.items() if n not in W1_STATIONS and n != "Anacostia"}
    if UPDATED:
        known = [v for ps in posts.values() for v in ps.values()]
        kx, ky = to_xy([v[0] for v in known], [v[1] for v in known])
        def near_known(v):
            x, y = to_xy([v[0]], [v[1]])
            return np.sqrt((kx - x[0]) ** 2 + (ky - y[0]) ** 2).min() <= 500
        other_stations = {n: v for n, v in other_stations.items() if not near_known(v)}

    inc = pd.read_csv(ROOT / "data/raw/mpd_incidents_2015_2026.csv", usecols=["REPORT_DAT", "OFFENSE", "METHOD", "LATITUDE", "LONGITUDE"],
                      parse_dates=["REPORT_DAT"])
    inc = inc.rename(columns={"REPORT_DAT": "date", "LATITUDE": "lat", "LONGITUDE": "lon"})
    shots = pd.read_csv(ROOT / "data/raw/shotspotter.csv", parse_dates=["DATETIME"]).drop_duplicates("ID")
    shots = shots[shots.TYPE.str.replace("_", " ").isin(["Single Gunshot", "Multiple Gunshots"])]
    shots = shots.rename(columns={"DATETIME": "date", "LATITUDE": "lat", "LONGITUDE": "lon"})

    allpts = pd.concat([inc[["lat", "lon"]], shots[["lat", "lon"]]])
    x, y = to_xy(allpts.lat, allpts.lon)
    cells = pd.DataFrame({"ix": (x // CELL).astype(int), "iy": (y // CELL).astype(int)}).drop_duplicates().reset_index(drop=True)
    cells["cell"] = cells.ix.astype(str) + "_" + cells.iy.astype(str)
    cx, cy = (cells.ix + 0.5) * CELL, (cells.iy + 0.5) * CELL
    for w, ps in posts.items():
        px, py = to_xy([v[0] for v in ps.values()], [v[1] for v in ps.values()])
        cells[f"d{w}"] = np.sqrt((cx.values[:, None] - px[None, :]) ** 2 + (cy.values[:, None] - py[None, :]) ** 2).min(axis=1)
    mx, my = to_xy([v[0] for v in MALL.values()], [v[1] for v in MALL.values()])
    cells["d_mall"] = np.sqrt((cx.values[:, None] - mx[None, :]) ** 2 + (cy.values[:, None] - my[None, :]) ** 2).min(axis=1)
    cells["cx"], cells["cy"] = cx, cy

    events = {"gunshots": shots,
              "violent_ex_adw": inc[inc.OFFENSE.isin(["HOMICIDE", "ROBBERY", "SEX ABUSE"])],
              "gun_violent": inc[inc.OFFENSE.isin(VIOLENT) & (inc.METHOD == "GUN")],
              "property": inc[~inc.OFFENSE.isin(VIOLENT)],
              "theft_other": inc[inc.OFFENSE == "THEFT/OTHER"],
              "theft_auto": inc[inc.OFFENSE == "THEFT F/AUTO"]}
    for k, e in events.items():
        e = e[e.date >= "2023-01-01"][["date", "lat", "lon"]].copy()
        x, y = to_xy(e.lat, e.lon)
        e["cell"] = (x // CELL).astype(int).astype(str) + "_" + (y // CELL).astype(int).astype(str)
        e["day"] = (e.date.dt.normalize() - P0).dt.days
        events[k] = e[["cell", "day"]]
    return cells, events, other_stations


def assign(cells, r, starts):
    """Earliest wave each cell is treated in (center within r of a post), its ring wave, and control status."""
    c = cells.copy()
    c["wave"] = 0
    for w in [3, 2, 1]:
        c.loc[c[f"d{w}"] <= r, "wave"] = w
    outer = max(2 * r, 1000)
    dmin = c[["d1", "d2", "d3", "d4"]].min(axis=1)
    c["control"] = dmin > outer
    c["ring_wave"] = 0
    for w in [3, 2, 1]:
        c.loc[(c.wave == 0) & (c[f"d{w}"] > r) & (c[f"d{w}"] <= outer), "ring_wave"] = w
    return c


def panel(events, outcome, c, w, start, ring=True, drop_fireworks=False, keep=None):
    """One stack: treated cells of wave w, its ring, and controls, from two years before the start."""
    s = (pd.Timestamp(start) - P0).days
    ws = s - 728
    end = (pd.Timestamp(END.get(outcome, END_DEFAULT)) - P0).days
    sel = c[(c.wave == w) | c.control | ((c.ring_wave == w) if ring else False)]
    if keep is not None:
        sel = sel[sel.cell.isin(keep)]
    e = events[outcome]
    e = e[(e.day >= ws) & (e.day <= end) & e.cell.isin(sel.cell)]
    p_lo, p_hi = ws // 28, end // 28
    periods = np.arange(p_lo, p_hi + 1)
    cnt = e.assign(p=e.day // 28).groupby(["cell", "p"]).size()
    idx = pd.MultiIndex.from_product([sel.cell, periods], names=["cell", "p"])
    df = cnt.reindex(idx, fill_value=0).rename("y").reset_index()
    first = np.maximum(periods * 28, ws)
    last = np.minimum(periods * 28 + 27, end)
    expo = pd.Series(last - first + 1, index=periods)
    share = pd.Series(np.clip((last - np.maximum(first, s) + 1) / (last - first + 1), 0, 1), index=periods)
    df["lexp"] = np.log(df.p.map(expo).astype(float))
    meta = sel.set_index("cell")
    is_t = df.cell.map(meta.wave == w).values
    is_r = df.cell.map(meta.ring_wave == w).values & ~is_t if ring else np.zeros(len(df), bool)
    df["treat"] = np.where(is_t, df.p.map(share), 0.0)
    df["ring"] = np.where(is_r, df.p.map(share), 0.0)
    df["is_treated"] = is_t
    df["kind"] = np.where(is_t, "treated", np.where(df.cell.map(meta.ring_wave == w).values & ring, "ring", "control"))
    df["rel"] = np.where(is_t, np.clip(df.p - s // 28, -13, 10), -1)
    df["stk"] = w
    if drop_fireworks:
        bad = []
        for y in range(2023, 2027):
            for d in [pd.Timestamp(f"{y}-07-04"), pd.Timestamp(f"{y}-12-31")]:
                bad.append((d - P0).days // 28)
        df = df[~df.p.isin(bad)]
    return df


def restrict_gunshot_cells(events, cells):
    e = events["gunshots"]
    pre = e[(e.day >= -731) & (e.day < 0)]
    return set(pre.cell)


def fit(df, ring=True):
    df = df.assign(cs=df.cell + "_" + df["stk"].astype(str), ps=df.p.astype(str) + "_" + df["stk"].astype(str))
    rhs = "treat + ring" if ring and df.ring.sum() > 0 else "treat"
    f = pf.fepois(f"y ~ {rhs} | cs + ps", data=df, offset="lexp", vcov={"CRV1": "cell"})
    t = f.tidy()
    out = {}
    for v in t.index:
        b, se, p = t.loc[v, "Estimate"], t.loc[v, "Std. Error"], t.loc[v, "Pr(>|t|)"]
        rr, lo, hi = np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
        out[v] = {"rr": round(float(rr), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "p": round(float(p), 4)}
    out["verdict"] = verdict(out["treat"])
    out["n_obs"] = int(getattr(f, "_N", len(df)))
    out["treated_cells"] = int(df.loc[df.is_treated, "cell"].nunique())
    return out


def verdict(t):
    lo, hi = t["ci95"]
    if hi < 1:
        return "reduction"
    if lo > 1:
        return "increase"
    if lo >= SESOI:
        return "rules out a reduction of 15% or more"
    return "inconclusive"


def event_study(df):
    d = df[df.kind != "ring"].copy()
    d = d.assign(cs=d.cell + "_" + d["stk"].astype(str), ps=d.p.astype(str) + "_" + d["stk"].astype(str))
    f = pf.fepois("y ~ i(rel, ref=-1) | cs + ps", data=d, offset="lexp", vcov={"CRV1": "cell"})
    t = f.tidy()
    names = list(f._coefnames)
    rows = []
    for v in t.index:
        k = int(re.findall(r"-?\d+", str(v))[-1])
        rows.append({"rel": k, "rr": float(np.exp(t.loc[v, "Estimate"])), "lo": float(np.exp(t.loc[v, "2.5%"])),
                     "hi": float(np.exp(t.loc[v, "97.5%"]))})
    lead = [i for i, v in enumerate(names) if int(re.findall(r"-?\d+", str(v))[-1]) <= -2]
    b = f.coef().values[lead]
    V = f._vcov[np.ix_(lead, lead)]
    chi2 = float(b @ np.linalg.solve(V, b))
    from scipy.stats import chi2 as C2
    return pd.DataFrame(rows).sort_values("rel"), {"leads": len(lead), "chi2": round(chi2, 2), "p": round(float(C2.sf(chi2, len(lead))), 4)}


def stacked(events, outcome, cells, r=500, starts=None, **kw):
    starts = starts or START
    c = assign(cells, r, starts)
    keep = restrict_gunshot_cells(events, cells) if outcome == "gunshots" else None
    stacks = {w: panel(events, outcome, c, w, starts[w], keep=keep, **kw) for w in [1, 2, 3]}
    return stacks


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def placebo(events, cells, other_stations, outcome, n, r=500, seed=20261008):
    """Wave 1 Metro posts only: real estimate, then n draws of 12 stations without Guard posts."""
    rng = np.random.default_rng(seed)
    st = json.loads((ROOT / "data/external/metro_stations.geojson").read_text())["features"]
    stations = {f["properties"]["NAME"]: (f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in st}
    keep = restrict_gunshot_cells(events, cells) if outcome == "gunshots" else None
    c0 = assign(cells, r, START)
    cx, cy = c0.cx.values, c0.cy.values

    def dist_to(names):
        px, py = to_xy([stations[n][0] for n in names], [stations[n][1] for n in names])
        return np.sqrt((cx[:, None] - px[None, :]) ** 2 + (cy[:, None] - py[None, :]) ** 2).min(axis=1)

    real_near = c0[["d1", "d2", "d3"]].min(axis=1) <= r

    def estimate(names, real):
        c = c0.copy()
        near = dist_to(names) <= r
        c["wave"] = np.where(near if real else (near & ~real_near), 1, 0)
        c["control"] = c0.control & ~near
        c["ring_wave"] = 0
        df = panel(events, outcome, c, 1, START[1], ring=False, keep=keep)
        if df.loc[df.is_treated, "y"].sum() == 0:
            return None
        return fit(df, ring=False)["treat"]["rr"]

    real = estimate(W1_STATIONS, True)
    cand = sorted(other_stations)
    draws = []
    for _ in range(n):
        names = list(rng.choice(cand, size=len(W1_STATIONS), replace=False))
        v = estimate(names, False)
        if v is not None:
            draws.append(v)
    draws = np.array(draws)
    return {"real_rr": real, "n_placebos": int(len(draws)), "candidates": len(cand),
            "share_placebos_at_or_below_real": round(float((draws <= real).mean()), 4),
            "placebo_quantiles": {str(q): round(float(np.quantile(draws, q)), 3) for q in [0.025, 0.25, 0.5, 0.75, 0.975]}}


def negbin_wave1(stack):
    """Robustness: negative binomial for the Wave 1 stack, cell and period dummies (pyfixest has no FE negbin)."""
    import statsmodels.api as sm
    d = stack[stack.kind != "ring"]
    d = d[d.groupby("cell").y.transform("sum") > 0]
    X = pd.get_dummies(d[["cell", "p"]].astype(str), drop_first=True, dtype=float)
    X.insert(0, "treat", d.treat.values)
    X = sm.add_constant(X)
    m = sm.NegativeBinomial(d.y.values, X, offset=d.lexp.values).fit(disp=0, maxiter=200, method="lbfgs")
    b, se = m.params["treat"], m.bse["treat"]
    return {"rr": round(float(np.exp(b)), 3), "ci95": [round(float(np.exp(b - 1.96 * se)), 3), round(float(np.exp(b + 1.96 * se)), 3)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ri", type=int, default=0)
    ap.add_argument("--updated", action="store_true")
    a = ap.parse_args()
    global UPDATED, START
    UPDATED = a.updated
    if UPDATED:
        START = {**START, 2: "2025-09-01"}
    sfx = "_updated" if UPDATED else ""
    cells, events, other_stations = load()
    res = {"plan": "docs/test-plan.md", "updated_sites": UPDATED, "sesoi_rr": SESOI, "cells": int(len(cells)),
           "placebo_pool": sorted(other_stations), "outcomes": {}}
    ev_rows = []
    for k in [PRIMARY] + SECONDARY:
        st = stacked(events, k, cells)
        pooled = pd.concat(st.values(), ignore_index=True)
        out = {"label": LABELS[k], "pooled": fit(pooled), "by_wave": {w: fit(df) for w, df in st.items()}}
        es, lead_test = event_study(pooled)
        out["lead_test"] = lead_test
        ev_rows.append(es.assign(outcome=k))
        rob = {}
        for r in [250, 1000]:
            rob[f"radius_{r}"] = fit(pd.concat(stacked(events, k, cells, r=r).values(), ignore_index=True))["treat"]
        rob["sensitivity_starts"] = fit(pd.concat(stacked(events, k, cells, starts=START_SENS).values(), ignore_index=True))["treat"]
        rob["drop_fireworks"] = fit(pd.concat(stacked(events, k, cells, drop_fireworks=True).values(), ignore_index=True))["treat"]
        c = assign(cells, 500, START)
        mall = set(c.loc[c.d_mall <= 500, "cell"])
        rob["drop_mall"] = fit(pooled[~pooled.cell.isin(mall)])["treat"]
        pre = pooled[(pooled.p < 0) & (pooled.kind == "control")].groupby("cell").y.sum()
        busy = set(pre[pre >= pre.quantile(0.8)].index)
        rob["busy_controls"] = fit(pooled[(pooled.kind != "control") | pooled.cell.isin(busy)])["treat"]
        if k == PRIMARY:
            try:
                rob["negbin_wave1"] = negbin_wave1(st[1])
            except Exception as ex:          # reported, not hidden
                rob["negbin_wave1"] = {"error": str(ex)[:200]}
        out["robustness"] = rob
        res["outcomes"][k] = out
        print(k, out["pooled"]["treat"], out["pooled"].get("ring"), out["verdict"] if "verdict" in out else out["pooled"]["verdict"], lead_test)
    ps = [res["outcomes"][k]["pooled"]["treat"]["p"] for k in SECONDARY]
    for k, pa in zip(SECONDARY, holm(ps)):
        res["outcomes"][k]["pooled"]["treat"]["p_holm"] = round(float(pa), 4)
    pd.concat(ev_rows).to_csv(ROOT / f"out/grid_event{sfx}.csv", index=False)
    (ROOT / f"out/grid{sfx}.json").write_text(json.dumps(res, indent=1))
    print(f"wrote out/grid{sfx}.json, out/grid_event{sfx}.csv")
    if a.ri:
        ri = {k: placebo(events, cells, other_stations, k, a.ri) for k in [PRIMARY] + SECONDARY}
        (ROOT / f"out/grid_ri{sfx}.json").write_text(json.dumps(ri, indent=1))
        print(json.dumps(ri, indent=1))


if __name__ == "__main__":
    main()
