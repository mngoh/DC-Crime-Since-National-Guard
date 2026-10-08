"""Test 3 of docs/test-plan.md: Metro stations with Guard posts against stations without.

Uses data/interim/mtpd_incidents.csv (scripts/metro_parse.py). Crimes with victims only: assault
(not on police), robbery, theft and larceny, sexual offenses, homicide, burglary, motor vehicle
theft. Enforcement offenses (fare evasion, drugs, weapons, warrants, disorderly) are left out
because they measure police activity, and fare enforcement grew about fifty-fold over the period.

Guard stations: the 12 reported in August 2025. The DC Guard's Sept 12, 2025 update says 18
stations; the other 6 are not named, so some "other" stations had troops. Anacostia (Guard
patrols from about January 2026) is left out of both groups.

  .venv/bin/python scripts/metro.py   ->  out/metro.json
"""
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grid as G  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATION_M = 300
FIX = {"1402 Eye St NW": "McPherson Sq"}
VICTIM = r"ASSAULT|ROBBERY|THEFT|LARCENY|SEXUAL|SEX ABUSE|RAPE|FONDL|HOMICIDE|MURDER|BURGLARY|MOTOR VEHICLE THEFT"
VIOLENT = r"ROBBERY|ASSAULT WITH A DANGEROUS|AGGRAVATED ASSAULT|HOMICIDE|MURDER|SEXUAL ABUSE|RAPE"
START = pd.Timestamp("2025-08-11")
FIRST, LAST = "2023-08-01", "2026-08-31"   # two years before, as in the plan; --since-2022 for all months


def load():
    df = pd.read_csv(ROOT / "data/interim/mtpd_incidents.csv", parse_dates=["date"])
    st = json.loads((ROOT / "data/external/metro_stations.geojson").read_text())["features"]
    s = pd.DataFrame([{"station": f["properties"]["NAME"], "slon": f["geometry"]["coordinates"][0], "slat": f["geometry"]["coordinates"][1]} for f in st])
    ok = df.lat.notna()
    dy = (df.loc[ok, "lat"].values[:, None] - s.slat.values[None, :]) * 110_540
    dx = (df.loc[ok, "lon"].values[:, None] - s.slon.values[None, :]) * 111_320 * np.cos(np.radians(38.9))
    d = np.hypot(dx, dy)
    df["station"] = None
    df.loc[ok, "station"] = np.where(d.min(axis=1) <= STATION_M, s.station.values[d.argmin(axis=1)], None)
    for loc, stn in FIX.items():
        df.loc[df.location == loc, "station"] = stn
    # each offense on its own; an incident counts if any offense is a crime with a victim (assault on police is not)
    ex = df.offense.fillna("").str.upper().str.split(r" \| ").explode()
    ex = ex[~ex.str.contains("ON POLICE")]
    df["victim"] = ex.str.contains(VICTIM).groupby(level=0).any().reindex(df.index, fill_value=False)
    df["violent"] = ex.str.contains(VIOLENT).groupby(level=0).any().reindex(df.index, fill_value=False)
    return df[df.station.notna() & df.date.between(FIRST, LAST)], s


def panel(df, outcome, treated, controls):
    stations = list(treated) + list(controls)
    months = pd.date_range(FIRST, LAST, freq="MS")
    x = df[df[outcome] & df.station.isin(stations)]
    cnt = x.groupby(["station", x.date.dt.to_period("M").dt.to_timestamp()]).size()
    idx = pd.MultiIndex.from_product([stations, months], names=["station", "m"])
    p = cnt.reindex(idx, fill_value=0).rename("y").reset_index()
    def share_of(m):
        end = m + pd.offsets.MonthEnd(0)
        days = (end - max(m, START)).days + 1
        return min(1.0, max(0.0, days / m.days_in_month)) if end >= START else 0.0
    share = pd.Series([share_of(m) for m in months], index=months)
    p["treat"] = np.where(p.station.isin(treated), p.m.map(share), 0.0)
    p["mm"] = p.m.dt.strftime("%Y-%m")
    return p


def fit(p):
    p = p[p.groupby("station").y.transform("sum") > 0]
    f = pf.fepois("y ~ treat | station + mm", data=p, vcov={"CRV1": "station"})
    t = f.tidy().loc["treat"]
    rr, lo, hi = np.exp(t["Estimate"]), np.exp(t["Estimate"] - 1.96 * t["Std. Error"]), np.exp(t["Estimate"] + 1.96 * t["Std. Error"])
    return {"rr": round(float(rr), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "p": round(float(t["Pr(>|t|)"]), 4)}


def event(df, outcome, treated, controls):
    """Quarterly leads and lags against the quarter before the deployment, and a joint test of the leads."""
    from scipy.stats import chi2
    p = panel(df, outcome, treated, controls)
    q = ((p.m.dt.year - 2025) * 12 + p.m.dt.month - 8) // 3          # quarter 0 starts Aug 2025
    p["rel"] = np.where(p.station.isin(treated), q, -1)
    p = p[p.groupby("station").y.transform("sum") > 0]
    f = pf.fepois("y ~ i(rel, ref=-1) | station + mm", data=p, vcov={"CRV1": "station"})
    t = f.tidy()
    import re
    rel = [int(re.findall(r"-?\d+", str(v))[-1]) for v in t.index]
    lead = [i for i, r in enumerate(rel) if r <= -2]
    b = f.coef().values[lead]
    V = f._vcov[np.ix_(lead, lead)]
    stat = float(b @ np.linalg.solve(V, b))
    return {"by_quarter": {str(r): round(float(np.exp(e)), 2) for r, e in zip(rel, t["Estimate"])},
            "lead_test_p": round(float(chi2.sf(stat, len(lead))), 4)}


def main():
    global FIRST
    if "--since-2022" in sys.argv:
        FIRST = "2022-01-01"
    df, s = load()
    guard = G.W1_STATIONS
    G.UPDATED = True
    _, _, pool = G.load()
    pool = sorted(pool)
    others = [x for x in s.station if x not in guard and x != "Anacostia"]
    res = {"guard_stations": guard, "comparison_all_others": others, "comparison_clean_pool": pool, "outcomes": {}}
    rng = np.random.default_rng(20261008)
    for k in ["victim", "violent"]:
        out = {}
        for name, ctrl in {"vs_all_other_dc_stations": others, "vs_stations_away_from_known_sites": pool}.items():
            out[name] = fit(panel(df, k, guard, ctrl))
        # simple before/after: Aug 11, 2025 - Aug 31, 2026 against the same dates a year earlier
        def chg(stns):
            x = df[df[k] & df.station.isin(stns)]
            a = x.date.between("2025-08-11", "2026-08-31").sum()
            b = x.date.between("2024-08-11", "2025-08-31").sum()
            return int(b), int(a), round(100 * (a / b - 1), 1)
        out["counts_year_before_to_guard_year"] = {"guard": chg(guard), "all_others": chg(others), "clean_pool": chg(pool)}
        # placebo: 12 stations from the clean pool, against the rest of the pool
        draws = []
        for _ in range(1000):
            fake = list(rng.choice(pool, size=len(guard) if len(pool) > 2 * len(guard) else len(pool) // 2, replace=False))
            rest = [x for x in pool if x not in fake]
            try:
                draws.append(fit(panel(df, k, fake, rest))["rr"])
            except Exception:
                pass
        draws = np.array(draws)
        real = out["vs_stations_away_from_known_sites"]["rr"]
        out["placebo"] = {"n": int(len(draws)), "placebo_size": int(len(guard) if len(pool) > 2 * len(guard) else len(pool) // 2),
                          "share_at_or_below_real": round(float((draws <= real).mean()), 3),
                          "quantiles": {str(q): round(float(np.quantile(draws, q)), 3) for q in [0.025, 0.5, 0.975]}}
        out["event_study"] = event(df, k, guard, pool)
        res["outcomes"][k] = out
        print(k, json.dumps(out))
    by = df[df.victim].groupby([df.station.isin(guard).map({True: "guard", False: "other"}), df.date.dt.year]).size().unstack(0)
    res["victim_by_year"] = by.to_dict()
    res["window_start"] = FIRST
    (ROOT / ("out/metro_since2022.json" if "--since-2022" in sys.argv else "out/metro.json")).write_text(json.dumps(res, indent=1, default=int))


if __name__ == "__main__":
    main()
