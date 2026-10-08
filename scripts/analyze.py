"""Reported crime in DC since the National Guard deployment, against a 10-year baseline.

Deployment: August 11, 2025 (Executive Order 14333; Guard and federal agents on the street from
August 11-12). A "deployment year" runs August 11 to August 10. The baseline is the 10 deployment
years before it, August 11, 2015 to August 10, 2025, so it contains no post-deployment days.

  python scripts/analyze.py   ->  out/results.json, out/monthly.csv, out/weekly.csv

Inputs
  data/raw/mpd_incidents_2015_2026.csv        scripts/fetch_incidents.py
  data/external/rtci_Jan17_Jun26.csv          Real-Time Crime Index (AH Datalytics), national sample
  data/external/wards_2022.geojson            DC GIS, Ward - 2022
  data/external/central_business_district.geojson   DC GIS, DDOT Central Business District
  ../DC-Assault/data/interim/victim_offenses.csv    FBI NIBRS for MPD, 2022-2025 (optional check)
"""
import json
import pathlib

import numpy as np
import pandas as pd
import shapely
import statsmodels.api as sm
from shapely.geometry import shape

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEPLOY = pd.Timestamp("2025-08-11")
END = pd.Timestamp("2026-09-30")  # last complete month; October 1-5, 2026 left out
BASE_START, BASE_END = pd.Timestamp("2015-08-11"), pd.Timestamp("2025-08-10")

CATS = {"HOMICIDE": "homicide", "SEX ABUSE": "sex_abuse", "ASSAULT W/DANGEROUS WEAPON": "adw",
        "ROBBERY": "robbery", "BURGLARY": "burglary", "MOTOR VEHICLE THEFT": "mvt",
        "THEFT F/AUTO": "theft_auto", "THEFT/OTHER": "theft_other", "ARSON": "arson"}
VIOLENT = ["homicide", "sex_abuse", "adw", "robbery"]
PROPERTY = ["burglary", "mvt", "theft_auto", "theft_other", "arson"]
SERIES = ["total", "violent", "property", "gun_violent", "violent_ex_adw"] + VIOLENT + PROPERTY
LABELS = {"total": "All reported crime", "violent": "Violent crime", "property": "Property crime",
          "gun_violent": "Violent crime with a gun",
          "violent_ex_adw": "Homicide, robbery and sex abuse", "homicide": "Homicide", "sex_abuse": "Sex abuse",
          "adw": "Assault with a dangerous weapon", "robbery": "Robbery", "burglary": "Burglary",
          "mvt": "Motor vehicle theft", "theft_auto": "Theft from auto", "theft_other": "Other theft",
          "arson": "Arson"}


def pct(a, b):
    return None if not b else round(100 * (a / b - 1), 1)


def load():
    df = pd.read_csv(ROOT / "data/raw/mpd_incidents_2015_2026.csv", parse_dates=["REPORT_DAT"], low_memory=False)
    df["cat"] = df.OFFENSE.map(CATS)
    assert df.cat.notna().all(), df.OFFENSE[df.cat.isna()].unique()
    df["date"] = df.REPORT_DAT.dt.normalize()
    df = df[df.date <= END].copy()
    # fixed geography: 2022 wards and the DDOT central business district, by point in polygon,
    # because the WARD field follows whichever ward map was in force when the report was taken
    wards = json.loads((ROOT / "data/external/wards_2022.geojson").read_text())["features"]
    ok = df.LATITUDE.between(38.7, 39.1) & df.LONGITUDE.between(-77.2, -76.8)
    df["ward22"] = pd.NA
    for f in wards:
        inside = ok & shapely.contains_xy(shape(f["geometry"]), df.LONGITUDE.values, df.LATITUDE.values)
        df.loc[inside, "ward22"] = int(f["properties"]["WARD"])
    cbd = shape(json.loads((ROOT / "data/external/central_business_district.geojson").read_text())["features"][0]["geometry"])
    df["downtown"] = ok & shapely.contains_xy(cbd, df.LONGITUDE.values, df.LATITUDE.values)
    return df, {"rows": int(len(df)), "no_ward": int(df.ward22.isna().sum()), "bad_coords": int((~ok).sum()),
                "downtown_share": round(float(df.downtown.mean()), 4)}


def daily(df):
    days = pd.date_range("2015-01-01", END, freq="D")
    d = df.groupby(["date", "cat"]).size().unstack(fill_value=0).reindex(days, fill_value=0)
    for c in CATS.values():
        if c not in d:
            d[c] = 0
    d["violent"] = d[VIOLENT].sum(axis=1)
    d["property"] = d[PROPERTY].sum(axis=1)
    d["total"] = d["violent"] + d["property"]
    d["violent_ex_adw"] = d[["homicide", "robbery", "sex_abuse"]].sum(axis=1)
    g = df[df.cat.isin(VIOLENT) & (df.METHOD == "GUN")].groupby("date").size()
    d["gun_violent"] = g.reindex(days, fill_value=0)
    return d[SERIES]


def norms(d):
    """Mean count on each calendar day over the 10 baseline deployment years."""
    b = d.loc[BASE_START:BASE_END]
    key = b.index.strftime("%m-%d")
    return b.groupby(key).mean()


def window_norm(nrm, start, end, cols=None):
    key = pd.date_range(start, end).strftime("%m-%d")
    return nrm.loc[key, cols] if cols is not None else nrm.loc[key]


def dy_bounds(y):
    return pd.Timestamp(f"{y}-08-11"), pd.Timestamp(f"{y + 1}-08-10")


def deployment_years(d):
    rows = {}
    for y in range(2015, 2026):
        s, e = dy_bounds(y)
        rows[y] = d.loc[s:e].sum()
    t = pd.DataFrame(rows).T
    out = {}
    for c in SERIES:
        base = t.loc[2015:2024, c]
        v, prev = t.loc[2025, c], t.loc[2024, c]
        out[c] = {"by_year": {int(k): int(x) for k, x in t[c].items()},
                  "avg10": round(float(base.mean()), 1), "min10": int(base.min()), "max10": int(base.max()),
                  "min10_year": int(base.idxmin()),
                  "deploy_year": int(v), "vs_avg10_pct": pct(v, base.mean()),
                  "year_before": int(prev), "year_before_vs_avg10_pct": pct(prev, base.mean()),
                  "vs_year_before_pct": pct(v, prev),
                  "rank_low_of_11": int((t[c] < v).sum() + 1)}
    return out


WINDOWS = {
    "since_deployment": ("2025-08-11", END, "Aug 11, 2025 to Sep 30, 2026"),
    "deploy_year_1": ("2025-08-11", "2026-08-10", "First year: Aug 11, 2025 to Aug 10, 2026"),
    "first_months": ("2025-08-11", "2025-12-31", "Aug 11 to Dec 31, 2025"),
    "first_30_days": ("2025-08-11", "2025-09-09", "Federal control of MPD: Aug 11 to Sep 9, 2025"),
    "y2026": ("2026-01-01", END, "Jan 1 to Sep 30, 2026"),
    "deploy_year_2": ("2026-08-11", END, "Second year so far: Aug 11 to Sep 30, 2026"),
    "pre_12m": ("2024-08-11", "2025-08-10", "12 months before: Aug 11, 2024 to Aug 10, 2025"),
    "pre_2025": ("2025-01-01", "2025-08-10", "2025 before deployment: Jan 1 to Aug 10, 2025"),
    "pre_3m": ("2025-05-11", "2025-08-10", "3 months before: May 11 to Aug 10, 2025"),
}


def windows(d, nrm):
    out = {}
    for k, (s, e, label) in WINDOWS.items():
        act = d.loc[s:e].sum()
        nm = window_norm(nrm, s, e).sum()
        out[k] = {"label": label, "start": str(pd.Timestamp(s).date()), "end": str(pd.Timestamp(e).date()),
                  "series": {c: {"actual": int(act[c]), "avg10": round(float(nm[c]), 1), "vs_avg10_pct": pct(act[c], nm[c])}
                             for c in SERIES}}
    # calendar reading of "this year vs the past 10 years": Jan 1 - Sep 30, 2026 against Jan 1 - Sep 30 of 2016-2025
    ytd = {}
    for c in SERIES:
        per = {y: int(d.loc[f"{y}-01-01":f"{y}-09-30", c].sum()) for y in range(2016, 2027)}
        avg = np.mean([per[y] for y in range(2016, 2026)])
        ytd[c] = {"by_year": per, "avg_2016_2025": round(float(avg), 1), "y2026": per[2026], "vs_avg_pct": pct(per[2026], avg),
                  "rank_low_of_11": int(sum(per[y] < per[2026] for y in per) + 1)}
    out["ytd_calendar"] = {"label": "Jan 1 to Sep 30, 2026 vs Jan 1 to Sep 30 of 2016-2025", "series": ytd}
    return out


def monthly(d, nrm):
    m = d.resample("MS").sum()
    key = d.index.strftime("%m-%d")
    nd = nrm.reindex(key).set_axis(d.index)
    nm = nd.resample("MS").sum()
    ratio = (m / nm).round(4)
    post_frac = pd.Series(d.index >= DEPLOY, index=d.index).resample("MS").mean()
    out = pd.concat({"count": m, "norm": nm.round(2), "ratio": ratio}, axis=1)
    out[("post", "frac")] = post_frac.round(4)
    return out


def weekly(d, nrm):
    key = d.index.strftime("%m-%d")
    nd = nrm.reindex(key).set_axis(d.index)
    # weeks aligned on the deployment day (Monday, August 11, 2025)
    w = d.resample("W-SUN", label="left", closed="left").sum()
    wn = nd.resample("W-SUN", label="left", closed="left").sum()
    w.index = w.index + pd.Timedelta(days=1)
    wn.index = wn.index + pd.Timedelta(days=1)
    keep = (w.index >= "2023-01-02") & (w.index + pd.Timedelta(days=6) <= END)
    return pd.concat({"count": w[keep], "norm": wn[keep].round(2), "ratio": (w[keep] / wn[keep]).round(4)}, axis=1)


def its(mon, series, start, trend=True):
    """Poisson model of monthly counts with the 10-year norm as offset, a linear trend and a step at deployment."""
    m = mon[(mon.index >= start)]
    y = m[("count", series)].values
    off = np.log(m[("norm", series)].values)
    t = np.arange(len(m)) / 12.0
    post = m[("post", "frac")].values
    X = np.column_stack([np.ones(len(m)), t, post]) if trend else np.column_stack([np.ones(len(m)), post])
    fit = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    b, se = fit.params[-1], fit.bse[-1]
    res = {"start": str(pd.Timestamp(start).date()), "trend": trend, "months": int(len(m)),
           "step_pct": round(100 * (np.exp(b) - 1), 1),
           "ci95": [round(100 * (np.exp(b - 1.96 * se) - 1), 1), round(100 * (np.exp(b + 1.96 * se) - 1), 1)],
           "p": float(round(fit.pvalues[-1], 4))}
    if trend:
        res["pre_trend_pct_per_year"] = round(100 * (np.exp(fit.params[1]) - 1), 1)
    return res


def its_all(mon):
    specs = [("2024-08-01", False), ("2024-01-01", True), ("2023-01-01", True), ("2023-08-01", True)]
    return {c: [its(mon, c, s, tr) for s, tr in specs] for c in ["total", "violent", "property", "gun_violent", "violent_ex_adw", "homicide",
                                                                 "adw", "robbery", "burglary", "mvt", "theft_auto", "theft_other"]}


def national(mon):
    """DC against the Real-Time Crime Index national sample (590 agencies, DC removed), through June 2026."""
    r = pd.read_csv(ROOT / "data/external/rtci_Jan17_Jun26.csv", low_memory=False)
    f = ["murder_total", "robbery_total", "assault_total", "burglary_total", "theft_total", "motor_total"]
    us = r[(r.id == "us") & (r["size"] == "all")].set_index(["year", "month"])[f + ["agencies"]]
    dcr = r[r.id == "DCMPD0000"].set_index(["year", "month"])[f].fillna(0)
    nat = us[f].sub(dcr.reindex(us.index).fillna(0))
    nat.index = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in nat.index])
    nat.columns = ["murder", "robbery", "assault", "burglary", "theft", "motor"]
    nat["violent"] = nat[["murder", "robbery", "assault"]].sum(axis=1)
    nat["murder_robbery"] = nat[["murder", "robbery"]].sum(axis=1)
    nat["property"] = nat[["burglary", "theft", "motor"]].sum(axis=1)
    c = mon["count"]
    dc = pd.DataFrame({"murder": c.homicide, "robbery": c.robbery, "assault": c.adw, "burglary": c.burglary,
                       "theft": c.theft_auto + c.theft_other, "motor": c.mvt})
    dc["violent"] = dc[["murder", "robbery", "assault"]].sum(axis=1)
    dc["murder_robbery"] = dc[["murder", "robbery"]].sum(axis=1)
    dc["property"] = dc[["burglary", "theft", "motor"]].sum(axis=1)
    dc = dc.loc["2017-01-01":"2026-06-01"]

    def span(x, a, b):
        return x.loc[a:b].sum()

    out = {"agencies": int(us.agencies.iloc[-1]), "last_month": "2026-06", "series": {}}
    # same ten calendar months (September to June) so seasonality cancels
    P = {"post": ("2025-09-01", "2026-06-01"), "pre": ("2024-09-01", "2025-06-01"), "prepre": ("2023-09-01", "2024-06-01")}
    base_years = [(f"{y}-09-01", f"{y + 1}-06-01") for y in range(2017, 2025)]  # 8 Sep-Jun seasons, 2017-18 to 2024-25
    for k in ["violent", "murder_robbery", "murder", "robbery", "assault", "property", "burglary", "theft", "motor"]:
        res = {}
        for name, x in [("dc", dc[k]), ("national", nat[k])]:
            post, pre, pp = (span(x, *P[p]) for p in ["post", "pre", "prepre"])
            avg8 = np.mean([span(x, a, b) for a, b in base_years])
            res[name] = {"post": float(post), "pre": float(pre), "prepre": float(pp),
                         "post_vs_pre_pct": pct(post, pre), "pre_vs_prepre_pct": pct(pre, pp),
                         "post_vs_avg8_pct": pct(post, avg8), "pre_vs_avg8_pct": pct(pre, avg8)}
        d_post = np.log(res["dc"]["post"] / res["dc"]["pre"]) - np.log(res["national"]["post"] / res["national"]["pre"])
        d_pre = np.log(res["dc"]["pre"] / res["dc"]["prepre"]) - np.log(res["national"]["pre"] / res["national"]["prepre"])
        res["dc_minus_national_post_pts"] = round(100 * (np.exp(d_post) - 1), 1)
        res["dc_minus_national_pre_pts"] = round(100 * (np.exp(d_pre) - 1), 1)
        res["did_pct"] = round(100 * (np.exp(d_post - d_pre) - 1), 1)
        out["series"][k] = res
    # monthly year-over-year change, DC and national
    yoy = pd.DataFrame({f"{n}_{k}": (x[k] / x[k].shift(12) - 1).round(4) for n, x in [("dc", dc), ("national", nat)]
                        for k in ["violent", "property", "murder", "robbery", "motor"]})
    out["monthly_yoy"] = yoy.loc["2023-09-01":].reset_index(names="month").assign(month=lambda z: z.month.dt.strftime("%Y-%m")).to_dict("records")
    return out, r


def big_cities(r, dc_mon):
    """Each large agency's change, September-June 2025-26 against 2024-25, and the change in that change."""
    a = r[(r["size"] == "all") & (r.agencies == 1)].copy()
    a["m"] = pd.to_datetime(a.year.astype(str) + "-" + a.month.astype(str) + "-01")
    a = a[a.population >= 250_000]
    names = {"murder": "murder_total", "robbery": "robbery_total", "motor": "motor_total", "burglary": "burglary_total", "theft": "theft_total"}
    out = {}
    windows = {"post": ("2025-09-01", "2026-06-01"), "pre": ("2024-09-01", "2025-06-01"), "prepre": ("2023-09-01", "2024-06-01")}
    for k, col in names.items():
        rows = []
        for aid, g in a.groupby("id"):
            g = g.set_index("m")[col].loc["2023-09-01":"2026-06-01"]
            if len(g) < 34 or g.isna().any() or (g == 0).sum() > 6:
                continue
            s = {w: g.loc[x:y].sum() for w, (x, y) in windows.items()}
            if min(s.values()) < 30:
                continue
            rows.append({"id": aid, "post_pct": pct(s["post"], s["pre"]), "pre_pct": pct(s["pre"], s["prepre"]),
                         "accel": round(100 * (np.log(s["post"] / s["pre"]) - np.log(s["pre"] / s["prepre"])), 1)})
        t = pd.DataFrame(rows)
        dcrow = t[t.id == "DCMPD0000"]
        others = t[t.id != "DCMPD0000"]
        if dcrow.empty:
            continue
        dcv = dcrow.iloc[0]
        out[k] = {"n_agencies": int(len(t)), "dc_post_pct": float(dcv.post_pct), "dc_pre_pct": float(dcv.pre_pct),
                  "dc_accel_logpts": float(dcv.accel),
                  "median_post_pct": float(others.post_pct.median()), "median_accel_logpts": float(others.accel.median()),
                  "dc_rank_post_drop": int((others.post_pct < dcv.post_pct).sum() + 1),
                  "dc_rank_accel": int((others.accel < dcv.accel).sum() + 1)}
    return out


def by_area(df, nrm_unused=None):
    """Deployment year vs the 10-year average and the year before, by 2022 ward and downtown."""
    out = {}
    groups = {"violent": df.cat.isin(VIOLENT), "property": df.cat.isin(PROPERTY),
              "gun_violent": df.cat.isin(VIOLENT) & (df.METHOD == "GUN"), "violent_ex_adw": df.cat.isin(["homicide", "robbery", "sex_abuse"]),
              "robbery": df.cat == "robbery", "mvt": df.cat == "mvt", "theft_auto": df.cat == "theft_auto", "theft_other": df.cat == "theft_other"}
    areas = {f"Ward {w}": df.ward22 == w for w in range(1, 9)}
    areas["Downtown (central business district)"] = df.downtown
    areas["Outside downtown"] = ~df.downtown
    for name, mask in areas.items():
        res = {}
        for grp, gmask in groups.items():
            s = df[mask & gmask]
            counts = {y: int(s.date.between(*dy_bounds(y)).sum()) for y in range(2015, 2026)}
            avg = np.mean([counts[y] for y in range(2015, 2025)])
            res[grp] = {"deploy_year": counts[2025], "year_before": counts[2024], "avg10": round(float(avg), 1),
                        "vs_avg10_pct": pct(counts[2025], avg), "year_before_vs_avg10_pct": pct(counts[2024], avg),
                        "vs_year_before_pct": pct(counts[2025], counts[2024])}
        out[name] = res
    return out


def nibrs_check():
    """Did assaults shift from aggravated to simple after August 11, 2025? MPD NIBRS through December 2025."""
    p = ROOT.parent / "DC-Assault/data/interim/victim_offenses.csv"
    if not p.exists():
        return None
    v = pd.read_csv(p, usecols=["agency", "incident_id", "incident_date", "offense_code"], low_memory=False)
    v = v[v.offense_code.isin(["13A", "13B", "13C"]) & v.agency.str.contains("Washington", na=False)]
    v["date"] = pd.to_datetime(v.incident_date)
    inc = v.drop_duplicates(["incident_id", "offense_code"])
    out = {"agencies": sorted(v.agency.unique().tolist()), "last_date": str(inc.date.max().date())}
    win = {"post": ("08-11", "12-31")}
    res = {}
    for code in ["13A", "13B", "13C"]:
        x = inc[inc.offense_code == code]
        per = {y: int(x.date.between(f"{y}-08-11", f"{y}-12-31").sum()) for y in [2022, 2023, 2024, 2025]}
        pre = {y: int(x.date.between(f"{y}-01-01", f"{y}-08-10").sum()) for y in [2022, 2023, 2024, 2025]}
        res[code] = {"aug11_dec31": per, "jan1_aug10": pre,
                     "post_vs_2024_pct": pct(per[2025], per[2024]), "pre_vs_2024_pct": pct(pre[2025], pre[2024]),
                     "post_vs_avg3_pct": pct(per[2025], np.mean([per[y] for y in [2022, 2023, 2024]])),
                     "pre_vs_avg3_pct": pct(pre[2025], np.mean([pre[y] for y in [2022, 2023, 2024]]))}
    out["by_code"] = res
    agg_share = {}
    for y in [2022, 2023, 2024, 2025]:
        for nm, (a, b) in {"jan1_aug10": ("01-01", "08-10"), "aug11_dec31": ("08-11", "12-31")}.items():
            a13 = res["13A"][nm][y]
            b13 = res["13B"][nm][y]
            agg_share[f"{y}_{nm}"] = round(a13 / (a13 + b13), 4)
    out["aggravated_share"] = agg_share
    return out


def main():
    df, geo = load()
    d = daily(df)
    nrm = norms(d)
    mon = monthly(d, nrm)
    wk = weekly(d, nrm)
    nat, rtci = national(mon)
    res = {
        "deployment_date": str(DEPLOY.date()), "data_end": str(END.date()),
        "baseline": "10 deployment years, Aug 11, 2015 to Aug 10, 2025",
        "labels": LABELS, "geo": geo,
        "deployment_years": deployment_years(d),
        "windows": windows(d, nrm),
        "its": its_all(mon),
        "national": nat,
        "big_cities": big_cities(rtci, mon),
        "areas": by_area(df),
        "nibrs_assault_check": nibrs_check(),
    }
    (ROOT / "out/results.json").write_text(json.dumps(res, indent=1, default=float))
    mon.columns = [f"{a}_{b}" for a, b in mon.columns]
    mon.to_csv(ROOT / "out/monthly.csv", index_label="month")
    wk.columns = [f"{a}_{b}" for a, b in wk.columns]
    wk.to_csv(ROOT / "out/weekly.csv", index_label="week_start")
    print("wrote out/results.json, out/monthly.csv, out/weekly.csv")


if __name__ == "__main__":
    main()
