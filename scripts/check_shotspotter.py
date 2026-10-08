"""Could the August 2025 drop in gunshot detections be a sensor or system change rather than less gunfire?

Signatures checked:
  outage           runs of days with no detections
  lost coverage    ~1 km squares that were busy the year before and went silent
  reclassified     detections moving into the excluded "Probable gunfire" category
  every area       the drop in each of the six coverage areas, week by week
  independent      MPD reported gun crimes (no sensors involved), same weeks

  python scripts/check_shotspotter.py   ->  out/shotspotter_checks.json
"""
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
GUNSHOT = ["Single Gunshot", "Multiple Gunshots"]


def main():
    s = pd.read_csv(ROOT / "data/raw/shotspotter.csv", parse_dates=["DATETIME"]).drop_duplicates("ID")
    s["t"] = s.TYPE.str.replace("_", " ")
    g = s[s.t.isin(GUNSHOT)].copy()
    out = {"data_end": str(s.DATETIME.max())}

    d = g.set_index("DATETIME").resample("D").size()
    since = d.loc["2024-01-01":]
    zero = since[since == 0]
    out["zero_days_since_2024"] = [str(x.date()) for x in zero.index]

    pre = (g.DATETIME >= "2024-08-11") & (g.DATETIME <= "2025-06-30 23:59")
    post = (g.DATETIME >= "2025-08-11") & (g.DATETIME <= "2026-06-30 23:59")
    sq = (g.LATITUDE / 0.01).round().astype(int).astype(str) + "," + (g.LONGITUDE / 0.01).round().astype(int).astype(str)
    a, b = sq[pre].value_counts(), sq[post].value_counts()
    busy = a[a >= 20].index
    r = b.reindex(busy, fill_value=0) / a[busy]
    out["busy_squares"] = {"n": int(len(busy)), "went_silent": int((b.reindex(busy, fill_value=0) == 0).sum()),
                           "post_over_pre_quantiles": {str(q): round(float(v), 2) for q, v in r.quantile([.1, .25, .5, .75, .9]).items()}}

    m = s.groupby([s.DATETIME.dt.to_period("M").astype(str), "t"]).size().unstack(fill_value=0)
    out["monthly_by_type"] = m.loc["2025-05":"2025-12"].to_dict("index")

    g["area"] = g.SOURCE.str[-2:]
    w = g.set_index("DATETIME").groupby("area").resample("W-MON", label="left", closed="left").size().unstack(0).fillna(0)
    before = w.loc["2025-07-14":"2025-08-04"].sum()
    after = w.loc["2025-08-11":"2025-09-01"].sum()
    out["by_area_4wk_before_after"] = {k: [int(before[k]), int(after[k])] for k in w.columns}

    inc = pd.read_csv(ROOT / "data/raw/mpd_incidents_2015_2026.csv", parse_dates=["REPORT_DAT"], low_memory=False)
    gun = inc[inc.METHOD.eq("GUN") & inc.OFFENSE.isin(["HOMICIDE", "ASSAULT W/DANGEROUS WEAPON", "ROBBERY", "SEX ABUSE"])]
    wg = gun.set_index("REPORT_DAT").resample("W-MON", label="left", closed="left").size()
    ws = g.set_index("DATETIME").resample("W-MON", label="left", closed="left").size()
    both = pd.DataFrame({"gunshots": ws, "reported_gun_crimes": wg}).loc["2025-07-14":"2025-09-29"]
    out["weekly_gunshots_vs_reported_gun_crimes"] = {str(k.date()): {c: int(v) for c, v in row.items()} for k, row in both.iterrows()}

    (ROOT / "out/shotspotter_checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["zero_days_since_2024", "busy_squares", "by_area_4wk_before_after"]}, indent=1))


if __name__ == "__main__":
    main()
