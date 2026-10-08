"""Test 5 of docs/test-plan.md: the September 2026 drawdown. Descriptive, because which posts were
dropped is not known.

Troops fell from 5,148 (July 10, 2026) to 2,921 (September 29, 2026). If troops were holding crime
down, crime should rise against its 10-year norm once they left. Compares the weeks from the first
documented withdrawal to the end of the data with the eight weeks before, citywide and in the cells
near Wave 1 to 3 posts (500 m), and the change near posts against the change elsewhere.
Gunshots are added when MPD publishes the third quarter of 2026.

  .venv/bin/python scripts/drawdown.py   ->  out/drawdown.json
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grid as G  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
WITHDRAW = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-09-01")   # first documented withdrawal
END = pd.Timestamp("2026-10-04")   # last complete day with little reporting lag
BASE = (pd.Timestamp("2015-08-11"), pd.Timestamp("2025-08-10"))
VIOLENT = ["HOMICIDE", "SEX ABUSE", "ASSAULT W/DANGEROUS WEAPON", "ROBBERY"]
OUT = {"total": None, "violent_ex_adw": ["HOMICIDE", "ROBBERY", "SEX ABUSE"], "gun_violent": "GUN", "property": "PROPERTY"}


def main():
    old = pd.read_csv(ROOT / "data/raw/mpd_incidents_2015_2026.csv", parse_dates=["REPORT_DAT"], low_memory=False)
    new = pd.read_csv(ROOT / "data/raw/mpd_incidents_2026_latest.csv", parse_dates=["REPORT_DAT"], low_memory=False)
    inc = pd.concat([old[old.REPORT_DAT < "2026-01-01"], new], ignore_index=True)
    inc["date"] = inc.REPORT_DAT.dt.normalize()
    x, y = G.to_xy(inc.LATITUDE, inc.LONGITUDE)
    inc["cell"] = (x // G.CELL).astype(int).astype(str) + "_" + (y // G.CELL).astype(int).astype(str)
    cells, _, _ = G.load()
    c = G.assign(cells, 500, G.START)
    near = set(c.loc[c.wave > 0, "cell"])
    inc["near"] = inc.cell.isin(near)

    def series(mask):
        d = inc[mask].groupby("date").size().reindex(pd.date_range("2015-01-01", END), fill_value=0)
        b = d.loc[BASE[0]:BASE[1]]
        norm = b.groupby(b.index.strftime("%m-%d")).mean()
        return d, norm

    before = (WITHDRAW - pd.Timedelta(weeks=8), WITHDRAW - pd.Timedelta(days=1))
    after = (WITHDRAW, END)
    res = {"withdrawal_start": str(WITHDRAW.date()), "before": [str(before[0].date()), str(before[1].date())],
           "after": [str(after[0].date()), str(after[1].date())], "outcomes": {}}
    for k, sel in OUT.items():
        if sel is None:
            m = pd.Series(True, index=inc.index)
        elif sel == "GUN":
            m = inc.OFFENSE.isin(VIOLENT) & (inc.METHOD == "GUN")
        elif sel == "PROPERTY":
            m = ~inc.OFFENSE.isin(VIOLENT)
        else:
            m = inc.OFFENSE.isin(sel)
        row = {}
        for area, am in {"citywide": pd.Series(True, index=inc.index), "near_posts": inc.near, "elsewhere": ~inc.near}.items():
            d, norm = series(m & am)

            def ratio(a, b):
                act = d.loc[a:b].sum()
                nm = norm.reindex(pd.date_range(a, b).strftime("%m-%d")).sum()
                return int(act), float(nm)
            ab, nb = ratio(*before)
            aa, na = ratio(*after)
            rel = (aa / na) / (ab / nb)
            se = np.sqrt(1 / aa + 1 / ab) if aa and ab else np.nan
            row[area] = {"before": ab, "before_vs_norm_pct": round(100 * (ab / nb - 1), 1), "after": aa,
                         "after_vs_norm_pct": round(100 * (aa / na - 1), 1), "change_vs_norm_pct": round(100 * (rel - 1), 1),
                         "ci95_pct": [round(100 * (rel * np.exp(-1.96 * se) - 1), 1), round(100 * (rel * np.exp(1.96 * se) - 1), 1)]}
        t, e = row["near_posts"], row["elsewhere"]
        dd = ((t["after"] / t["before"]) / (e["after"] / e["before"]))
        se = np.sqrt(1 / t["after"] + 1 / t["before"] + 1 / e["after"] + 1 / e["before"])
        row["near_vs_elsewhere_ratio"] = {"rr": round(float(dd), 3), "ci95": [round(float(dd * np.exp(-1.96 * se)), 3), round(float(dd * np.exp(1.96 * se)), 3)]}
        res["outcomes"][k] = row
        print(k, {a: row[a]["change_vs_norm_pct"] for a in ["citywide", "near_posts", "elsewhere"]}, row["near_vs_elsewhere_ratio"])
    (ROOT / "out/drawdown.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
