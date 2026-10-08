"""Troops vs no troops, in two phases, because the Guard's posts moved.

Phase 1, Aug 11 - Nov 26, 2025 (to the shooting of two Guard members, after which patrols and
troop numbers changed). Posts were downtown, on the Mall, at 10 Metro stations, Union Station and
Navy Yard, and by November also the H Street corridor, Dupont Circle and 14th Street (federal court
opinion, Nov 20, 2025). Test: Wards 2 and 6 against Wards 3, 4, 5, 7, 8 (Ward 1 is left out
because 14th Street runs through it; it is added back as a check), and within 500 m of a post
against the rest of the city.

Phase 2, Jan 1 - Jun 30, 2026. Guard "presence patrols" in Anacostia (Florida Guard, then Georgia
Guard from January 2026; DVIDS, Jan 30, 2026). No Guard patrols are documented in Ward 7. Test: MPD's
7th District (Ward 8 east of the river, including Historic Anacostia) against the 6th District
(mostly Ward 7 east of the river), the two neighboring districts with similar crime. Gunshots use
the matching ShotSpotter coverage areas (7D and 6D). Check: the same comparison for Aug 11 - Nov 26,
2025, when federal agents were in both districts but Guard patrols in Anacostia are not documented.

Each comparison is the Guard period against the same dates a year earlier, with the year before
that as a placebo (did the groups already diverge?). Ratios above 1: the troop group fell less.
Ranges are block bootstraps: 14-day blocks resampled within each period, both groups sharing the
same blocks, 4,000 resamples.

  python scripts/phases.py   ->  out/phases.json
"""
import json
import pathlib

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

ROOT = pathlib.Path(__file__).resolve().parent.parent
VIOLENT = ["HOMICIDE", "SEX ABUSE", "ASSAULT W/DANGEROUS WEAPON", "ROBBERY"]
STATIONS = ["Foggy Bottom-GWU", "Smithsonian", "Eastern Market", "Stadium Armory", "Waterfront", "McPherson Sq",
            "L'Enfant Plaza", "Gallery Pl-Chinatown", "Metro Center", "NoMa - Gallaudet U", "Union Station", "Navy Yard - Ballpark"]
# approximate centers of the other places named as Guard posts by November 2025
OTHER_SITES = {"Washington Monument": (38.8895, -77.0353), "Lincoln Memorial": (38.8893, -77.0502),
               "Dupont Circle": (38.9096, -77.0434), "14th and U St NW": (38.9170, -77.0320),
               "H St NE at 8th": (38.9002, -76.9950), "Georgetown, M St at Wisconsin": (38.9050, -77.0627)}
NEAR_M = 500
BLOCK_DAYS = 14


def metres(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 110_540
    dx = (lon2 - lon1) * 111_320 * np.cos(np.radians((lat1 + lat2) / 2))
    return np.hypot(dx, dy)


def load():
    wpoly = {int(f["properties"]["WARD"]): shape(f["geometry"])
             for f in json.loads((ROOT / "data/external/wards_2022.geojson").read_text())["features"]}
    st = json.loads((ROOT / "data/external/metro_stations.geojson").read_text())["features"]
    sites = [(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in st if f["properties"]["NAME"] in STATIONS]
    assert len(sites) == len(STATIONS)
    sites += list(OTHER_SITES.values())

    inc = pd.read_csv(ROOT / "data/raw/mpd_incidents_2015_2026.csv", parse_dates=["REPORT_DAT"], low_memory=False)
    inc = inc.rename(columns={"REPORT_DAT": "when", "LATITUDE": "lat", "LONGITUDE": "lon"})
    inc["district"] = inc.DISTRICT.map(lambda d: f"{int(d)}D" if pd.notna(d) else None)
    shots = pd.read_csv(ROOT / "data/raw/shotspotter.csv", parse_dates=["DATETIME"]).drop_duplicates("ID")
    shots = shots[shots.TYPE.str.replace("_", " ").isin(["Single Gunshot", "Multiple Gunshots"])]
    shots = shots.rename(columns={"DATETIME": "when", "LATITUDE": "lat", "LONGITUDE": "lon"})
    shots["district"] = shots.SOURCE.str[-2:]
    events = {"gunshots": shots,
              "homicide": inc[inc.OFFENSE == "HOMICIDE"],
              "gun_violent": inc[inc.OFFENSE.isin(VIOLENT) & (inc.METHOD == "GUN")],
              "violent": inc[inc.OFFENSE.isin(VIOLENT)],
              "property": inc[~inc.OFFENSE.isin(VIOLENT)],
              "theft_other": inc[inc.OFFENSE == "THEFT/OTHER"],
              "theft_auto": inc[inc.OFFENSE == "THEFT F/AUTO"]}
    cbd = shape(json.loads((ROOT / "data/external/central_business_district.geojson").read_text())["features"][0]["geometry"])
    for k, e in events.items():
        e = e[e.when >= "2023-01-01"][["when", "lat", "lon", "district"]].copy()
        e["ward"] = -1
        for w, g in wpoly.items():
            e.loc[shapely.contains_xy(g, e.lon.values, e.lat.values), "ward"] = w
        d = np.stack([metres(e.lat.values, e.lon.values, la, lo) for la, lo in sites])
        e["near_post"] = d.min(axis=0) <= NEAR_M
        e["downtown"] = shapely.contains_xy(cbd, e.lon.values, e.lat.values)
        events[k] = e
    return events


def window(year, md):
    a, b = md
    s = pd.Timestamp(f"{year}-{a}")
    e = pd.Timestamp(f"{year + (1 if b < a else 0)}-{b}")
    return s, e


def blocks(e, mask, s, end):
    n = (end - s).days + 1
    sub = e[mask & (e.when >= s) & (e.when < end + pd.Timedelta(days=1))]
    day = (sub.when.dt.normalize() - s).dt.days.values
    x = np.bincount(day, minlength=n)[:n]
    return np.array([x[i:i + BLOCK_DAYS].sum() for i in range(0, n, BLOCK_DAYS)])


def compare(e, tmask, cmask, md, years, reps=4000, seed=7):
    """years = (post, pre, pre2) start years for the same calendar window."""
    rng = np.random.default_rng(seed)
    B = {y: (blocks(e, tmask, *window(y, md)), blocks(e, cmask, *window(y, md))) for y in years}
    tot = {y: (int(B[y][0].sum()), int(B[y][1].sum())) for y in years}
    post, pre, pre2 = years

    def pct(a, b):
        return None if not b else round(100 * (a / b - 1), 1)

    out = {"troops": {"post": tot[post][0], "pre": tot[pre][0], "pre2": tot[pre2][0], "change_pct": pct(tot[post][0], tot[pre][0]),
                      "placebo_change_pct": pct(tot[pre][0], tot[pre2][0])},
           "no_troops": {"post": tot[post][1], "pre": tot[pre][1], "pre2": tot[pre2][1], "change_pct": pct(tot[post][1], tot[pre][1]),
                         "placebo_change_pct": pct(tot[pre][1], tot[pre2][1])}}
    for name, (a, b) in {"ratio": (post, pre), "placebo": (pre, pre2)}.items():
        ta, ca = tot[a]
        tb, cb = tot[b]
        if min(ta, ca, tb, cb) == 0:
            out[name] = None
            continue
        point = (ta / tb) / (ca / cb)
        draws = []
        for _ in range(reps):
            ia = rng.integers(0, len(B[a][0]), len(B[a][0]))
            ib = rng.integers(0, len(B[b][0]), len(B[b][0]))
            x = [B[a][0][ia].sum(), B[b][0][ib].sum(), B[a][1][ia].sum(), B[b][1][ib].sum()]
            if min(x) > 0:
                draws.append((x[0] / x[1]) / (x[2] / x[3]))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out[name] = {"point": round(point, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}
    return out


def main():
    events = load()
    P1, P2 = ("08-11", "11-26"), ("01-01", "06-30")
    res = {"phase1": {"window": "Aug 11 - Nov 26; 2025 vs 2024, placebo 2024 vs 2023"},
           "phase2": {"window": "Jan 1 - Jun 30; 2026 vs 2025, placebo 2025 vs 2024"},
           "phase2_fall_check": {"window": "Aug 11 - Nov 26; 2025 vs 2024, placebo 2024 vs 2023; 7D vs 6D"},
           "sites": STATIONS + list(OTHER_SITES)}
    for k, e in events.items():
        w = e.ward
        res["phase1"][k] = {
            "wards_2_6_vs_3_4_5_7_8": compare(e, w.isin([2, 6]), w.isin([3, 4, 5, 7, 8]), P1, (2025, 2024, 2023)),
            "wards_2_6_vs_all_others": compare(e, w.isin([2, 6]), w.isin([1, 3, 4, 5, 7, 8]), P1, (2025, 2024, 2023)),
            "near_post_vs_rest": compare(e, e.near_post, ~e.near_post & (w > 0), P1, (2025, 2024, 2023)),
            "downtown_vs_rest": compare(e, e.downtown, ~e.downtown & (w > 0), P1, (2025, 2024, 2023)),
            "downtown_vs_rest_first_year": compare(e, e.downtown, ~e.downtown & (w > 0), ("08-11", "08-10"), (2025, 2024, 2023)),
        }
        res["phase2"][k] = {"7D_vs_6D": compare(e, e.district == "7D", e.district == "6D", P2, (2026, 2025, 2024))}
        res["phase2_fall_check"][k] = {"7D_vs_6D": compare(e, e.district == "7D", e.district == "6D", P1, (2025, 2024, 2023))}
    (ROOT / "out/phases.json").write_text(json.dumps(res, indent=1))
    print("wrote out/phases.json")


if __name__ == "__main__":
    main()
