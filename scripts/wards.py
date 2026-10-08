"""Did crime fall less where the Guard was posted? Wards with troops against wards without.

Guard posts (WJLA, Aug 2025: troops at 10 Metrorail stations; plus Union Station, Navy Yard /
Nationals Park and the National Mall, which every account lists):
  Foggy Bottom, Smithsonian, Eastern Market, Stadium Armory, Waterfront, McPherson Sq,
  L'Enfant Plaza, Gallery Place, Metro Center, NoMa-Gallaudet, Union Station, Navy Yard,
  Washington Monument, Lincoln Memorial.

Window: August 11 to June 30, because the gunshot detections end June 30, 2026. Every period
uses the same dates, so seasons match:
  Guard period    Aug 11, 2025 - Jun 30, 2026
  year before     Aug 11, 2024 - Jun 30, 2025
  two years before  Aug 11, 2023 - Jun 30, 2024   (placebo: were troop wards already diverging?)
  10-year average   Aug 11, Y - Jun 30, Y+1 for Y = 2015-2024

  python scripts/wards.py   ->  out/wards.json
"""
import json
import pathlib

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

ROOT = pathlib.Path(__file__).resolve().parent.parent
GUARD_STATIONS = ["Foggy Bottom-GWU", "Smithsonian", "Eastern Market", "Stadium Armory", "Waterfront", "McPherson Sq",
                  "L'Enfant Plaza", "Gallery Pl-Chinatown", "Metro Center", "NoMa - Gallaudet U", "Union Station",
                  "Navy Yard - Ballpark"]
EXTRA_SITES = {"Washington Monument": (38.8895, -77.0353), "Lincoln Memorial": (38.8893, -77.0502)}
NEAR_M = 500       # "near a Guard post": within 500 m, about six blocks
STATION_M = 400    # station areas for the station-to-station comparison
VIOLENT = ["HOMICIDE", "SEX ABUSE", "ASSAULT W/DANGEROUS WEAPON", "ROBBERY"]


def period_of(dates):
    p = pd.Series(pd.NA, index=dates.index, dtype="object")
    for y in range(2015, 2026):
        s, e = pd.Timestamp(f"{y}-08-11"), pd.Timestamp(f"{y + 1}-06-30 23:59")
        p[(dates >= s) & (dates <= e)] = y
    return p


def metres(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 110_540
    dx = (lon2 - lon1) * 111_320 * np.cos(np.radians((lat1 + lat2) / 2))
    return np.hypot(dx, dy)


def load():
    wards = json.loads((ROOT / "data/external/wards_2022.geojson").read_text())["features"]
    wpoly = {int(f["properties"]["WARD"]): shape(f["geometry"]) for f in wards}

    def ward_of(lat, lon):
        out = np.full(len(lat), -1)
        for w, g in wpoly.items():
            out[shapely.contains_xy(g, lon, lat)] = w
        return out

    st = json.loads((ROOT / "data/external/metro_stations.geojson").read_text())["features"]
    stations = pd.DataFrame([{"name": f["properties"]["NAME"], "lon": f["geometry"]["coordinates"][0],
                              "lat": f["geometry"]["coordinates"][1]} for f in st])
    missing = set(GUARD_STATIONS) - set(stations.name)
    assert not missing, missing
    stations["guard"] = stations.name.isin(GUARD_STATIONS)
    stations["ward"] = ward_of(stations.lat.values, stations.lon.values)
    sites = pd.concat([stations[stations.guard][["name", "lat", "lon"]],
                       pd.DataFrame([{"name": k, "lat": v[0], "lon": v[1]} for k, v in EXTRA_SITES.items()])], ignore_index=True)
    sites["ward"] = ward_of(sites.lat.values, sites.lon.values)

    inc = pd.read_csv(ROOT / "data/raw/mpd_incidents_2015_2026.csv", parse_dates=["REPORT_DAT"], low_memory=False)
    inc = inc.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon", "REPORT_DAT": "when"})
    shots = pd.read_csv(ROOT / "data/raw/shotspotter.csv", parse_dates=["DATETIME"]).rename(
        columns={"LATITUDE": "lat", "LONGITUDE": "lon", "DATETIME": "when"})
    # gunshots only, under both spellings the system has used; "Gunshot or firecracker" (to 2020)
    # and "Probable gunfire" (2021 on) are left out so the definition holds across years
    shots = shots[shots.TYPE.str.replace("_", " ").isin(["Single Gunshot", "Multiple Gunshots"])].drop_duplicates("ID")

    events = {
        "homicide": inc[inc.OFFENSE == "HOMICIDE"],
        "gunshots": shots,
        "gun_violent": inc[inc.OFFENSE.isin(VIOLENT) & (inc.METHOD == "GUN")],
        "violent": inc[inc.OFFENSE.isin(VIOLENT)],
        "property": inc[~inc.OFFENSE.isin(VIOLENT)],
    }
    for k, e in events.items():
        e = e[["when", "lat", "lon"]].copy()
        e["period"] = period_of(e.when)
        e = e[e.period.notna()].copy()
        e["ward"] = ward_of(e.lat.values, e.lon.values)
        d = np.stack([metres(e.lat.values, e.lon.values, la, lo) for la, lo in zip(sites.lat, sites.lon)])
        e["near_guard"] = d.min(axis=0) <= NEAR_M
        ds = np.stack([metres(e.lat.values, e.lon.values, la, lo) for la, lo in zip(stations.lat, stations.lon)])
        nearest = ds.argmin(axis=0)
        e["station_area"] = np.where(ds.min(axis=0) <= STATION_M, np.where(stations.guard.values[nearest], "guard", "other"), "none")
        events[k] = e
    return events, stations, sites


def summarize(e, mask):
    c = e[mask].groupby("period").size().reindex(range(2015, 2026), fill_value=0)
    avg10 = c.loc[2015:2024].mean()
    return {"guard_period": int(c[2025]), "year_before": int(c[2024]), "two_before": int(c[2023]), "avg10": round(float(avg10), 1),
            "vs_year_before_pct": None if not c[2024] else round(100 * (c[2025] / c[2024] - 1), 1),
            "vs_avg10_pct": None if not avg10 else round(100 * (c[2025] / avg10 - 1), 1),
            "year_before_vs_two_before_pct": None if not c[2023] else round(100 * (c[2024] / c[2023] - 1), 1)}


def contrast(t, c):
    """Ratio of the troop group's change to the comparison group's change, with an approximate Poisson 95% CI.
    Above 1: the troop group fell less (or rose more)."""
    out = {}
    for name, (a, b) in {"guard_vs_year_before": ("guard_period", "year_before"),
                         "placebo_year_before_vs_two_before": ("year_before", "two_before")}.items():
        n = [t[a], t[b], c[a], c[b]]
        if min(n) == 0:
            out[name] = None
            continue
        lr = np.log(t[a] / t[b]) - np.log(c[a] / c[b])
        se = np.sqrt(sum(1 / x for x in n))
        out[name] = {"ratio": round(float(np.exp(lr)), 3), "ci95": [round(float(np.exp(lr - 1.96 * se)), 3), round(float(np.exp(lr + 1.96 * se)), 3)]}
    # against the 10-year average: 10 baseline years, so the average's variance is avg/10
    lr = np.log(t["guard_period"] / t["avg10"]) - np.log(c["guard_period"] / c["avg10"])
    se = np.sqrt(1 / t["guard_period"] + 1 / (10 * t["avg10"]) + 1 / c["guard_period"] + 1 / (10 * c["avg10"]))
    out["guard_vs_avg10"] = {"ratio": round(float(np.exp(lr)), 3), "ci95": [round(float(np.exp(lr - 1.96 * se)), 3), round(float(np.exp(lr + 1.96 * se)), 3)]}
    return out


def block_bootstrap(e, tm, cm, reps=4000, block=28, seed=11):
    """Ratio of changes (troops / no troops) with a block bootstrap CI.

    The Poisson interval in contrast() assumes counts vary only by chance from day to day. Real
    counts swing more than that (a bad weekend, a feud, a heat wave) and those swings cluster in
    time, so it is too narrow. Here each period is cut into 4-week blocks, blocks are resampled
    with replacement, and the troop and no-troop groups share the same resampled blocks so that
    citywide swings hit both. 95% percentile interval over 4,000 resamples."""
    rng = np.random.default_rng(seed)
    blocks = {}
    for y in [2023, 2024, 2025]:
        start = pd.Timestamp(f"{y}-08-11")
        n = (pd.Timestamp(f"{y + 1}-06-30") - start).days + 1
        day = (e.when.dt.normalize() - start).dt.days
        inp = e.period == y
        t = np.bincount(day[inp & tm], minlength=n)[:n]
        c = np.bincount(day[inp & cm], minlength=n)[:n]
        edges = list(range(0, n, block))
        blocks[y] = np.array([[t[a:a + block].sum(), c[a:a + block].sum()] for a in edges])
    out = {}
    for name, (a, b) in {"guard_vs_year_before": (2025, 2024), "placebo_year_before_vs_two_before": (2024, 2023)}.items():
        ratios, skipped = [], 0
        for _ in range(reps):
            sa = blocks[a][rng.integers(0, len(blocks[a]), len(blocks[a]))].sum(axis=0)
            sb = blocks[b][rng.integers(0, len(blocks[b]), len(blocks[b]))].sum(axis=0)
            if sb.min() == 0 or sa.min() == 0:
                skipped += 1
                continue
            ratios.append((sa[0] / sb[0]) / (sa[1] / sb[1]))
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        out[name] = {"ci95": [round(float(lo), 3), round(float(hi), 3)], "reps_skipped_for_zero": skipped}
    return out


def main():
    events, stations, sites = load()
    site_wards = sites.groupby("ward").name.apply(list).to_dict()
    core = [2, 6]                      # where the posts cluster
    any_site = sorted(site_wards)      # any ward with at least one post
    res = {"sites_by_ward": {int(k): v for k, v in site_wards.items()},
           "window": "Aug 11 to Jun 30", "near_metres": NEAR_M, "station_metres": STATION_M,
           "groups": {"core": core, "any_site": any_site}, "measures": {}}
    for k, e in events.items():
        m = {"by_ward": {w: summarize(e, e.ward == w) for w in range(1, 9)}}
        splits = {"core_troop_wards": (e.ward.isin(core), e.ward.isin([w for w in range(1, 9) if w not in core])),
                  "any_site_wards": (e.ward.isin(any_site), e.ward.isin([w for w in range(1, 9) if w not in any_site])),
                  "near_guard_post": (e.near_guard, ~e.near_guard & (e.ward > 0)),
                  "guard_vs_other_stations": (e.station_area == "guard", e.station_area == "other")}
        for name, (tm, cm) in splits.items():
            t, c = summarize(e, tm), summarize(e, cm)
            m[name] = {"troops": t, "no_troops": c, "contrast": contrast(t, c), "bootstrap": block_bootstrap(e, tm, cm)}
        res["measures"][k] = m
    (ROOT / "out/wards.json").write_text(json.dumps(res, indent=1))
    print("wrote out/wards.json")


if __name__ == "__main__":
    main()
