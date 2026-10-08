"""Fetch every MPD public crime incident, 2015 to today.

Source: DC Open Data "Crime Incidents in <year>", served from the MPD feed on maps2.dcgis.dc.gov.
Covers the nine DC Code categories MPD publishes: homicide, sex abuse, assault with a dangerous
weapon, robbery, burglary, motor vehicle theft, theft from auto, other theft, arson. The current
year's layer is refreshed daily and is preliminary.

  python scripts/fetch_incidents.py   ->  data/raw/mpd_incidents_2015_2026.csv
"""
import datetime as dt
import json
import pathlib
import urllib.parse
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVICE = "https://maps2.dcgis.dc.gov/dcgis/rest/services/FEEDS/MPD/MapServer"
LAYERS = {2015: 27, 2016: 26, 2017: 38, 2018: 0, 2019: 1, 2020: 2, 2021: 3, 2022: 4, 2023: 5, 2024: 6, 2025: 7, 2026: 41}
FIELDS = ["CCN", "REPORT_DAT", "START_DATE", "END_DATE", "OFFENSE", "METHOD", "SHIFT", "WARD", "ANC",
          "DISTRICT", "PSA", "NEIGHBORHOOD_CLUSTER", "CENSUS_TRACT", "BID", "BLOCK", "LATITUDE", "LONGITUDE"]
PAGE = 1000


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "dc-crime-guard-analysis/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def layer(year, lid):
    rows, offset = [], 0
    while True:
        q = {"where": "1=1", "outFields": ",".join(FIELDS), "returnGeometry": "false", "orderByFields": "OBJECTID",
             "resultOffset": offset, "resultRecordCount": PAGE, "f": "json"}
        d = get(f"{SERVICE}/{lid}/query?" + urllib.parse.urlencode(q))
        feats = d.get("features", [])
        rows += [f["attributes"] for f in feats]
        if len(feats) < PAGE and not d.get("exceededTransferLimit"):
            break
        offset += len(feats)
    df = pd.DataFrame(rows)
    df.insert(0, "layer_year", year)
    for c in ["REPORT_DAT", "START_DATE", "END_DATE"]:
        df[c] = pd.to_datetime(df[c], unit="ms", errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M")
    print(f"{year}: {len(df):,} incidents")
    return df


def main():
    df = pd.concat([layer(y, lid) for y, lid in LAYERS.items()], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(["CCN", "OFFENSE"])
    if before != len(df):
        print(f"dropped {before - len(df):,} rows repeated across yearly layers")
    dest = ROOT / "data" / "raw" / "mpd_incidents_2015_2026.csv"
    df.to_csv(dest, index=False)
    (ROOT / "data" / "raw" / "FETCHED.txt").write_text(f"fetched {dt.datetime.now().isoformat(timespec='minutes')} from {SERVICE}\n")
    print("wrote", dest, f"({len(df):,} rows)")


if __name__ == "__main__":
    main()
