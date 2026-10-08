"""Fetch MPD's ShotSpotter gunshot detections (2014 to latest quarterly update) and DC Metro stations.

Gunshot detections do not depend on anyone calling the police or on how an officer classifies a
report, so they test violent crime without the recording problems in the incident data.

  python scripts/fetch_shotspotter.py  ->  data/raw/shotspotter.csv, data/external/metro_stations.geojson
"""
import json
import pathlib
import urllib.parse
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHOTS = "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Public_Safety_WebMercator/MapServer/29"
STATIONS = "https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Transportation_Rail_Bus_WebMercator/MapServer/52"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "dc-crime-guard-analysis/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def main():
    rows, offset = [], 0
    while True:
        q = {"where": "1=1", "outFields": "ID,TYPE,SOURCE,DATETIME,LATITUDE,LONGITUDE", "returnGeometry": "false",
             "orderByFields": "OBJECTID", "resultOffset": offset, "resultRecordCount": 1000, "f": "json"}
        d = json.loads(get(f"{SHOTS}/query?" + urllib.parse.urlencode(q)))
        feats = d.get("features", [])
        rows += [f["attributes"] for f in feats]
        if len(feats) < 1000 and not d.get("exceededTransferLimit"):
            break
        offset += len(feats)
    df = pd.DataFrame(rows)
    df["DATETIME"] = pd.to_datetime(df.DATETIME, unit="ms", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M")
    df.to_csv(ROOT / "data/raw/shotspotter.csv", index=False)
    print(f"{len(df):,} detections, {df.DATETIME.min()} to {df.DATETIME.max()}", df.TYPE.value_counts().to_dict())
    q = {"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson"}
    (ROOT / "data/external/metro_stations.geojson").write_bytes(get(f"{STATIONS}/query?" + urllib.parse.urlencode(q)))


if __name__ == "__main__":
    main()
