"""Parse Metro Transit Police monthly blotters (wmata.com, Jan 2022 - Aug 2026) into incidents,
and place DC incidents at Metro stations.

The blotters list each incident's date, time, disposition, street address and offenses (prefixed
by jurisdiction, e.g. "DC - ASSAULT"). There is no station field; station incidents carry the
station's street address. DC addresses (ending NW, NE, SE or SW) are geocoded with the US Census
geocoder and assigned to the nearest DC Metro station within 200 m. Intersections, which are mostly
bus stops, are not geocoded. WMATA notes the blotter is not a complete list.

  python scripts/metro_parse.py   ->  data/interim/mtpd_incidents.csv, data/interim/mtpd_geocodes.csv
"""
import io
import json
import pathlib
import re
import subprocess
import time
import urllib.request
import uuid

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/mtpd"
OUT = ROOT / "data/interim"
LINE = re.compile(r"^\s*(\d\d/\d\d/\d{4})\s+(\d\d:\d\d:\d\d)\s+(.+?)\s{2,}(.+?)(?:\s{2,}(.*))?$")
CONT = re.compile(r"^\s{20,}(\S.*)$")
QUAD = re.compile(r"\b(NW|NE|SE|SW)\b\.?\s*$", re.I)
STATION_M = 200


def parse_pdf(path):
    txt = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
    rows, cur = [], None
    for ln in txt.splitlines():
        m = LINE.match(ln)
        if m:
            date, tm, dispo, loc, off = m.groups()
            cur = {"date": date, "time": tm, "dispo": dispo.strip(), "location": loc.strip(), "offenses": [off.strip()] if off else []}
            rows.append(cur)
            continue
        c = CONT.match(ln)
        if c and cur is not None and not re.search(r"Blotter List|Metro Transit Police|Date Range|For Law Enforcement|Bus - Rail|\d+ of \d+", ln):
            cur["offenses"].append(c.group(1).strip())
    return rows


def geocode(addresses):
    """Census batch geocoder, 1,000 at a time."""
    out = []
    addrs = list(addresses)
    for i in range(0, len(addrs), 1000):
        chunk = addrs[i:i + 1000]
        csv = "".join(f'{j},"{a}",Washington,DC,\n' for j, a in enumerate(chunk))
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"benchmark\"\r\n\r\nPublic_AR_Current\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"addressFile\"; filename=\"a.csv\"\r\nContent-Type: text/csv\r\n\r\n{csv}\r\n"
                f"--{boundary}--\r\n").encode()
        req = urllib.request.Request("https://geocoding.geo.census.gov/geocoder/locations/addressbatch", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            res = pd.read_csv(io.StringIO(r.read().decode()), header=None, names=["id", "input", "match", "type", "matched", "coords", "tiger", "side"])
        for _, x in res.iterrows():
            lon = lat = np.nan
            if x["match"] == "Match" and isinstance(x["coords"], str):
                lon, lat = map(float, x["coords"].split(","))
            out.append({"location": chunk[int(x["id"])], "lat": lat, "lon": lon})
        time.sleep(1)
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(RAW.glob("blotter_*.pdf")):
        r = parse_pdf(f)
        for x in r:
            x["file"] = f.stem
        rows += r
        print(f.stem, len(r))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.date, format="%m/%d/%Y")
    df["offense"] = df.offenses.map(lambda o: " | ".join(o))
    df = df.drop(columns="offenses")
    df["dc_address"] = df.location.str.contains(QUAD) & ~df.location.str.contains("/")
    geo_path = OUT / "mtpd_geocodes.csv"
    have = pd.read_csv(geo_path) if geo_path.exists() else pd.DataFrame(columns=["location", "lat", "lon"])
    todo = sorted(set(df.loc[df.dc_address, "location"]) - set(have.location))
    if todo:
        have = pd.concat([have, geocode(todo)], ignore_index=True)
        have.to_csv(geo_path, index=False)
    have = have.astype({"lat": float, "lon": float})
    df = df.merge(have, on="location", how="left")
    st = json.loads((ROOT / "data/external/metro_stations.geojson").read_text())["features"]
    s = pd.DataFrame([{"station": f["properties"]["NAME"], "slon": f["geometry"]["coordinates"][0], "slat": f["geometry"]["coordinates"][1]} for f in st])
    ok = df.lat.notna()
    dy = (df.loc[ok, "lat"].values[:, None] - s.slat.values[None, :]) * 110_540
    dx = (df.loc[ok, "lon"].values[:, None] - s.slon.values[None, :]) * 111_320 * np.cos(np.radians(38.9))
    d = np.hypot(dx, dy)
    df["station"] = None
    df.loc[ok, "station"] = np.where(d.min(axis=1) <= STATION_M, s.station.values[d.argmin(axis=1)], None)
    df.to_csv(OUT / "mtpd_incidents.csv", index=False)
    print(f"{len(df):,} incidents, {df.dc_address.sum():,} at DC street addresses, {df.station.notna().sum():,} placed at DC stations")


if __name__ == "__main__":
    main()
