# DC crime since the National Guard

Did crime in Washington, DC fall after the National Guard deployed on August 11, 2025, and did it fall more where the troops were?

**Page:** https://mngoh.github.io/DC-Crime-Since-National-Guard/

## Results

Violence fell by close to half after the deployment. It fell just as much where no troops were posted, so nothing in the data shows the troops themselves adding to the decline.

- First Guard year (Aug 11, 2025 to Aug 10, 2026) against the same dates averaged over the 10 years before: all reported crime −34% (the year before: −11%), violent crime −41% (−32%), homicide −47% (−5%).
- Homicides fell from 167 to 94. Gunshots detected by sensors fell 45% (Aug 11 to Jun 30). DC's murder drop was the second largest of 36 large police departments, September to June.
- Fall 2025, wards with Guard posts (2 and 6) against wards without: gunshots −57% and −60%. 2026, Anacostia (Guard patrols) against Ward 7 (none documented): −35% and −40%.

Caveats: federal agents arrived the same day; MPD's crime data is under investigation for downgraded reports; troop locations are reconstructed from news reports and Guard releases. See the page.

## Reproduce

```
python scripts/fetch_incidents.py     # MPD crime incidents, 2015 to today
python scripts/fetch_shotspotter.py   # ShotSpotter gunshots and Metro stations
# download the Real-Time Crime Index CSV to data/external/rtci_Jan17_Jun26.csv
python scripts/analyze.py             # 10-year baseline, trend model, national comparison
python scripts/wards.py               # wards, posts, Metro stations
python scripts/phases.py              # troops vs no troops, two phases
python scripts/check_shotspotter.py   # is the gunshot drop real
python scripts/build_page.py          # index.html
```

Needs pandas, numpy, statsmodels and shapely.
