# DC crime since the National Guard

Did crime in Washington, DC fall after the National Guard deployed on August 11, 2025, and did it fall more where the troops were?

**Page:** https://mngoh.github.io/DC-Crime-Since-National-Guard/

## Results

Violence fell by close to half after the deployment. It fell just as much where no troops were posted, so nothing in the data shows the troops themselves adding to the decline.

- First Guard year (Aug 11, 2025 to Aug 10, 2026) against the same dates averaged over the 10 years before: all reported crime −34% (the year before: −11%), violent crime −41% (−32%), homicide −47% (−5%).
- Homicides fell from 167 to 94. Gunshots detected by sensors fell 45% (Aug 11 to Jun 30). DC's murder drop was the second largest of 36 large police departments, September to June.
- Fall 2025, wards with Guard posts (2 and 6) against wards without: gunshots −57% and −60%. 2026, Anacostia (Guard patrols) against Ward 7 (none documented): −35% and −40%.

Caveats: federal agents arrived the same day; MPD's crime data is under investigation for downgraded reports; troop locations are reconstructed from news reports and Guard releases. See the page.

## Tests written down in advance

The [test plan](docs/test-plan.md) was committed before these ran. Every test is reported on the page.

- Near troop posts against comparable areas (about 800 half-kilometer squares, three waves): no reduction on any of six measures. Property crime rules out a reduction of 15% or more with either list of sites; other theft only with the Guard's September list. Gunshots: 0.94 (0.76 to 1.17).
- Homicide, robbery and sex abuse near posts: +43% against comparable areas with the planned sites, which holds after correcting for six measures; +30% with the September list, which does not. About half of the gap was opening before the troops came.
- Robustness checks from the plan (radius, start dates, holiday periods, the Mall, busier controls, negative binomial): no version shows a reduction near posts.
- Placebo posts: for gunshots, 93% of 1,000 random sets of Metro stations did better than the real Guard stations.
- Synthetic DC from other large agencies (the whole federal surge): property crime -22% (p = 0.04), murder -27% (p = 0.10), robbery +4% (p = 0.70).
- Metro stations (Metro Transit Police blotters): inconclusive. The Guard stations had been worsening for two years before August 2025.
- Drawdown (5,148 troops in July 2026 to 2,864 in October): citywide crime did not change against normal.
- A Guard daily update filed in court (DC v. Trump, ECF 83-1, Sept 12, 2025) listed more patrol sites than the plan had; the tests were rerun with them and both versions are reported.
- [Records requests](docs/records-requests.md) for the Guard's actual post list are drafted.

## Reproduce

```
python scripts/fetch_incidents.py     # MPD crime incidents, 2015 to today
python scripts/fetch_shotspotter.py   # ShotSpotter gunshots and Metro stations
# download the Real-Time Crime Index CSV to data/external/rtci_Jan17_Jun26.csv
python scripts/analyze.py             # 10-year baseline, trend model, national comparison
python scripts/wards.py               # wards, posts, Metro stations
python scripts/phases.py              # troops vs no troops, two phases
python scripts/check_shotspotter.py   # is the gunshot drop real
.venv/bin/python scripts/grid.py --ri 1000            # tests 1-2, planned sites
.venv/bin/python scripts/grid.py --updated --ri 1000  # same, with the Guard's Sept 12, 2025 list
.venv/bin/python scripts/grid_extra.py                # checks added after the plan (labeled)
.venv/bin/python scripts/synth.py                     # test 4, synthetic DC
python scripts/metro_parse.py && .venv/bin/python scripts/metro.py   # test 3, Metro blotters
.venv/bin/python scripts/drawdown.py 2026-07-13       # test 5
python scripts/build_page.py          # index.html
```

Needs pandas, numpy, scipy, statsmodels, shapely and pyfixest, and pdftotext for the Metro blotters.
