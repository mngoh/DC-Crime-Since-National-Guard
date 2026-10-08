# Test plan: did crime fall more where National Guard troops were posted?

Written October 8, 2026, and committed before any test below was run. The commit timestamp is the record.

**What was already seen.** Ward-level and 500 m comparisons for August 2025 to June 2026 (`scripts/wards.py`, `scripts/phases.py`) found no clear difference between areas with and without troops. This plan was written after seeing those results, so it is not blind. What it fixes in advance is the specification of the new tests: units, dates, outcomes, model, thresholds and what each result would mean. Every test listed here is reported whatever it shows.

## Question

Did crime fall more within 500 m of a documented National Guard post, after the post began, than in comparable parts of DC without posts?

This tests troop presence at posts. It does not test the federal law enforcement surge that began the same day, which covered the whole city. Test 4 estimates the surge as a whole.

## Smallest effect that matters

A 15% reduction near posts. If the 95% interval for the effect excludes reductions of 15% or more, the result is reported as "rules out a reduction of 15% or more". If it includes both zero and 15%, it is reported as inconclusive. If it excludes zero on the side of a reduction, it is reported as a reduction, with its size.

## Posts and start dates

There is no official map. Posts come from WJLA (Aug 2025), a federal court opinion (Nov 20, 2025), DVIDS (Jan 30, 2026), Army.mil (Aug 3, 2026) and the Washingtonian (Aug 11, 2026).

| Wave | Posts | Start, primary | Start, sensitivity |
|---|---|---|---|
| 1 | Metro stations: Foggy Bottom-GWU, Smithsonian, Eastern Market, Stadium-Armory, Waterfront, McPherson Sq, L'Enfant Plaza, Gallery Pl-Chinatown, Metro Center, NoMa-Gallaudet U, Union Station, Navy Yard-Ballpark. The Mall: Washington Monument, Lincoln Memorial | Aug 11, 2025 | none |
| 2 | H Street NE corridor (at 4th, 8th and 12th St NE), Dupont Circle, 14th St NW (at P St and U St), Georgetown (M St at Wisconsin Ave) | Nov 20, 2025 (first documented) | Sep 1, 2025 |
| 3 | Historic Anacostia: Anacostia Metro, and Martin Luther King Jr Ave at Good Hope Rd SE | Jan 1, 2026 (Georgia Guard took over from Florida Guard in January) | Dec 1, 2025 |
| 4 | Ward 8 and Trinidad, Starburst Plaza (August 2026) | Exploratory only: under eight weeks of data after it | |

## Units and periods

- Units: 500 m square cells covering DC, keeping cells with at least one recorded incident or gunshot from 2015 on.
- Treated: cell center within 500 m of a post, from the wave's start date.
- Ring: cell center 500 m to 1,000 m from a post. Estimated separately, to measure crime pushed next door.
- Controls: cells more than 1,000 m from every post in every wave.
- Periods: 4-week periods aligned so that one begins Monday, August 11, 2025. A period that a start date splits is coded by the share of its days after the start.
- Window: two years before each wave's start through the end of the data. Gunshots run through June 30, 2026, and MPD incidents through September 30, 2026.

## Outcomes

Primary:
- Gunshots: ShotSpotter detections classed as single or multiple gunshots, in cells with at least one detection in the two years before wave 1.

Secondary, Holm-corrected as a family:
- Homicide, robbery and sex abuse together. Assault with a dangerous weapon is left out because its recording changed in 2025.
- Reported violent crime with a gun.
- Property crime.
- Other theft.
- Theft from cars.

## Tests

1. **Stacked event study.** For each wave, take its treated cells and the control cells, and stack the waves. Poisson regression of counts on treatment, with cell-by-stack and period-by-stack fixed effects. Errors are clustered by cell. Report:
   - the pooled effect and each wave's effect, as a rate ratio with its 95% interval;
   - an event-study plot of 4-week leads and lags;
   - a joint test that the leads are zero (were treated cells already diverging?);
   - the ring effect.
2. **Placebo posts (randomization inference).** Take the 12 Wave 1 Metro posts. Draw 12 stations at random, 1,000 times, from the DC Metro stations that had no Guard post. Re-estimate the Wave 1 Metro-post effect each time. Report where the real estimate falls among the placebos. The p-value is the share of placebo estimates at least as large a reduction as the real one.
3. **Metro.** If station-level Metro Transit Police data can be had, compare stations with and without posts the same way. If only systemwide data exists, report it descriptively and say so.
4. **Synthetic DC.** For murder, robbery and property crime (burglary, theft and car theft):
   - Index each agency's monthly series to its 2017-2024 average.
   - Fit DC's January 2017 to July 2025 series with non-negative weights summing to 1. The donors are Real-Time Crime Index agencies of 250,000 or more people with complete data, DC excluded.
   - The effect is the average gap, September 2025 to June 2026, as a percent of synthetic DC. August 2025 is left out as a partial month.
   - Inference: rerun with each donor as a fake DC, and rank DC's post/pre RMSPE ratio (how much worse the fit gets after August 2025, relative to before).
   - This measures the whole federal surge, not the Guard.
5. **Drawdown.** Troops fell from 5,148 (July 10, 2026) to 2,921 (September 29, 2026). Compare crime against the 10-year norm in the weeks after the first documented withdrawal with the eight weeks before, citywide and in Wave 1 to 3 cells. This test is descriptive, because which posts were dropped is not known. Gunshots are added when MPD publishes the third quarter of 2026.

## Robustness, all reported

- Radius of 250 m and 1,000 m instead of 500 m.
- Sensitivity start dates for waves 2 and 3.
- Dropping periods that contain July 4 or December 31 (fireworks).
- Dropping cells on the National Mall, where the Park Police record most crime.
- Negative binomial instead of Poisson.
- Controls restricted to cells in the top fifth of pre-period counts, so controls are as busy as treated cells.

## What would change the conclusion

- A reduction near posts that clears 15%, holds in the placebo test (real estimate in the most extreme 5%), and shows no pre-trend. That would be reported as evidence that posts reduced crime nearby.
- A reduction near posts matched by an increase in the ring. That would be reported as crime moved, not prevented.
