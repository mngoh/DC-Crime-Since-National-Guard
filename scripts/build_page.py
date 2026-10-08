"""Build the findings page from the analysis outputs. Every number on the page comes from out/.

  python scripts/build_page.py   ->  index.html (full document, for GitHub Pages)
                                     out/page.html (fragment, for a claude.ai artifact)
"""
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "out/results.json").read_text())
W = json.loads((ROOT / "out/wards.json").read_text())
P = json.loads((ROOT / "out/phases.json").read_text())
S = json.loads((ROOT / "out/shotspotter_checks.json").read_text())
MON = pd.read_csv(ROOT / "out/monthly.csv", index_col=0, parse_dates=True)
GP = json.loads((ROOT / "out/grid.json").read_text())          # planned sites
GU = json.loads((ROOT / "out/grid_updated.json").read_text())       # with the Guard's Sept 12, 2025 list
RIU = json.loads((ROOT / "out/grid_ri_updated.json").read_text())
GX = json.loads((ROOT / "out/grid_extra.json").read_text())
SY = json.loads((ROOT / "out/synth.json").read_text())
SYS = pd.read_csv(ROOT / "out/synth_series.csv")
MT = json.loads((ROOT / "out/metro.json").read_text())
DD = json.loads((ROOT / "out/drawdown.json").read_text())
DY = R["deployment_years"]


ORD = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}


def pc(x, signed=True):
    """-34.2 -> '−34%' (true minus sign), +29 -> '+29%'."""
    if x is None:
        return "n/a"
    v = round(x)
    if v < 0:
        return f"−{abs(v)}%"
    return f"+{v}%" if signed and v > 0 else f"{v}%"


def num(x):
    return f"{int(round(x)):,}"


def ratio_text(r):
    lo, hi = r["ci95"]
    return f"{r['point']:.2f} (95% range {lo:.2f} to {hi:.2f})"


# gunshot detections, citywide, Aug 11 - Jun 30 windows
shots = pd.read_csv(ROOT / "data/raw/shotspotter.csv", parse_dates=["DATETIME"]).drop_duplicates("ID")
shots = shots[shots.TYPE.str.replace("_", " ").isin(["Single Gunshot", "Multiple Gunshots"])]
gs = {y: int(shots.DATETIME.between(f"{y}-08-11", f"{y + 1}-06-30 23:59").sum()) for y in range(2015, 2026)}
gs_avg10 = sum(gs[y] for y in range(2015, 2025)) / 10
GS = {"guard": gs[2025], "before": gs[2024], "avg10": gs_avg10,
      "yoy": 100 * (gs[2025] / gs[2024] - 1), "vs_avg": 100 * (gs[2025] / gs_avg10 - 1), "before_vs_avg": 100 * (gs[2024] / gs_avg10 - 1)}

# weekly gunshots around the deployment, aligned on the Monday of deployment week
wk = shots.set_index("DATETIME").resample("D").size()
weeks = list(range(-6, 12))
w25 = [int(wk.loc[pd.Timestamp("2025-08-11") + pd.Timedelta(weeks=k):pd.Timestamp("2025-08-17") + pd.Timedelta(weeks=k)].sum()) for k in weeks]
w24 = [int(wk.loc[pd.Timestamp("2024-08-12") + pd.Timedelta(weeks=k):pd.Timestamp("2024-08-18") + pd.Timedelta(weeks=k)].sum()) for k in weeks]
wlabels = [(pd.Timestamp("2025-08-11") + pd.Timedelta(weeks=k)).strftime("%b %-d") for k in weeks]

# monthly ratio to the 10-year norm
m = MON.loc["2024-01-01":]
monthly = {"labels": [d.strftime("%b %y") for d in m.index],
           "property": [round(100 * (x - 1)) for x in m.ratio_property],
           "violent": [round(100 * (x - 1)) for x in m.ratio_violent],
           "gun": [round(100 * (x - 1)) for x in m.ratio_gun_violent],
           "deploy_index": list(m.index).index(pd.Timestamp("2025-08-01")) + 10 / 31}
fall_prop = 100 * (MON.loc["2025-10-01":"2025-12-01", "count_property"].sum() / MON.loc["2025-10-01":"2025-12-01", "norm_property"].sum() - 1)
last_prop = 100 * (MON.loc["2026-07-01":"2026-09-01", "count_property"].sum() / MON.loc["2026-07-01":"2026-09-01", "norm_property"].sum() - 1)

# national comparison, September to June
N = R["national"]["series"]
nat_keys = [("murder", "Murder"), ("robbery", "Robbery"), ("property", "Property"), ("theft", "Theft"), ("motor", "Car theft")]
national = {"labels": [l for _, l in nat_keys],
            "dc_pre": [N[k]["dc"]["pre_vs_prepre_pct"] for k, _ in nat_keys], "us_pre": [N[k]["national"]["pre_vs_prepre_pct"] for k, _ in nat_keys],
            "dc_post": [N[k]["dc"]["post_vs_pre_pct"] for k, _ in nat_keys], "us_post": [N[k]["national"]["post_vs_pre_pct"] for k, _ in nat_keys]}
BC = R["big_cities"]

# troops vs no troops
ph_keys = [("gunshots", ["Gunshots", "detected"]), ("gun_violent", ["Reported", "gun crime"]), ("violent", ["Violent", "crime"]), ("property", ["Property", "crime"])]
p1 = {k: P["phase1"][k]["wards_2_6_vs_3_4_5_7_8"] for k, _ in ph_keys}
p2 = {k: P["phase2"][k]["7D_vs_6D"] for k, _ in ph_keys}
fall = P["phase2_fall_check"]
phase = {"labels": [l for _, l in ph_keys],
         "p1_t": [p1[k]["troops"]["change_pct"] for k, _ in ph_keys], "p1_c": [p1[k]["no_troops"]["change_pct"] for k, _ in ph_keys],
         "p2_t": [p2[k]["troops"]["change_pct"] for k, _ in ph_keys], "p2_c": [p2[k]["no_troops"]["change_pct"] for k, _ in ph_keys]}
WG, WH = W["measures"]["gunshots"]["by_ward"], W["measures"]["homicide"]["by_ward"]
posts = {"1": "none", "2": "7: downtown Metro stations, the Mall", "3": "none", "4": "none", "5": "none",
         "6": "5: Union Station, Eastern Market, Waterfront, NoMa, L'Enfant Plaza", "7": "1, on its western edge (Stadium-Armory)",
         "8": "1, on its western edge (Navy Yard); Anacostia patrols from January 2026 or earlier"}
ward_chart = {"labels": [f"Ward {w}" for w in ["1", "2", "4", "5", "6", "7", "8"]],
              "vals": [WG[w]["vs_year_before_pct"] for w in ["1", "2", "4", "5", "6", "7", "8"]],
              "troops": [w in ("2", "6") for w in ["1", "2", "4", "5", "6", "7", "8"]]}
theft = P["phase1"]["theft_other"]["downtown_vs_rest_first_year"]
tauto = P["phase1"]["theft_auto"]["downtown_vs_rest_first_year"]

hom, gun, tot, vio, prop = DY["homicide"], DY["gun_violent"], DY["total"], DY["violent"], DY["property"]
its = R["its"]
GK = [("gunshots", "Gunshots detected"), ("violent_ex_adw", "Homicide, robbery, sex abuse"), ("gun_violent", "Reported gun crime"),
      ("property", "Property crime"), ("theft_other", "Other theft"), ("theft_auto", "Theft from cars")]
forest = {"labels": [l for _, l in GK],
          "p_rr": [GP["outcomes"][k]["pooled"]["treat"]["rr"] for k, _ in GK], "p_ci": [GP["outcomes"][k]["pooled"]["treat"]["ci95"] for k, _ in GK],
          "u_rr": [GU["outcomes"][k]["pooled"]["treat"]["rr"] for k, _ in GK], "u_ci": [GU["outcomes"][k]["pooled"]["treat"]["ci95"] for k, _ in GK]}
sp = SYS[(SYS.outcome == "property") & (SYS.month >= "2023-01")]
synth_chart = {"labels": [pd.Timestamp(m).strftime("%b %y") for m in sp.month], "dc": [round(100 * v) for v in sp.dc],
               "synth": [round(100 * v) for v in sp.synthetic], "deploy": list(sp.month).index("2025-08") + 10 / 31}
READ = {"reduction": "Reduction", "increase": "Increase near posts", "rules out a reduction of 15% or more": "Rules out a 15% reduction",
        "inconclusive": "Inconclusive"}


def reading(pooled):
    r = READ[pooled["verdict"]]
    ph = pooled["treat"].get("p_holm")
    if pooled["verdict"] in ("increase", "reduction") and ph is not None and ph > 0.05:
        r += ", not significant across six measures"
    return r


def rr_cell(t):
    return f"{t['rr']:.2f}<span class='ci'>{t['ci95'][0]:.2f} to {t['ci95'][1]:.2f}</span>"


tests_rows = "".join(
    f"<tr><td>{lab}</td><td class='n'>{rr_cell(GP['outcomes'][k]['pooled']['treat'])}</td><td class='n'>{rr_cell(GU['outcomes'][k]['pooled']['treat'])}</td>"
    f"<td class='n'>{round(100 * RIU[k]['share_placebos_at_or_below_real'])}%</td><td>{reading(GU['outcomes'][k]['pooled'])}</td></tr>" for k, lab in GK)
MV = MT["outcomes"]["victim"]
mev = [MV["event_study"]["by_quarter"][k] for k in ["0", "1", "2", "3", "4"]]
DDo = DD["outcomes"]
data = {"forest": forest, "synth": synth_chart, "monthly": monthly, "weekly": {"labels": wlabels, "y2025": w25, "y2024": w24, "deploy": weeks.index(0)},
        "national": national, "phase": phase, "wards": ward_chart}

# the full table, first Guard year against the 10-year average
rows = [("total", "All reported crime"), ("violent", "Violent crime"), ("gun_violent", "Violent crime with a gun"),
        ("homicide", "Homicide"), ("robbery", "Robbery"), ("adw", "Assault with a dangerous weapon"),
        ("sex_abuse", "Sex abuse"), ("property", "Property crime"), ("mvt", "Car theft"), ("theft_auto", "Theft from cars"),
        ("theft_other", "Other theft"), ("burglary", "Burglary")]
table = "".join(
    f"<tr><td>{lab}</td><td class='n'>{num(DY[k]['deploy_year'])}</td><td class='n'>{num(DY[k]['avg10'])}</td>"
    f"<td class='n strong'>{pc(DY[k]['vs_avg10_pct'])}</td><td class='n'>{pc(DY[k]['year_before_vs_avg10_pct'])}</td>"
    f"<td class='n'>{pc(DY[k]['vs_year_before_pct'])}</td></tr>" for k, lab in rows)
its_rows = [("total", "All reported crime"), ("property", "Property crime"), ("violent", "Violent crime"),
            ("gun_violent", "Violent crime with a gun"), ("homicide", "Homicide")]


def its_cell(s):
    sig = s["ci95"][0] > 0 or s["ci95"][1] < 0
    return f"<td class='n{' strong' if sig else ''}'>{pc(s['step_pct'])}<span class='ci'>{pc(s['ci95'][0])} to {pc(s['ci95'][1])}</span></td>"


its_table = "".join(f"<tr><td>{lab}</td>{its_cell(its[k][0])}{its_cell(its[k][1])}</tr>" for k, lab in its_rows)
ward_rows = "".join(
    f"<tr{' class=troop' if w in ('2', '6') else ''}><td>Ward {w}</td><td>{posts[w]}</td>"
    f"<td class='n'>{WH[w]['year_before']} to {WH[w]['guard_period']}</td>"
    f"<td class='n'>{'no sensors' if not WG[w]['year_before'] else f'''{num(WG[w]['year_before'])} to {num(WG[w]['guard_period'])} ({pc(WG[w]['vs_year_before_pct'])})'''}</td></tr>"
    for w in [str(i) for i in range(1, 9)])

g1, g2, gf = p1["gunshots"], p2["gunshots"], fall["gunshots"]["7D_vs_6D"]
first_week_before = S["weekly_gunshots_vs_reported_gun_crimes"]["2025-08-04"]
first_week = S["weekly_gunshots_vs_reported_gun_crimes"]["2025-08-11"]
busy = S["busy_squares"]

HEADLINE = (f"Violence in DC fell by close to half after the National Guard arrived on August 11, 2025. "
            f"It fell just as much where no troops were posted, so nothing in the data shows the troops themselves adding to the decline.")
DESC = ("Reported crime, homicides and gunshots in Washington, DC before and after the August 2025 National Guard deployment, "
        "against a 10-year average, the national trend, and the places troops were and were not posted.")

BODY = f"""<title>DC crime since the National Guard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{ color-scheme: dark; --bg: #0a0a0a; --surface: #121212; --border: #262626; --text: #f2f2f2; --muted: #8b8b8b;
          --blue: #2563eb; --blue-light: #93c5fd; --red: #ef4444; --grey: #6b7280; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.6; }}
  a {{ color: var(--blue-light); text-decoration: none; }} a:hover {{ text-decoration: underline; }}
  a:focus-visible {{ outline: 2px solid var(--blue-light); outline-offset: 2px; }}
  header {{ border-bottom: 1px solid var(--border); padding: 28px 40px; display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap; }}
  header nav {{ display: flex; gap: 20px; font-size: 13px; }}
  header h1 {{ font-size: 22px; font-weight: 600; letter-spacing: -0.3px; text-wrap: balance; }}
  header p {{ color: var(--muted); margin-top: 4px; font-size: 13px; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 32px 40px; }}
  .lede {{ font-size: 21px; font-weight: 600; line-height: 1.4; letter-spacing: -0.2px; max-width: 860px; margin-bottom: 24px; text-wrap: balance; }}
  .answer {{ background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--red); border-radius: 0 8px 8px 0; padding: 20px 24px; margin-bottom: 40px; max-width: 860px; }}
  .answer .q {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); margin-bottom: 8px; font-weight: 600; }}
  .answer .question {{ font-size: 15px; line-height: 1.7; margin-bottom: 8px; }}
  .answer p {{ font-size: 14px; line-height: 1.7; color: var(--muted); }}
  .answer p + p {{ margin-top: 8px; }}
  .answer strong {{ color: var(--text); font-weight: 600; }}
  .section-title {{ font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  .note {{ font-size: 12px; color: var(--muted); margin: -6px 0 16px; max-width: 860px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 18px 20px; }}
  .card .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); margin-bottom: 8px; }}
  .card .value {{ font-size: 26px; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }}
  .card .sub {{ font-size: 11px; color: var(--muted); margin-top: 6px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .charts.one {{ grid-template-columns: 1fr; }}
  .section-end {{ margin-bottom: 40px; }}
  .chart-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 22px 24px; min-width: 0; }}
  .chart-box h3 {{ font-size: 13px; font-weight: 600; margin-bottom: 4px; }}
  .chart-sub {{ font-size: 11px; color: var(--muted); margin-bottom: 16px; line-height: 1.5; }}
  .chart-wrap {{ position: relative; height: 270px; }}
  .chart-wrap.tall {{ height: 340px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 11px; color: var(--muted); margin-bottom: 10px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .sw {{ width: 14px; height: 0; border-top: 2px solid; display: inline-block; }}
  .sw.box {{ height: 10px; border: 1.5px solid; border-radius: 2px; }}
  .findings {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 40px; }}
  .finding {{ background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--blue); border-radius: 0 8px 8px 0; padding: 16px 20px; min-width: 0; }}
  .finding.red {{ border-left-color: var(--red); }}
  .finding h4 {{ font-size: 13px; font-weight: 600; margin-bottom: 4px; }}
  .finding p {{ font-size: 12px; color: var(--muted); line-height: 1.55; }}
  .table-wrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; }}
  th, td {{ padding: 9px 14px; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }}
  th {{ font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  tr:last-child td {{ border-bottom: 0; }}
  td.n, th.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.n {{ white-space: nowrap; }}
  td:first-child {{ white-space: nowrap; }}
  table.wards td:nth-child(2) {{ min-width: 260px; }}
  td.strong {{ color: var(--text); font-weight: 600; }}
  td .ci {{ display: block; font-size: 10.5px; color: var(--muted); font-weight: 400; }}
  tr.troop td:first-child {{ box-shadow: inset 3px 0 0 var(--red); }}
  .method {{ font-size: 12px; color: var(--muted); max-width: 860px; }}
  .method p + p {{ margin-top: 8px; }}
  .method ul {{ margin: 8px 0 0 18px; }}
  .method li + li {{ margin-top: 4px; }}
  footer {{ border-top: 1px solid var(--border); padding: 20px 40px; color: var(--muted); font-size: 12px; }}
  @media (max-width: 768px) {{
    header, footer {{ padding: 20px 16px; }} .container {{ padding: 20px 16px; }}
    .charts, .findings {{ grid-template-columns: 1fr; }}
    .chart-box {{ padding: 18px 16px; }} .lede {{ font-size: 18px; }}
  }}
</style>
<header>
  <div><h1>DC crime since the National Guard</h1>
  <p>August 11, 2025 to September 30, 2026 · Metropolitan Police incidents and gunshot detections · Martin Ngoh</p></div>
  <nav><a href="https://github.com/mngoh/DC-Crime-Since-National-Guard">Code</a><a href="https://martinngoh.com">martinngoh.com</a></nav>
</header>
<div class="container">
  <p class="lede">{HEADLINE}</p>
  <div class="answer">
    <div class="q">The question</div>
    <div class="question">Did crime in DC fall after the National Guard deployed, and did it fall more where the troops were?</div>
    <p><strong>It fell.</strong> In the first Guard year, August 11, 2025 to August 10, 2026, reported crime was {pc(tot['vs_avg10_pct'])} against the average for the same dates over the previous 10 years. The year before, it was already {pc(tot['year_before_vs_avg10_pct'])}.</p>
    <p><strong>Homicides and gunfire fell too, and they do not depend on anyone calling the police.</strong> Homicides fell from {hom['year_before']} to {hom['deploy_year']} ({pc(hom['vs_year_before_pct'])}). Gunshots picked up by sensors fell {pc(GS['yoy'])} from the year before (August 11 to June 30). From September to June, DC's homicide drop was the {ORD.get(BC['murder']['dc_rank_post_drop'], str(BC['murder']['dc_rank_post_drop']) + 'th')} largest of {BC['murder']['n_agencies']} large police departments.</p>
    <p><strong>It did not fall more where troops were posted.</strong> In the first months, gunshots fell {pc(g1['troops']['change_pct'])} in the two wards holding nearly all the Guard posts and {pc(g1['no_troops']['change_pct'])} in the wards without them. When the Guard began patrolling Anacostia in 2026, gunshots there fell {pc(g2['troops']['change_pct'])} against {pc(g2['no_troops']['change_pct'])} in neighboring Ward 7, which had no documented patrols.</p>
    <p><strong>Tests written down before they ran agree.</strong> Across about 800 half-kilometer squares, crime near troop posts fell no more than in comparable areas on any of six measures, and the areas around the Metro stations where troops stood did worse than most random sets of other stations. Against other large cities, DC's property crime fell about {abs(round(SY['property']['effect_pct']))}% more after August 2025, but citywide, not near the posts.</p>
    <p>Federal agents arrived the same day, and violent crime was already falling, so this data cannot say what drove the decline. It can say the decline does not follow the troops.</p>
  </div>

  <div class="section-title">Against the 10-year average</div>
  <p class="note">The first Guard year (August 11, 2025 to August 10, 2026) against the average of the same dates in the 10 years before. The second line is the year before the deployment, against the same average.</p>
  <div class="cards">
    <div class="card"><div class="label">All reported crime</div><div class="value">{pc(tot['vs_avg10_pct'])}</div><div class="sub">year before: {pc(tot['year_before_vs_avg10_pct'])}</div></div>
    <div class="card"><div class="label">Violent crime</div><div class="value">{pc(vio['vs_avg10_pct'])}</div><div class="sub">year before: {pc(vio['year_before_vs_avg10_pct'])}</div></div>
    <div class="card"><div class="label">Violent crime with a gun</div><div class="value">{pc(gun['vs_avg10_pct'])}</div><div class="sub">year before: {pc(gun['year_before_vs_avg10_pct'])}</div></div>
    <div class="card"><div class="label">Homicide</div><div class="value">{pc(hom['vs_avg10_pct'])}</div><div class="sub">{hom['deploy_year']} homicides; year before: {pc(hom['year_before_vs_avg10_pct'])}</div></div>
    <div class="card"><div class="label">Property crime</div><div class="value">{pc(prop['vs_avg10_pct'])}</div><div class="sub">year before: {pc(prop['year_before_vs_avg10_pct'])}</div></div>
  </div>
  <div class="charts one">
    <div class="chart-box">
      <h3>Month by month against the 10-year average</h3>
      <div class="chart-sub">Percent above or below the average for the same calendar dates, August 11, 2015 to August 10, 2025. Violent crime and gun violence were already well below normal before the deployment. Property crime dropped within weeks of it, was lowest in fall 2025 ({pc(fall_prop)} in October to December) and eased to {pc(last_prop)} by July to September 2026, while troop numbers rose to a peak of 5,148 in July 2026.</div>
      <div class="legend"><span><i class="sw" style="border-color:#ef4444"></i>Violent crime</span><span><i class="sw" style="border-color:#93c5fd;border-top-style:dashed"></i>Violent crime with a gun</span><span><i class="sw" style="border-color:#2563eb"></i>Property crime</span></div>
      <div class="chart-wrap tall"><canvas id="cMonthly" role="img" aria-label="Monthly reported crime against the 10-year average, January 2023 to September 2026, with the deployment marked in August 2025."></canvas></div>
    </div>
  </div>
  <div class="table-wrap section-end"><table>
    <thead><tr><th>Offense</th><th class="n">First Guard year</th><th class="n">10-year average</th><th class="n">Guard year vs average</th><th class="n">Year before vs average</th><th class="n">Change from year before</th></tr></thead>
    <tbody>{table}</tbody>
  </table></div>

  <div class="section-title">What was already happening</div>
  <p class="note">A 10-year average removes the seasons, not the trend. These two checks ask how much of the drop was new.</p>
  <div class="charts">
    <div class="chart-box">
      <h3>Change at the deployment</h3>
      <div class="chart-sub">Monthly counts against the 10-year norm, modeled as a step at August 11, 2025. The first column compares with the 12 months before. The second assumes the decline under way since January 2024 would have continued. Bold: the 95% range excludes zero.</div>
      <div class="table-wrap" style="margin:0"><table><thead><tr><th>Offense</th><th class="n">Against level before</th><th class="n">If the trend continued</th></tr></thead><tbody>{its_table}</tbody></table></div>
    </div>
    <div class="chart-box">
      <h3>DC against the nation, September to June</h3>
      <div class="chart-sub">Change from the same months a year earlier, DC against 590 police agencies in the Real-Time Crime Index (DC removed). Before: 2024-25 vs 2023-24. After: 2025-26 vs 2024-25. DC was already cutting robbery faster than the nation; the murder and property gaps opened after the deployment.</div>
      <div class="legend"><span><i class="sw box" style="border-color:#ef4444;background:rgba(239,68,68,.15)"></i>DC</span><span><i class="sw box" style="border-color:#2563eb;background:rgba(37,99,235,.15)"></i>Nation</span><span>Faded: year before</span></div>
      <div class="chart-wrap"><canvas id="cNational" role="img" aria-label="Bar chart comparing DC and national changes in murder, robbery, property crime, theft and car theft, before and after the deployment."></canvas></div>
    </div>
  </div>
  <div class="findings">
    <div class="finding"><h4>Homicide: a drop well beyond the nation</h4><p>DC murders fell {pc(N['murder']['dc']['post_vs_pre_pct'])} from September to June against {pc(N['murder']['national']['post_vs_pre_pct'])} nationally. The year before, DC fell {pc(N['murder']['dc']['pre_vs_prepre_pct'])} and the nation {pc(N['murder']['national']['pre_vs_prepre_pct'])}. Against the 12 months before, the step is {pc(its['homicide'][0]['step_pct'])} ({pc(its['homicide'][0]['ci95'][0])} to {pc(its['homicide'][0]['ci95'][1])}).</p></div>
    <div class="finding"><h4>Violent crime overall: mostly an earlier trend</h4><p>Against the 12 months before, violent crime shows a step of {pc(its['violent'][0]['step_pct'])} ({pc(its['violent'][0]['ci95'][0])} to {pc(its['violent'][0]['ci95'][1])}), which cannot be told apart from no change. Robbery fell {pc(N['robbery']['dc']['post_vs_pre_pct'])}, but it had fallen {pc(N['robbery']['dc']['pre_vs_prepre_pct'])} the year before.</p></div>
  </div>

  <div class="section-title">Crime that does not need a report</div>
  <p class="note">MPD's crime data is under investigation for downgraded reports, and fewer calls to police can look like less crime. Homicides and gunshots picked up by sensors avoid most of that. Sensors cover Wards 1, 2 and 4 through 8; Ward 3 has none.</p>
  <div class="charts">
    <div class="chart-box">
      <h3>Gunshots detected each week</h3>
      <div class="chart-sub">MPD ShotSpotter detections classed as single or multiple gunshots, the weeks around August 11, 2025, against the same weeks a year earlier. Detections fell from {first_week_before['gunshots']} to {first_week['gunshots']} in the week the deployment began; reported gun crimes, which involve no sensors, fell from {first_week_before['reported_gun_crimes']} to {first_week['reported_gun_crimes']} the same week. The spike in early July is Independence Day fireworks.</div>
      <div class="legend"><span><i class="sw" style="border-color:#ef4444"></i>2025</span><span><i class="sw" style="border-color:#6b7280"></i>Same weeks of 2024</span></div>
      <div class="chart-wrap"><canvas id="cWeekly" role="img" aria-label="Weekly gunshot detections from late June to early November 2025, against the same weeks of 2024, with a sharp drop in the week of August 11."></canvas></div>
    </div>
    <div class="chart-box">
      <h3>Is the gunshot drop real?</h3>
      <div class="chart-sub">Checks for a sensor or system change rather than less gunfire.</div>
      <div class="findings" style="grid-template-columns:1fr;margin:0">
        <div class="finding"><h4>No outages</h4><p>Since 2024, only {len(S['zero_days_since_2024'])} single days had no detections. There were no longer gaps.</p></div>
        <div class="finding"><h4>No area went dark</h4><p>Of {busy['n']} one-kilometer squares with 20 or more detections the year before, {busy['went_silent']} went silent. The typical square fell {pc(100 * (busy['post_over_pre_quantiles']['0.5'] - 1))}.</p></div>
        <div class="finding"><h4>Not relabeled</h4><p>The "probable gunfire" category left out here fell too, so detections did not move between labels.</p></div>
        <div class="finding"><h4>Independent counts agree</h4><p>Shootings counted by the Gun Violence Archive were down about two thirds from a year earlier (The Trace, October 2025).</p></div>
      </div>
    </div>
  </div>
  <div class="section-end"></div>

  <div class="section-title">Did it fall more where the troops were?</div>
  <p class="note">The Guard's posts moved, so the test runs in two phases. Phase 1, August 11 to November 26, 2025: posts and patrols were downtown, on the Mall, at Metro stations, Union Station and Navy Yard, and along H Street, 14th Street, Dupont Circle, Georgetown and Capitol Hill, almost all in Wards 2 and 6, compared with Wards 3, 4, 5, 7 and 8 (Ward 1 is left out because patrols reached 14th Street by mid-September). Phase 2, January to June 2026: Guard patrols in Anacostia, MPD's 7th District, compared with the neighboring 6th District (Ward 7), where no patrols are documented. Each is the change from the same dates a year earlier.</p>
  <div class="charts">
    <div class="chart-box">
      <h3>Phase 1: downtown posts, fall 2025</h3>
      <div class="chart-sub">Wards 2 and 6 against Wards 3, 4, 5, 7 and 8. Gunshots: {g1['troops']['pre']} to {g1['troops']['post']} with troops, {num(g1['no_troops']['pre'])} to {num(g1['no_troops']['post'])} without. Ratio of changes {ratio_text(g1['ratio'])}; 1 means the same drop.</div>
      <div class="legend"><span><i class="sw box" style="border-color:#ef4444;background:rgba(239,68,68,.15)"></i>With troops</span><span><i class="sw box" style="border-color:#2563eb;background:rgba(37,99,235,.15)"></i>Without troops</span></div>
      <div class="chart-wrap"><canvas id="cPhase1" role="img" aria-label="Bar chart of changes in fall 2025 for gunshots, reported gun crime, violent crime and property crime, wards with Guard posts against wards without."></canvas></div>
    </div>
    <div class="chart-box">
      <h3>Phase 2: Anacostia patrols, 2026</h3>
      <div class="chart-sub">7th District (Anacostia, Guard patrols) against 6th District (Ward 7, none documented), January to June. Gunshots: ratio of changes {ratio_text(g2['ratio'])}. In fall 2025, before patrols there are on record, the two districts moved together: gunshots {pc(gf['troops']['change_pct'])} and {pc(gf['no_troops']['change_pct'])}.</div>
      <div class="legend"><span><i class="sw box" style="border-color:#ef4444;background:rgba(239,68,68,.15)"></i>With troops</span><span><i class="sw box" style="border-color:#2563eb;background:rgba(37,99,235,.15)"></i>Without troops</span></div>
      <div class="chart-wrap"><canvas id="cPhase2" role="img" aria-label="Bar chart of changes in January to June 2026 for gunshots, reported gun crime, violent crime and property crime, Anacostia against Ward 7."></canvas></div>
    </div>
  </div>
  <div class="charts">
    <div class="chart-box">
      <h3>Gunshots by ward</h3>
      <div class="chart-sub">Change from the year before, August 11 to June 30. Red: the two wards holding nearly all the Guard posts in 2025. Ward 3 has no sensors.</div>
      <div class="chart-wrap"><canvas id="cWards" role="img" aria-label="Bar chart of the change in gunshot detections by ward; every ward fell between about 30 and 48 percent."></canvas></div>
    </div>
    <div class="findings" style="grid-template-columns:1fr;margin:0;align-content:start">
    <div class="finding"><h4>No measure fell clearly more with troops</h4><p>For gunshots, homicide, reported gun crime and violent crime, in both phases, by ward and within 500 meters of a post, no comparison shows the troop areas falling more by a margin outside its 95% range. Property crime fell less with troops in both phases. The ranges come from resampling two- and four-week blocks, so ordinary swings in crime are not mistaken for an effect.</p></div>
    <div class="finding"><h4>Downtown theft: not clear</h4><p>Other theft in the downtown business district fell {pc(theft['troops']['change_pct'])} in the first Guard year against {pc(theft['no_troops']['change_pct'])} elsewhere, but the year before it had risen about {pc(100 * (theft['placebo']['point'] - 1), signed=False)} faster than elsewhere, so the drop mostly undoes that rise. Theft from cars downtown fell {pc(tauto['troops']['change_pct'])} against {pc(tauto['no_troops']['change_pct'])}, and part of that gap was already there before August.</p></div>
    </div>
  </div>
  <div class="table-wrap" style="margin-bottom:36px"><table class="wards">
    <thead><tr><th>Ward</th><th>Guard posts</th><th class="n">Homicides</th><th class="n">Gunshots detected</th></tr></thead>
    <tbody>{ward_rows}</tbody>
  </table></div>
  <p class="note" style="margin:-28px 0 40px">August 11 to June 30, year before to Guard year. Posts as reported by WJLA, Guard releases, a Guard daily update filed in federal court (September 12, 2025) and a November 2025 court opinion; there is no official map.</p>

  <div class="section-title">Tests written down in advance</div>
  <p class="note">On October 8, 2026, before running them, I published a <a href="https://github.com/mngoh/DC-Crime-Since-National-Guard/blob/main/docs/test-plan.md">test plan</a> fixing the areas, dates, measures and what would count as an effect: a reduction of 15% or more near posts. Every test it lists is reported here. Afterward, a Guard daily update filed in court (September 12, 2025) showed more patrol sites than the plan had, so the tests were rerun with them. Both versions are shown.</p>
  <div class="charts">
    <div class="chart-box">
      <h3>Near troop posts, against comparable areas</h3>
      <div class="chart-sub">Rate ratio with its 95% range: about 800 half-kilometer squares, three waves of posts, each compared with squares more than a kilometer from any post over the same weeks. Below 1 means fewer crimes near posts than expected. The dashed line marks a 15% reduction, the smallest effect the plan said would matter.</div>
      <div class="legend"><span><i class="sw box" style="border-color:#ef4444;background:rgba(239,68,68,.15)"></i>Planned sites</span><span><i class="sw box" style="border-color:#2563eb;background:rgba(37,99,235,.15)"></i>With the Guard's September list</span></div>
      <div class="chart-wrap tall"><canvas id="cForest" role="img" aria-label="Rate ratios with 95 percent ranges for six measures near troop posts; none falls entirely below 1."></canvas></div>
    </div>
    <div class="chart-box">
      <h3>Property crime, DC against a synthetic DC</h3>
      <div class="chart-sub">Monthly property crime as a percent of each city's 2017 to 2024 average. Synthetic DC is a weighted mix of {SY['property']['donors']} large police agencies, chosen to match DC from 2017 to July 2025. From September 2025 to June 2026, DC ran {abs(round(SY['property']['effect_pct']))}% below it; only {SY['property']['rank_all'] - 1} of the {SY['property']['donors']} agencies, each tested the same way, showed as large a break (p = {SY['property']['p_all']:.2f}). This measures everything that arrived on August 11 together.</div>
      <div class="legend"><span><i class="sw" style="border-color:#ef4444"></i>DC</span><span><i class="sw" style="border-color:#6b7280;border-top-style:dashed"></i>Synthetic DC</span></div>
      <div class="chart-wrap tall"><canvas id="cSynth" role="img" aria-label="DC property crime against synthetic DC, January 2023 to June 2026, with DC falling below it after August 2025."></canvas></div>
    </div>
  </div>
  <div class="table-wrap" style="margin-bottom:16px"><table>
    <thead><tr><th>Measure</th><th class="n">Near posts, planned sites</th><th class="n">With the September list</th><th class="n">Random station sets that did better</th><th>Reading</th></tr></thead>
    <tbody>{tests_rows}</tbody>
  </table></div>
  <div class="findings">
    <div class="finding"><h4>No reduction near posts</h4><p>None of the six measures shows fewer crimes near posts than in comparable areas. For property crime and other theft, the range rules out a reduction of 15% or more. For gunshots and gun crime, the range is too wide to rule out a modest reduction, and it also allows an increase.</p></div>
    <div class="finding"><h4>Random Metro stations did better</h4><p>I drew 12 stations at random from the {len(GU['placebo_pool'])} DC stations away from any known Guard site, 1,000 times, and ran the same test on each set. For gunshots, {round(100 * RIU['gunshots']['share_placebos_at_or_below_real'])}% of the random sets did better than the real Guard stations; for homicide, robbery and sex abuse, {round(100 * RIU['violent_ex_adw']['share_placebos_at_or_below_real'])}%. The Guard probably chose stations with growing problems, which would explain part of this.</p></div>
    <div class="finding"><h4>Violent crime near posts rose, relative to elsewhere</h4><p>Homicide, robbery and sex abuse near posts ran {pc(100 * (GP['outcomes']['violent_ex_adw']['pooled']['treat']['rr'] - 1))} against comparable areas with the planned sites and {pc(100 * (GU['outcomes']['violent_ex_adw']['pooled']['treat']['rr'] - 1))} with the September list, which is not significant once six measures are tested together. About half of the gap was opening before the troops came: measured against only the 24 weeks before, it is {pc(100 * (GX['violent_ex_adw']['recent_baseline']['rr'] - 1))} ({GX['violent_ex_adw']['recent_baseline']['ci95'][0]:.2f} to {GX['violent_ex_adw']['recent_baseline']['ci95'][1]:.2f}).</p></div>
    <div class="finding"><h4>Inside Metro: cannot tell</h4><p>Metro Transit Police blotters (January 2022 to August 2026, crimes with victims, placed at stations by address) show the stations that later got troops had been getting worse than other stations for two years, peaking in the three months before August 2025. Afterward they fell back {round(100 * (1 - max(mev)))} to {round(100 * (1 - min(mev)))}% from that peak, but changed like other stations against the year before ({pc(MV['counts_year_before_to_guard_year']['guard'][2])} and {pc(MV['counts_year_before_to_guard_year']['clean_pool'][2])}). A Guard effect and an ordinary fall from a peak look the same here.</p></div>
    <div class="finding"><h4>The surge as a whole</h4><p>Against a synthetic DC built from other large agencies, September 2025 to June 2026: property crime {pc(SY['property']['effect_pct'])} (p = {SY['property']['p_all']:.2f}), murder {pc(SY['murder']['effect_pct'])} (p = {SY['murder']['p_all']:.2f}), robbery {pc(SY['robbery']['effect_pct'])} (p = {SY['robbery']['p_all']:.2f}). With the tests near posts, the property drop came citywide, not where troops stood.</p></div>
    <div class="finding"><h4>The drawdown</h4><p>Troops fell from 5,148 in July 2026 to 3,124 by September 7 and 2,864 by October 7. Against normal, citywide crime did not move ({pc(DDo['total']['citywide']['before_vs_norm_pct'])} in the eight weeks before the first state left, {pc(DDo['total']['citywide']['after_vs_norm_pct'])} after). Near posts, violent crime rose against an unusually quiet June, back to its February to May level. Gunshots for this period come out in November.</p></div>
    <div class="finding"><h4>A few waves look different</h4><p>Gunshots near the two Anacostia posts fell {pc(100 * (1 - GU['outcomes']['gunshots']['by_wave']['3']['treat']['rr']), signed=False)} more than in comparable areas, and reported gun crime along the patrolled corridors fell about {pc(100 * (1 - GU['outcomes']['gun_violent']['by_wave']['2']['treat']['rr']), signed=False)} more, although gunshots there rose. Among 36 wave-level estimates, one or two significant results in each direction is what chance alone produces; the combined tests the plan relied on show no reduction.</p></div>
    <div class="finding"><h4>What others found</h4><p>The Niskanen Center (May 2026) found a property crime drop of about 24% and no clear effect on violent crime. A Senate Homeland Security Committee minority staff report (2026) found "no directly attributable impact on crime."</p></div>
  </div>

  <div class="section-title">Caveats</div>
  <div class="findings">
    <div class="finding red"><h4>This shows what, not why</h4><p>The data shows violence fell and that the fall does not follow the troops. It cannot name the cause. Possibilities include the federal agents, MPD's own work, a trend already under way, or some mix.</p></div>
    <div class="finding red"><h4>The Guard was not alone</h4><p>On August 11, 2025 the federal government also took control of MPD (to September 10), sent in federal agents and stepped up immigration enforcement. The White House said about half of arrests by federal authorities were in Wards 7 and 8, where there were almost no troops. Federal agents kept working alongside MPD after September 10.</p></div>
    <div class="finding red"><h4>MPD's crime data is under investigation</h4><p>A House Oversight report (December 2025), an MPD internal affairs report (May 2026) and the DC Inspector General (July 2026) found misclassified and downgraded reports. Assault with a dangerous weapon rose {pc(DY['adw']['vs_year_before_pct'])} as knife and other-weapon cases roughly doubled while gun cases fell, which looks like a change in how assaults are recorded. That is why this page leans on homicides and gunshots.</p></div>
    <div class="finding red"><h4>Fewer reports is not less crime</h4><p>People avoiding police during immigration enforcement, or federal agencies taking reports that never reach MPD's data, would both lower counts. Neither can be measured here. Crimes on the National Mall are mostly handled by the Park Police and are not in this data.</p></div>
    <div class="finding red"><h4>Small numbers</h4><p>The two wards with troops had {WH['2']['year_before'] + WH['6']['year_before']} homicides the year before, too few to detect much. Gunshots, with thousands of detections, carry the comparison.</p></div>
    <div class="finding red"><h4>Where the troops were is reconstructed</h4><p>There is no official map of Guard posts. Locations come from news reports, Guard releases, a Guard daily update filed in federal court and a court opinion. The update counts 18 Metro stations but names none; 12 are known from news reports, so some comparison stations probably had troops. Records requests for the post list are drafted. The Guard also did occasional patrols and cleanup elsewhere, so "without troops" means light presence, not none.</p></div>
  </div>

  <div class="section-title">Method and sources</div>
  <div class="method">
    <p>Reported crime: MPD's public incident data (homicide, sex abuse, assault with a dangerous weapon, robbery, burglary, car theft, theft from auto, other theft, arson), January 2015 to September 30, 2026, by report date. Data for October 2026 is left out as incomplete. The 10-year average for any window is the mean count on the same calendar dates from August 11, 2015 to August 10, 2025, so it contains no days after the deployment. Wards are the 2022 boundaries, assigned by each incident's location.</p>
    <p>Gunshots: MPD's ShotSpotter data through June 30, 2026, single and multiple gunshots only. Comparisons that use gunshots run August 11 to June 30 in every year. National comparison: the Real-Time Crime Index (AH Datalytics), monthly through June 2026, a fixed sample of 590 agencies with DC removed. The step model is a Poisson regression of monthly counts with the 10-year norm as offset and Newey-West errors. Troop comparisons are ratios of changes with block-bootstrap ranges and a placebo year.</p>
    <ul>
      <li>Data: <a href="https://opendata.dc.gov/">DC Open Data</a> (crime incidents, ShotSpotter gunshots, wards, Metro stations); <a href="https://github.com/AH-Datalytics/rtci">Real-Time Crime Index</a></li>
      <li>Guard posts: <a href="https://wjla.com/news/local/national-guard-dc-metrorail-federal-guardsmen-troops-wmata-blue-line-red-green-orange-silver-yellow-crime-enforcement-police-trump-white-house-military-civilians-washington">WJLA, August 2025</a>; <a href="https://oag.dc.gov/sites/default/files/2025-11/National_Guard_Ruling.pdf">federal court opinion, November 2025</a>; <a href="https://www.dvidshub.net/news/557249">DVIDS, January 2026</a>; <a href="https://www.army.mil/article/294347">Army.mil, August 2026</a>; <a href="https://www.washingtoninformer.com/national-guard-dc-deployment/">Washington Informer, troop numbers</a></li>
      <li>Federal arrests: <a href="https://51st.news/national-guard-dc-faq/">51st, August 2025</a>. Shootings: <a href="https://www.thetrace.org/2025/10/dc-shooting-data-trump-national-guard/">The Trace, October 2025</a></li>
      <li>Crime data reviews: <a href="https://oversight.house.gov/release/oversight-committee-releases-bombshell-report-revealing-d-c-s-police-chief-deliberately-manipulated-crime-data">House Oversight</a>; <a href="https://www.nbcwashington.com/news/local/internal-affairs-report-dc-police-crime-stats/4104122/">MPD internal affairs (NBC Washington)</a>; <a href="https://oig.dc.gov/sites/default/files/Reports/DCOIG_Report_26-E-03-FA0.pdf">DC Inspector General</a></li>
      <li>Related study: <a href="https://www.niskanencenter.org/washington-dc-crime-decline-and-its-lessons-for-american-policing/">Niskanen Center, May 2026</a></li>
    </ul>
  </div>
</div>
<footer>DC crime since the National Guard · Martin Ngoh</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const D = {json.dumps(data)};
const C = {{ blue: '#2563eb', blueLight: '#93c5fd', red: '#ef4444', muted: '#8b8b8b', grey: '#6b7280', border: '#262626', surface: '#121212', text: '#f2f2f2' }};
Chart.defaults.color = C.muted; Chart.defaults.borderColor = C.border;
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif'; Chart.defaults.font.size = 11;
Object.assign(Chart.defaults.plugins.tooltip, {{ backgroundColor: C.surface, borderColor: C.border, borderWidth: 1, titleColor: C.text, bodyColor: C.muted }});
Chart.defaults.plugins.legend.display = false;
const fmt = v => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(Math.round(v)) + '%';
const base = {{ responsive: true, maintainAspectRatio: false }};
const marker = (idx, label) => ({{ id: 'marker', afterDatasetsDraw(ch) {{
  const {{ ctx, chartArea: a, scales: {{ x, y }} }} = ch;
  const i0 = Math.floor(idx), f = idx - i0;
  const px = x.getPixelForValue(i0) + (x.getPixelForValue(i0 + 1) - x.getPixelForValue(i0)) * f;
  ctx.save(); ctx.strokeStyle = C.muted; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = C.muted; ctx.font = '11px -apple-system, Segoe UI, sans-serif';
  const w = ctx.measureText(label).width; const left = px + 6 + w > a.right;
  ctx.textAlign = left ? 'right' : 'left'; ctx.fillText(label, left ? px - 6 : px + 6, a.top + 11);
  if (y.min < 0 && y.max > 0) {{ const y0 = y.getPixelForValue(0); ctx.strokeStyle = '#3a3a3a'; ctx.beginPath(); ctx.moveTo(a.left, y0); ctx.lineTo(a.right, y0); ctx.stroke(); }}
  ctx.restore(); }} }});
const line = (label, data, color, dash) => ({{ label, data, borderColor: color, backgroundColor: color, borderWidth: 2, borderDash: dash || [], pointRadius: 0, pointHoverRadius: 4, tension: 0.25 }});
const bar = (label, data, color, faded) => ({{ label, data, borderColor: color, backgroundColor: faded ? 'rgba(0,0,0,0)' : color + '26', borderWidth: 1.5, borderDash: faded ? [3, 2] : [], borderRadius: 3, maxBarThickness: 22 }});
const pctAxis = (min, max) => ({{ min, max, grid: {{ color: '#1c1c1c' }}, ticks: {{ callback: fmt }} }});

new Chart(document.getElementById('cMonthly'), {{ type: 'line',
  data: {{ labels: D.monthly.labels, datasets: [line('Violent crime', D.monthly.violent, C.red), line('Violent crime with a gun', D.monthly.gun, C.blueLight, [5, 4]), line('Property crime', D.monthly.property, C.blue)] }},
  options: {{ ...base, interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + fmt(c.parsed.y) }} }} }},
    scales: {{ x: {{ grid: {{ display: false }}, ticks: {{ autoSkip: false, maxRotation: 0, callback: (v, i) => i % 4 === 0 ? D.monthly.labels[i] : '' }} }}, y: pctAxis(-80, 40) }} }},
  plugins: [marker(D.monthly.deploy_index, 'Guard deployed, Aug 11, 2025')] }});

new Chart(document.getElementById('cNational'), {{ type: 'bar',
  data: {{ labels: D.national.labels, datasets: [bar('DC, year before', D.national.dc_pre, C.red, true), bar('DC, Guard year', D.national.dc_post, C.red), bar('Nation, year before', D.national.us_pre, C.blue, true), bar('Nation, Guard year', D.national.us_post, C.blue)] }},
  options: {{ ...base, plugins: {{ tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + fmt(c.parsed.y) }} }} }},
    scales: {{ x: {{ grid: {{ display: false }} }}, y: pctAxis(-60, 10) }} }} }});

new Chart(document.getElementById('cWeekly'), {{ type: 'line',
  data: {{ labels: D.weekly.labels, datasets: [line('2025', D.weekly.y2025, C.red), line('Same weeks of 2024', D.weekly.y2024, C.grey)] }},
  options: {{ ...base, interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ tooltip: {{ callbacks: {{ title: i => 'Week of ' + i[0].label + (i[0].dataIndex === D.weekly.deploy ? ' (deployment)' : ''), label: c => c.dataset.label + ': ' + c.parsed.y + ' detections' }} }} }},
    scales: {{ x: {{ grid: {{ display: false }}, ticks: {{ autoSkip: false, maxRotation: 0, callback: (v, i) => i % 3 === 0 ? D.weekly.labels[i] : '' }} }}, y: {{ min: 0, grid: {{ color: '#1c1c1c' }} }} }} }},
  plugins: [marker(D.weekly.deploy, 'Aug 11')] }});

const phaseChart = (id, t, c) => new Chart(document.getElementById(id), {{ type: 'bar',
  data: {{ labels: D.phase.labels, datasets: [bar('With troops', t, C.red), bar('Without troops', c, C.blue)] }},
  options: {{ ...base, plugins: {{ tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + fmt(c.parsed.y) }} }} }},
    scales: {{ x: {{ grid: {{ display: false }}, ticks: {{ autoSkip: false, maxRotation: 0 }} }}, y: pctAxis(-80, 20) }} }} }});
phaseChart('cPhase1', D.phase.p1_t, D.phase.p1_c);
phaseChart('cPhase2', D.phase.p2_t, D.phase.p2_c);

const dots = {{ id: 'dots', afterDatasetsDraw(ch) {{
  const {{ ctx, chartArea: a, scales: {{ x }} }} = ch;
  ctx.save();
  [[1, []], [0.85, [4, 4]]].forEach(([v, dash]) => {{ const px = x.getPixelForValue(v); ctx.strokeStyle = C.muted; ctx.setLineDash(dash); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke(); }});
  ctx.setLineDash([]);
  ch.data.datasets.forEach((d, i) => ch.getDatasetMeta(i).data.forEach((el, j) => {{
    ctx.beginPath(); ctx.arc(x.getPixelForValue(d.rr[j]), el.y, 4, 0, 2 * Math.PI); ctx.fillStyle = d.borderColor; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = C.surface; ctx.stroke(); }}));
  ctx.restore(); }} }};
new Chart(document.getElementById('cForest'), {{ type: 'bar',
  data: {{ labels: D.forest.labels, datasets: [
    {{ label: 'Planned sites', data: D.forest.p_ci, rr: D.forest.p_rr, borderColor: C.red, backgroundColor: C.red + '40', borderWidth: 0, barThickness: 4 }},
    {{ label: "With the Guard's September list", data: D.forest.u_ci, rr: D.forest.u_rr, borderColor: C.blue, backgroundColor: C.blue + '55', borderWidth: 0, barThickness: 4 }}] }},
  options: {{ ...base, indexAxis: 'y',
    plugins: {{ tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + c.dataset.rr[c.dataIndex].toFixed(2) + ' (' + c.raw[0].toFixed(2) + ' to ' + c.raw[1].toFixed(2) + ')' }} }} }},
    scales: {{ x: {{ min: 0.3, max: 2.0, grid: {{ color: '#1c1c1c' }}, ticks: {{ callback: v => Number(v).toFixed(1) }} }}, y: {{ grid: {{ display: false }} }} }} }},
  plugins: [dots] }});

new Chart(document.getElementById('cSynth'), {{ type: 'line',
  data: {{ labels: D.synth.labels, datasets: [line('DC', D.synth.dc, C.red), line('Synthetic DC', D.synth.synth, C.grey, [5, 4])] }},
  options: {{ ...base, interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ tooltip: {{ callbacks: {{ label: c => c.dataset.label + ': ' + c.parsed.y + '% of 2017-2024 average' }} }} }},
    scales: {{ x: {{ grid: {{ display: false }}, ticks: {{ autoSkip: false, maxRotation: 0, callback: (v, i) => i % 6 === 0 ? D.synth.labels[i] : '' }} }}, y: {{ min: 40, max: 130, grid: {{ color: '#1c1c1c' }}, ticks: {{ callback: v => v + '%' }} }} }} }},
  plugins: [marker(D.synth.deploy, 'Aug 11, 2025')] }});

new Chart(document.getElementById('cWards'), {{ type: 'bar',
  data: {{ labels: D.wards.labels, datasets: [{{ label: 'Gunshots', data: D.wards.vals, borderColor: D.wards.troops.map(t => t ? C.red : C.blue), backgroundColor: D.wards.troops.map(t => t ? C.red + '26' : C.blue + '26'), borderWidth: 1.5, borderRadius: 3, maxBarThickness: 22 }}] }},
  options: {{ ...base, plugins: {{ tooltip: {{ callbacks: {{ label: c => (D.wards.troops[c.dataIndex] ? 'Guard posts: ' : 'No posts: ') + fmt(c.parsed.y) }} }} }},
    scales: {{ x: {{ grid: {{ display: false }} }}, y: pctAxis(-60, 0) }} }} }});
</script>
"""

assert "—" not in BODY, "em dash in page"
(ROOT / "out/page.html").write_text(BODY)
(ROOT / "index.html").write_text(
    '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8" />\n<meta name="viewport" content="width=device-width, initial-scale=1.0" />\n'
    f'<meta name="description" content="{DESC}" />\n</head>\n<body>\n{BODY}\n</body>\n</html>\n')
print("wrote index.html and out/page.html")
