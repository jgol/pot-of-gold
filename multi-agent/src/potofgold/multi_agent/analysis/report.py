"""Render the H3 report with embedded Seaborn figures from recomputed measurements."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

from potofgold.multi_agent.analysis.allocation import POLICIES, LABELS

def summary_plots(result):
    """Seaborn figures, embedded in HTML; no standalone figure files."""
    import base64
    from io import BytesIO
    import os
    import tempfile

    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'h3-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import pandas as pd
    import seaborn as sns

    data = result['variants']['score_only']
    teal = '#277F8E'
    ink, muted = '#233C38', '#60756E'
    names = {'court': 'Court', 'deploy_all': 'Deploy everything',
             'evidence_of_skill': 'Matched allocation', 'single_llm': 'Single LLM'}

    def embedded_png(fig):
        output = BytesIO()
        fig.savefig(output, format='png', dpi=150, facecolor='white', bbox_inches='tight',
                    metadata={'Software': 'Seaborn / Matplotlib'})
        plt.close(fig)
        return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii')

    with sns.axes_style('whitegrid'), sns.plotting_context('notebook', font_scale=0.92):
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.7), sharey=True)
        fig.subplots_adjust(left=.17, right=.98, bottom=.16, top=.77, wspace=.13)
        fig.suptitle('Risk comparisons at a glance', x=.015, y=.98,
                     ha='left', fontsize=18, fontweight='bold', color=ink)
        fig.text(.015, .91, 'Score-only allocation  ·  Observed values and 95% confidence intervals',
                 fontsize=11, color=muted)
        policies = POLICIES[:4]
        for ax, field, title, color, direction in zip(
                axes, ('maxdd', 'cvar'), ('Maximum drawdown', 'CVaR'),
                ('#197C83', '#6962A6'), ('smaller is better', 'less negative is better')):
            ax.set_facecolor('#FAFBFC')
            ax.axhspan(2.5, 3.5, color=color, alpha=.055, zorder=0)
            rows, bounds = [], []
            for row, policy in enumerate(policies):
                level = data['levels'][policy]
                x = level[field]
                low, high = level[f'{field}_ci95']
                y = len(policies) - 1 - row
                point_color = color if policy == 'court' else '#7F9690'
                ax.plot([low, high], [y, y], color=point_color, alpha=.12,
                        linewidth=10, solid_capstyle='round', zorder=2)
                ax.plot([low, high], [y, y], color=point_color, linewidth=2.2, zorder=3)
                ax.vlines([low, high], y-.065, y+.065, color=point_color, linewidth=1.4, zorder=3)
                ax.annotate(f'{x:,.2f}', (x, y), xytext=(0, 11), textcoords='offset points',
                            ha='center', va='bottom', color=point_color, fontsize=11, fontweight='bold')
                ax.annotate(f'[{low:,.2f}, {high:,.2f}]', (x, y), xytext=(0, -12),
                            textcoords='offset points', ha='center', va='top',
                            color=muted, fontsize=9)
                rows.append({'value': x, 'position': y, 'policy': policy})
                bounds.extend([low, high, x])
            sns.scatterplot(data=pd.DataFrame(rows), x='value', y='position', hue='policy',
                            palette={p: color if p == 'court' else '#7F9690' for p in policies},
                            s=115, edgecolor='white', linewidth=1.5, legend=False, ax=ax, zorder=4)
            left, right = min(0, min(bounds)), max(0, max(bounds))
            padding = (right-left) * .13
            ax.set(xlim=(left-padding, right+padding), ylim=(-.6, 3.6), ylabel='',
                   xlabel=f'Book P&L units · {direction}')
            ax.set_yticks([3, 2, 1, 0], labels=[names[p] for p in policies])
            ax.set_title(title, loc='left', fontsize=13, fontweight='bold', color=color, pad=17)
            ax.grid(axis='y', visible=False)
            ax.grid(axis='x', color='#DCE3E8', alpha=.6, linewidth=.8)
            ax.tick_params(axis='both', length=0, labelcolor=muted, pad=9)
            ax.xaxis.label.set_color(muted)
            sns.despine(ax=ax, left=True, bottom=True)
        axes[0].tick_params(axis='y', labelcolor=ink, labelsize=11)
        axes[0].get_yticklabels()[0].set_fontweight('bold')
        fig.text(.17, .035, 'CVaR = mean daily P&L over the worst 5% of days', fontsize=10, color=muted)
        risk_plot = embedded_png(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), sharey=True)
        fig.subplots_adjust(left=.235, right=.98, bottom=.18, top=.73, wspace=.15)
        fig.suptitle('Court risk reductions', x=.015, y=.98,
                     ha='left', fontsize=18, fontweight='bold', color=ink)
        fig.text(.015, .905, 'Paired comparisons  ·  Percentage reductions and 95% confidence intervals',
                 fontsize=11, color=muted)
        comparators = POLICIES[1:4]
        for ax, index, title, color in zip(
                axes, (2, 1), ('Maximum drawdown', 'CVaR'), ('#197C83', '#6962A6')):
            ax.set_facecolor('#FAFBFC')
            ax.axhspan(.5, 1.5, color='#F0F3F6', zorder=0)
            rows = []
            for row, policy in enumerate(comparators):
                comparison = data['comparisons'][policy]
                value = comparison['percent'][index]
                low, high = comparison['percent_ci95'][index]
                y = len(comparators) - 1 - row
                ax.plot([low, high], [y, y], color=color, alpha=.12,
                        linewidth=10, solid_capstyle='round', zorder=2)
                ax.plot([low, high], [y, y], color=color, linewidth=2.2, zorder=3)
                ax.vlines([low, high], y-.065, y+.065, color=color, linewidth=1.4, zorder=3)
                ax.annotate(f'{value:+.1f}%', (value, y), xytext=(0, 12),
                            textcoords='offset points', ha='center', va='bottom',
                            color=color, fontsize=11, fontweight='bold')
                ax.annotate(f'[{low:+.1f}%, {high:+.1f}%]', (value, y), xytext=(0, -13),
                            textcoords='offset points', ha='center', va='top',
                            color=muted, fontsize=9)
                rows.append({'reduction': value, 'position': y})
            sns.scatterplot(data=pd.DataFrame(rows), x='reduction', y='position',
                            color=color, s=115, edgecolor='white', linewidth=1.5,
                            legend=False, ax=ax, zorder=4)
            ax.axvline(0, color='#74828B', linewidth=1.2, linestyle=(0, (3, 4)))
            ax.set(xlim=(-9, 109), ylim=(-.6, 2.6), ylabel='', xlabel='')
            ax.set_yticks([2, 1, 0], labels=[f'Court vs {names[p]}' for p in comparators])
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.set_title(title, loc='left', fontsize=13, fontweight='bold', color=color, pad=17)
            ax.grid(axis='y', visible=False)
            ax.grid(axis='x', color='#DCE3E8', alpha=.6, linewidth=.8)
            ax.tick_params(axis='both', length=0, labelcolor=muted, pad=9)
            sns.despine(ax=ax, left=True, bottom=True)
        axes[0].tick_params(axis='y', labelcolor=ink, labelsize=10)
        fig.text(.235, .04, '0% = no difference', fontsize=10, color=muted)
        fig.text(.98, .04, 'Positive reductions favour the court →', ha='right', fontsize=10, color=muted)
        reduction_plot = embedded_png(fig)

        rows = []
        for policy in POLICIES[:4]:
            v = data['levels'][policy]
            rows.append({'policy': names[policy], 'pnl': v['pnl'],
                         'pnl_to_drawdown': v['pnl']/v['maxdd'],
                         'allocation': 100*v['exposure']})
        frame = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6), sharey=True)
        fig.subplots_adjust(left=.15, right=.98, bottom=.17, top=.79, wspace=.26)
        fig.suptitle('Return and allocation context', x=.015, y=.98, ha='left',
                     fontsize=17, fontweight='bold', color=ink)
        configs = [('pnl', 'Cumulative P&L', 'Book units', (-1080, 235)),
                   ('pnl_to_drawdown', 'P&L / max drawdown', 'Study-period ratio', (-.90, .80)),
                   ('allocation', 'Mean allocation', 'Percent of full-size strategy', (0, 124))]
        for ax, (field, title, xlabel, limits) in zip(axes, configs):
            sns.barplot(data=frame, x=field, y='policy',
                        order=[names[p] for p in POLICIES[:4]], color='#A2B6AE',
                        errorbar=None, saturation=.9,
                        legend=False, ax=ax)
            ax.patches[0].set_facecolor(teal)
            for container in ax.containers:
                labels = [(f'{v:+.1f}' if field == 'pnl' else
                           f'{v:+.2f}' if field == 'pnl_to_drawdown' else f'{v:.1f}%')
                          for v in container.datavalues]
                ax.bar_label(container, labels=labels, padding=3, fontsize=9, color=ink)
            ax.axvline(0, color=muted, linewidth=1)
            ax.set(xlabel=xlabel, ylabel='', xlim=limits)
            ax.set_title(title, loc='left', fontsize=12, fontweight='bold', color=ink, pad=13)
            ax.grid(axis='y', visible=False)
            ax.tick_params(axis='x', labelsize=9)
            if field == 'allocation':
                ax.set_xticks([0, 50, 100])
                ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            sns.despine(ax=ax, left=True)
        return {'risk': risk_plot, 'reductions': reduction_plot, 'context': embedded_png(fig)}


def render(result):
    e = html.escape
    result = dict(result)
    visible_policies = list(zip(POLICIES[:4], LABELS[:4]))
    visible_names = {p for p, _ in visible_policies}
    result['variants'] = {
        variant: {**data,
                  'levels': {p: values for p, values in data['levels'].items() if p in visible_names},
                  'comparisons': {p: values for p, values in data['comparisons'].items() if p in visible_names}}
        for variant, data in result['variants'].items() if variant == 'score_only'}
    result['render_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plots = summary_plots(result)
    score = result['variants']['score_only']
    supported_comparisons = sum(score['comparisons'][p]['percent_ci95'][index][0] > 0
                                for p in POLICIES[1:4] for index in (2, 1))
    reduction_summary = (
        'All six intervals lie entirely above zero: the court reduces both risk measures against all three '
        'comparators, supporting H3 under the reported bootstrap analysis.'
        if supported_comparisons == 6 else
        f'{supported_comparisons} of six intervals lie entirely above zero; the full H3 criterion is not met.')
    headline = ('The score-only policy supports H3 against the three specified baselines.'
                if supported_comparisons == 6 else
                'The score-only policy does not meet the full H3 criterion in this analysis.')
    sections = []
    for field, metric_index, heading, definition, criterion in [
        ('maxdd', 2, 'Maximum drawdown',
         'Largest peak-to-trough decline in cumulative daily P&amp;L. Smaller is better.',
         'The lower bound of the 95% reduction interval must be above zero.'),
        ('cvar', 1, 'CVaR · worst 5% of days',
         'Mean daily P&amp;L over the worst 5% of days. Less-negative values indicate smaller tail losses.',
         'The lower bound of the 95% improvement interval must be above zero.'),
    ]:
        comparisons_rows = []
        for policy, label in visible_policies[1:]:
            c = score['comparisons'][policy]
            absolute_low, absolute_high = c['difference_ci95'][metric_index]
            low, high = c['percent_ci95'][metric_index]
            comparisons_rows.append(
                f'<tr><th scope="row">{e(label)}</th>'
                f'<td>{score["levels"]["court"][field]:,.2f}</td>'
                f'<td>{score["levels"][policy][field]:,.2f}</td>'
                f'<td>{c["difference"][metric_index]:+,.2f}'
                f'<span class="ci">[{absolute_low:+,.2f}, {absolute_high:+,.2f}]</span></td>'
                f'<td>{c["percent"][metric_index]:+.1f}%'
                f'<span class="ci">[{low:+.1f}%, {high:+.1f}%]</span></td></tr>')
        sections.append(f'''<section id="{field}">
          <h2>{heading}</h2><p>{definition}</p>
          <div class="table-wrap"><table class="comparison">
            <caption>Court and baseline values are in book P&amp;L units. Brackets show 95% confidence intervals for each reduction.</caption>
            <thead><tr><th scope="col">Comparator</th><th scope="col">Court</th><th scope="col">Baseline</th>
            <th scope="col">Absolute reduction<br>(book units; 95% CI)</th>
            <th scope="col">Relative reduction<br>(%; 95% CI)</th></tr></thead>
            <tbody>{''.join(comparisons_rows)}</tbody></table></div>
          <p class="note">Positive reductions favor the court. {criterion}</p>
        </section>''')
    context = []
    for label, format_value in [
        ('Total P&L (book units)', lambda v: f'{v["pnl"]:+,.2f}'),
        ('P&L / max drawdown', lambda v: f'{v["pnl"]/v["maxdd"]:+.2f}'),
        ('Mean allocation', lambda v: f'{100*v["exposure"]:.1f}%'),
        ('Funded strategies', lambda v: f'{v["funded"]} / {result["metadata"]["n_champions"]}'),
    ]:
        values = ''.join(f'<td>{format_value(score["levels"][policy])}</td>' for policy, _ in visible_policies)
        context.append(f'<tr><th scope="row">{e(label)}</th>{values}</tr>')
    context_headers = ''.join(f'<th scope="col">{e(label)}</th>' for _, label in visible_policies)
    meta = result['metadata']
    mode = meta.get('mode', 'archive')
    scope_title = 'H3 · Recomputed v12 evaluation · Archived study · September 2026'
    scope_note = ('All 58 source champions matched their ledgers; one degraded court panel was excluded from every comparison. '
                  'The cohort consists of champions, not a documented census of every Pareto-front member. '
                  'The original report’s unverified arm-A ledger provenance (approximately 37% of the cohort) remains unresolved.')
    source_note = 'Computed from the archived v12 court/single-LLM decisions and original sealed held-out ledgers.'
    reproduction_note = ('From <code>multi-agent/</code>, run <code>potofgold-court-evaluate --output runs/reproduction/metrics.json</code>, '
                         'then <code>potofgold-court-report --metrics runs/reproduction/metrics.json --output runs/reproduction/report.html</code>.')
    if mode != 'archive':
        scope_title = 'Synthetic simulation · Mock agents' if mode == 'simulation' else 'New-data assessment'
        scope_note = e(' '.join(meta.get('limitations', [])))
        source_note = 'Computed from synthetic inputs.' if mode == 'simulation' else 'Computed from the supplied new-run decisions and outcomes.'
        reproduction_note = 'Use the commands in the module README to run this workflow.'
    if mode == 'simulation':
        headline = 'Offline simulation completed; no LLM providers were called.'
        reduction_summary = 'These illustrative results exercise the calculations and do not test H3 or compare real LLM reasoning.'
    embedded = json.dumps(result, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 · Score-only allocation and baseline comparisons</title>
<style>
:root{{color-scheme:light;--ink:#243b34;--muted:#60716a;--line:#dce5df;--green:#166650;--amber:#805c1b}}
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f2;color:var(--ink);font:15px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1120px;margin:32px auto;padding:44px 52px;background:white;border:1px solid var(--line);border-radius:12px}}
.eyebrow{{color:var(--green);font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}}h1{{font:500 clamp(30px,4vw,43px)/1.18 Georgia,serif;letter-spacing:-.02em;margin:12px 0 14px}}h2{{font-size:21px;line-height:1.3;margin:0 0 16px}}h3{{font-size:16px;margin:22px 0 10px}}p{{margin:0 0 14px}}section{{border-top:1px solid var(--line);padding-top:26px;margin-top:28px}}
.subhead,.note,footer{{color:var(--muted)}}.verdict{{background:#edf6ef;border-left:4px solid #408669;padding:18px 22px;margin:24px 0;border-radius:0 6px 6px 0}}.verdict strong{{display:block;color:#235941;font-size:19px;margin-bottom:7px}}.verdict p{{margin:0}}.note{{font-size:12px;line-height:1.65}}
.worse{{background:#f9e8e5;color:#923f35}}summary{{cursor:pointer;font-weight:650}}nav{{font-size:14px;margin:16px 0}}
.formula{{padding:16px 18px;border:1px solid var(--line);border-radius:7px;background:#f7faf7}}.formula h3{{margin:0 0 8px}}.equation{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:14px}}.formula p:last-child{{margin-bottom:0}}
figure{{margin:18px 0 22px}}figure img{{display:block;width:100%;height:auto}}figcaption{{color:var(--muted);font-size:12px;line-height:1.65;margin-top:12px}}
.comparison td:nth-child(2),.comparison thead th:nth-child(2){{background:#eaf5ee;font-weight:650}}
.table-wrap{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5;font-variant-numeric:tabular-nums}}caption{{text-align:left;color:var(--muted);font-size:12px;margin-bottom:9px}}th,td{{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top}}thead th{{background:#f0f5f1;font-size:12px}}th{{text-align:left}}tbody th{{font-weight:600}}td{{text-align:right;white-space:nowrap}}thead th:not(:first-child){{text-align:right}}.court{{background:#eaf5ee;font-weight:650}}.control{{background:#f4f6f9}}.ci{{display:block;font-size:12px;color:var(--muted);margin-top:3px}}.status{{font-size:11px;display:inline-block;padding:3px 7px;border-radius:4px}}.yes{{background:#e3f2e9;color:var(--green)}}.no{{background:#fbf1dc;color:var(--amber)}}ul{{margin:0;padding-left:21px}}li{{margin:8px 0}}a{{color:var(--green);text-underline-offset:3px}}code{{font-size:.9em}}footer{{border-top:1px solid var(--line);padding-top:18px;margin-top:28px;font-size:12px}}footer p{{margin:7px 0}}
@media(max-width:700px){{main{{margin:0;padding:26px 18px;border:0;border-radius:0}}.formulae{{grid-template-columns:1fr}}th,td{{padding:9px 7px}}}}
@media print{{body{{background:white;font-size:9pt}}main{{max-width:none;margin:0;border:0;padding:0}}h1{{font-size:25pt}}h2{{font-size:14pt}}table{{font-size:8pt}}section{{margin-top:18px;padding-top:15px}}thead{{display:table-header-group}}tr,.verdict,.formula{{break-inside:avoid}}h2,h3{{break-after:avoid}}*{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style></head><body><main>
<header><div class="eyebrow">{scope_title}</div>
<h1>Multi-agent validation of evolved 0DTE strategies</h1>
<p class="subhead">Score-only allocation, with separate drawdown and CVaR results against the three specified baselines.</p>
<nav aria-label="Results"><a href="#risk-plot">Risk comparisons</a> · <a href="#risk-reductions">Court reductions</a> · <a href="#maxdd">Maximum drawdown</a> · <a href="#cvar">CVaR</a> · <a href="#context">Return and allocation</a></nav>
<div class="verdict"><strong>{headline}</strong>
<p>The tables below place the court and each baseline side by side, with absolute and percentage reductions
and their 95% confidence intervals.</p></div></header>
<section><h2>Hypothesis assessed</h2>
<p>A multi-agent LLM validation layer—a prosecutor, a defender, and judges—applied to STGP-evolved 0DTE strategies reduces realized losses measured by 5% Conditional Value at Risk and maximum drawdown, at 95% confidence, compared with deploy-all, rule-based selection, and a single LLM.</p>
<p class="note">The 5% specifies the CVaR tail fraction; 95% specifies the confidence level. The hypothesis does not specify a minimum percentage reduction.</p></section>
<section><h2>Allocation rule</h2>
<p><strong>WONT_GENERALIZE receives zero allocation.</strong> Otherwise, allocation decreases monotonically with risk score:</p>
<div class="formula">
<p class="equation">size = (100 − risk score) / 100</p>
<p class="note">A risk score of 40 produces size 0.60, or 60% of one full-size strategy.</p></div>
<p class="note" style="margin-top:12px">The court uses the panel’s median risk score; the single LLM uses its own risk score.
The court funds {score['levels']['court']['funded']} strategies and the single LLM funds {score['levels']['single_llm']['funded']}.</p></section>
<section><h2>How the baselines are made comparable</h2>
<ul><li><strong>Discard no strategies:</strong> deploy-all funds all {meta['n_champions']} champions at full size, without validation.</li>
<li><strong>Evidence of skill:</strong> rank all champions by net training return and assign the court’s exact allocation sizes
from largest to smallest along that ranking. This preserves the allocation multiset, funded count, and total nominal allocation.
Individual court scores and held-out returns do not determine the rule’s ranking. Training-return ties are resolved by champion ID.</li>
<li><strong>Single LLM:</strong> use the same score-only formula and WONT veto as the court, with the single model’s own risk scores and verdicts.</li></ul>
<p class="note" style="margin-top:14px">All policies use the same {meta['n_champions']} champions and {meta['n_days']} dated held-out observations
({e(meta['start'])}–{e(meta['end'])}). Zero-allocation days remain on the calendar. The evidence-of-skill control uses the court’s
aggregate sizing schedule, but chooses where to place those sizes solely from training returns. No new LLM calls or fitting on test outcomes were used.</p>
<p><strong>H3 criterion:</strong> for each specified baseline, the 95% confidence intervals for CVaR improvement and maximum-drawdown reduction must both lie entirely above zero. The metrics are assessed separately below.</p></section>
<section id="risk-plot"><h2>Risk comparisons at a glance</h2>
<figure><img src="{plots['risk']}" alt="Maximum drawdown and CVaR with 95% confidence intervals for Court, Deploy everything, Matched allocation, and Single LLM.">
<figcaption>Points show each policy’s observed value; horizontal lines and brackets show its 95% bootstrap confidence interval.
The court is emphasized. Values are in book P&amp;L units. Smaller drawdown and less-negative CVaR are better.
The tables below report paired differences between policies and their confidence intervals.</figcaption></figure></section>
<section id="risk-reductions"><h2>Court risk reductions</h2>
<figure><img src="{plots['reductions']}" alt="Court percentage reductions in maximum drawdown and 5% CVaR loss against Deploy everything, Matched allocation, and Single LLM, with paired 95% confidence intervals and a zero reference line.">
<figcaption>Each dot shows how much the court reduces risk relative to the named comparator; lines and brackets
show the paired 95% bootstrap confidence interval. Zero means no difference. An interval entirely to the right
of zero supports a reduction. CVaR reductions refer to the magnitude of the worst-5% mean daily loss.
Matched allocation is the rule-based comparator.</figcaption></figure>
<p><strong>{reduction_summary}</strong></p></section>
{''.join(sections)}
<section id="context"><h2>Return and allocation context</h2>
<figure><img src="{plots['context']}" alt="Court and baseline cumulative P&amp;L, P&amp;L divided by maximum drawdown, and mean allocation.">
<figcaption>The court is teal; baselines are muted green. P&amp;L / maximum drawdown measures the study-period gain or loss
relative to the largest decline: higher is better, and a negative value indicates an overall loss. This is a nonannualized,
descriptive ratio. Mean allocation shows how much of a full-size strategy each policy funds, averaged over all {meta['n_champions']} champions.
These bars show observed values, without confidence intervals.</figcaption></figure>
<div class="table-wrap"><table class="comparison">
<caption>Observed P&amp;L and allocation context. The study-period P&amp;L/drawdown ratio is nonannualized.</caption>
<thead><tr><th scope="col">Measure</th>{context_headers}</tr></thead>
<tbody>{''.join(context)}</tbody></table></div></section>
<section><h2>Interpretation</h2>
<p><strong>{headline}</strong> {reduction_summary}</p>
<p><strong>Evidence of skill receives the court’s exact allocation sizes.</strong> This control preserves the funded count and total nominal allocation, assigning sizes by training-return rank. The tables show whether its CVaR and drawdown comparisons favor the court.</p>
<p><strong>These are comparisons of complete allocation policies.</strong> Evidence of skill matches the court’s allocation sizes by construction; deploy-all and the single LLM retain their specified allocation rules. Their mean sizes and funded counts are shown in the allocation context table.</p></section>
<section><h2>Metrics, uncertainty, and scope</h2>
<p>CVaR is mean daily P&amp;L over the worst 5% of days ({math.ceil(.05*meta['n_days'])} days here), not the single worst day.
Less-negative CVaR is better. Maximum drawdown is the largest peak-to-trough decline of cumulative daily P&amp;L, starting from zero.
P&amp;L and drawdown are in the source’s book units, not percentage returns. Mean size is the average allocation fraction across the {meta['n_champions']} champions;
matching it controls nominal allocation, not realized margin or risk on every individual day.</p>
<p class="note">Tables show rounded values; reductions are calculated from unrounded results. Absolute drawdown reduction is
baseline minus court. Absolute CVaR loss reduction is court minus baseline because CVaR is reported as signed P&amp;L.
Relative reduction divides the absolute reduction by the magnitude of the baseline value.</p>
<p class="note">All intervals were recomputed using {result['samples']:,} paired bootstrap samples and identical resampled dates for every policy.
P&amp;L/CVaR use iid day resampling; drawdown uses circular moving blocks of {result['block_length']} days. Seed {result['seed']}; percentile 95% intervals.
CVaR resampling does not preserve serial dependence. Intervals are pairwise and condition on the recorded strategies and LLM decisions.</p>
<p class="note"><strong>Scope and provenance:</strong> {scope_note}</p></section>
<footer><p>{source_note}
The released <code>potofgold-court-evaluate</code> and <code>potofgold-court-report</code> commands reproduce this report.</p>
<p>{reproduction_note}</p>
<p>One self-contained HTML report. Figures use Seaborn and Matplotlib and are embedded directly. Numerical results for the displayed policies,
resampling settings, and input hashes are embedded below as non-executing JSON; no external assets are required.</p></footer>
</main><script type="application/json" id="analysis-results">{embedded}</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.metrics.read_text())
    if result.get('schema_version') != 1:
        parser.error('Unsupported metrics schema')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(result))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
