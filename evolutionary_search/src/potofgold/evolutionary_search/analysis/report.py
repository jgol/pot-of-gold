"""Render evaluated search measurements as one HTML file with embedded Seaborn figures."""
from __future__ import annotations

import argparse
import base64
from html import escape
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

from .archive import CONDITIONS, LABELS

COLORS = {'indicators': '#78889b', 'embeddings': '#2e8c86', 'combined': '#9b79b2', 'permuted_embeddings': '#d19858'}


def _number(value, digits=2, percent=False):
    if value is None:
        return '—'
    return f'{100 * value:.1f}%' if percent else f'{value:,.{digits}f}'


def _interval(value, interval, digits=2):
    point = _number(value, digits)
    if not interval or any(v is None for v in interval):
        return point
    return f'{point} <span class="ci">[{_number(interval[0], digits)}, {_number(interval[1], digits)}]</span>'


def _table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def _figures(record):
    # Render directly into memory: no sibling image files are required to share the report.
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'potofgold-plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style='whitegrid', context='notebook', font='DejaVu Sans',
                  rc={'axes.spines.top': False, 'axes.spines.right': False, 'grid.alpha': .22,
                      'axes.labelcolor': '#405268', 'text.color': '#263849'})
    output = []
    discovery = pd.DataFrame(record['discovery'])
    if len(discovery):
        display = discovery[discovery.condition != 'permuted_embeddings'].copy()
        if display.empty:
            display = discovery.copy()
        display['yield_percent'] = 100 * display.discovery_yield
        order = [c for c in CONDITIONS if c in set(display.condition)]
        fig, ax = plt.subplots(figsize=(9.6, 4.2))
        sns.lineplot(data=display, x='seed', y='yield_percent', hue='condition', hue_order=order,
                     palette=COLORS, marker='o', markersize=8, linewidth=2, estimator=None, ax=ax)
        ax.set(xlabel='Search seed', ylabel='Profitable strategies (%)',
               title='Discovery yield across search seeds', xticks=sorted(display.seed.unique()))
        ax.set_ylim(bottom=0)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, [LABELS[c] for c in order], frameon=False, title=None,
                  loc='upper center', bbox_to_anchor=(.5, -.19), ncol=min(3, len(order)))
        fig.tight_layout()
        output.append(('Discovery yield across seeds', _encode_figure(fig)))
    curves = []
    for row in record.get('daily_portfolios', []):
        if row['condition'] == 'permuted_embeddings':
            continue
        cumulative = np.cumsum(row['active_pnl'])
        curves.extend(dict(date=d, cumulative=v, condition=row['condition'], seed=row['seed'])
                      for d, v in zip(row['dates'], cumulative))
    if curves:
        df = pd.DataFrame(curves); df['date'] = pd.to_datetime(df.date)
        fig, ax = plt.subplots(figsize=(9.6, 4.4))
        order = [c for c in CONDITIONS if c in set(df.condition)]
        sns.lineplot(data=df, x='date', y='cumulative', hue='condition', hue_order=order,
                     units='seed', estimator=None, palette=COLORS, alpha=.2, linewidth=.8, legend=False, ax=ax)
        sns.lineplot(data=df, x='date', y='cumulative', hue='condition', hue_order=order,
                     estimator='mean', errorbar=None, palette=COLORS, linewidth=2.4, ax=ax)
        ax.axhline(0, color='#68768a', linewidth=.8)
        ax.set(xlabel='Held-out evaluation date', ylabel='Cumulative proxy P&L units',
               title='Cumulative P&L of the active-member average')
        handles, _ = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        fig.legend(handles, [LABELS[c] for c in order], title=None, frameon=False,
                   loc='lower center', bbox_to_anchor=(.5, 0), ncol=min(3, len(order)))
        fig.autofmt_xdate(); fig.tight_layout(rect=(0, .12, 1, 1))
        output.append(('Cumulative held-out P&L; thin lines are seeds, thick lines are their mean', _encode_figure(fig)))
    return output


def _encode_figure(fig):
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def render(record):
    if record.get('schema_version') != 1 or 'discovery' not in record or 'portfolios' not in record:
        raise ValueError('This is not a search metrics file. Run the evaluation command first.')
    figures = _figures(record)
    source = record.get('source', {})
    synthetic = source.get('synthetic_data') is True
    title = escape('Synthetic demonstration: genetic programming search' if synthetic
                   else record.get('title', 'Genetic programming search results'))
    s = record['summary']; methods = record['methods']; archived = source.get('kind') == 'archived experiment'
    if synthetic:
        description = 'Synthetic inputs demonstrate the workflow. These measurements do not reproduce the original experiment or establish performance on real market data.'
    elif source.get('contains_synthetic_data'):
        description = 'The supplied runs include synthetic inputs mixed with other or unidentified data sources. Read their measurements as separate experiments.'
    elif archived:
        description = ('The experiment evaluates whether transformer embeddings improve genetic programming '
                       'discovery of zero-days-to-expiration (0DTE) options strategies. These measurements '
                       'compare indicators, embeddings, combined inputs and a permuted-embedding control '
                       'using the saved strategy outcomes.')
    else:
        description = 'Measurements from the supplied search runs.'
    parts = [f'<header><p class="eyebrow">EVOLUTIONARY SEARCH</p><h1>{title}</h1>',
             f'<p>{description}</p>',
             f'<p class="meta">{s["front_members"]:,} saved front members · {s["folds"]} condition/seed/fold runs · {s["calendar_days"]} held-out dates<br>{escape(s["date_start"])} to {escape(s["date_end"])}</p></header>']
    if archived:
        parts.append('<p>Each main condition uses five search seeds and five folds; the control uses one seed and five folds. Each run uses a search budget of 256 individuals and 100 generations. Whole days are grouped into folds, with four folds used for training and one for evaluation. The frozen encoder was trained over part of this evaluation calendar, so held-out GP dates were not necessarily unseen during encoder training.</p>')
    parts.append('<h2>Strategy discovery</h2><p>Discovery yield is the fraction of eligible saved front members with positive training and held-out mean profit and loss (P&amp;L). Eligibility requires held-out trading activity and a finite training mean. These descriptive counts are separate from the search\'s champion acceptance tests. Champions outside the saved fronts are excluded.</p>')
    parts.append(_table(['Input', 'Seed', 'Profitable / eligible', 'Yield', 'Saved members'],
        [[LABELS[r['condition']], str(r['seed']), f'{r["profitable_both"]} / {r["eligible_members"]}',
          _number(r['discovery_yield'], percent=True), str(r['saved_members'])] for r in record['discovery']]))
    if figures:
        parts.append(f'<figure><img alt="{escape(figures[0][0])}" src="data:image/png;base64,{figures[0][1]}"></figure>')
    if record.get('discovery_contrasts'):
        parts.append('<h3>Discovery differences against indicators</h3>')
        parts.append(_table(['Input', 'Seed', 'Difference, percentage points [95% CI]'],
            [[LABELS[r['condition']], str(r['seed']), _interval(r['difference_pp'], r['ci_pp'])]
             for r in record['discovery_contrasts']]))
        prefix = 'The original discovery intervals are retained: ' if archived else 'Discovery intervals use '
        parts.append('<p class="note">CI denotes a 95% confidence interval. ' + prefix + '2,000 independent strategy-record resamples per comparison, generator seed 0 reset for each comparison, and 2.5th/97.5th percentiles. These assume independent strategy records and are conditional on the recorded evaluation period.</p>')
    else:
        parts.append('<p class="note">Cross-input differences require an indicators run with the same search seed and held-out dates. Individual-run metrics remain available below.</p>')
    parts.append('<h2>Held-out portfolio performance</h2><p>These values use the daily average among strategies that traded. Nontrading dates contribute zero. P&amp;L, tail loss and drawdown use the pricing proxy\'s native units, without a dollar account or capital base. Brackets show 95% confidence intervals (CI); a dash indicates an unavailable statistic.</p>')
    active = [r for r in record['portfolios'] if r['allocation'] == 'active']
    parts.append(_table(['Input', 'Seed', 'Mean P&L/day [95% CI]', 'Annual Sharpe', 'PSR(0)', 'PSR(1)', '5% CVaR loss', 'Max drawdown'],
        [[LABELS[r['condition']], str(r['seed']), _interval(r['mean_pnl'], r['mean_pnl_ci'], 3),
          _number(r['sharpe_annual']), _number(r['psr0']), _number(r['psr1']),
          _number(r['cvar5_loss']), _number(r['max_drawdown'])] for r in active]))
    if len(figures) > 1:
        parts.append(f'<figure><img alt="{escape(figures[1][0])}" src="data:image/png;base64,{figures[1][1]}"><figcaption>Thin lines show individual seeds; thick lines show their mean. Each date uses the strategies from its held-out fold.</figcaption></figure>')
    parts.append(f'<p class="note">Portfolio intervals use {methods["portfolio"]["replicates"]:,} shared circular block resamples of {methods["portfolio"]["block_days"]} dates. Active-member averaging is a retrospective summary; it does not define a live capital-allocation policy. Fold-stitched curves are held-out evaluations, not a live deployment record.</p>')
    parts.append('<h2>Strategy quality and selection stability</h2><p>Quality eligibility requires at least 20 held-out trading days. The probabilistic Sharpe ratio (PSR) uses annual Sharpe benchmarks of zero and one. The probability of backtest overfitting (PBO) diagnostic is calculated within each saved front and summarized across folds.</p>')
    parts.append(_table(['Input', 'Seed', 'Quality eligible', 'PSR(0) ≥ 0.90', 'PSR(1) ≥ 0.90', 'Mean PBO', 'PBO range'],
        [[LABELS[r['condition']], str(r['seed']), str(r['quality_eligible']), str(r['psr0_pass']), str(r['psr1_pass']),
          _number(r['pbo_mean']), f'{_number(r["pbo_min"])}–{_number(r["pbo_max"])}'] for r in record['discovery']]))
    parts.append('<p class="note">The PBO diagnostic uses eight date blocks and 70 symmetric splits after removing flat and duplicate return vectors. It describes selection among surviving strategies; the archive cannot reconstruct every candidate considered during evolution.</p>')
    baseline = [r for r in record['portfolios'] if r['allocation'] == 'fixed' and r['baseline_mean_pnl'] is not None]
    if baseline:
        parts.append('<h2>Always-enter reference</h2><p>Strategies and their same-structure reference receive equal weight across the full saved front. The reference enters every eligible day; excess is evolved minus reference daily P&amp;L.</p>')
        parts.append(_table(['Input', 'Seed', 'Evolved mean P&L/day', 'Reference mean P&L/day', 'Mean excess/day'],
            [[LABELS[r['condition']], str(r['seed']), _number(r['mean_pnl'], 3),
              _number(r['baseline_mean_pnl'], 3), _number(r['mean_baseline_excess'], 3)] for r in baseline]))
    if record.get('negative_control'):
        parts.append('<h2>Permuted-embedding control</h2><p>The original and permuted searches are compared on their shared held-out dates. Positive differences favor original embeddings for yield, mean P&amp;L, Sharpe and PSR; lower drawdown, tail loss and concentration are preferable.</p>')
        names = dict(discovery_yield='Profitable fraction', mean_pnl='Mean P&L/day', sharpe_annual='Annual Sharpe',
                     psr0='Portfolio PSR(0)', psr1='Portfolio PSR(1)', max_drawdown='Maximum drawdown',
                     cvar5_loss='5% CVaR loss', largest_positive_day_share='Largest winning-day share')
        parts.append(_table(['Seed', 'Metric', 'Original', 'Permuted', 'Difference [95% CI where calculated]'],
            [[str(r['seed']), names[r['metric']], _number(r['original'], 3), _number(r['permuted'], 3),
              _interval(r['difference'], r['ci'], 3)] for r in record['negative_control']]))
        parts.append('<p class="note">Fractions are on a 0–1 scale. Control intervals use paired date resampling; the main discovery differences use the separate strategy-record method above. One permuted search is one control experiment, not a full permutation distribution.</p>')
    parts.extend(['<h2>Definitions and reproduction</h2><ul>',
        '<li>Maximum drawdown is the largest decline from a cumulative P&amp;L peak to a later trough. It is not a percentage return without a specified capital base.</li>',
        '<li>5% conditional value at risk (CVaR) is the nonnegative loss magnitude from the mean of the worst 5% of daily outcomes, rounding the number of observations up. Smaller values indicate less tail loss.</li>',
        '<li>Annual Sharpe is the daily mean P&amp;L divided by its sample standard deviation, scaled by the square root of 252. PSR(0) and PSR(1) compare that estimate with annual Sharpe benchmarks of zero and one. PSR assumes independent, identically distributed observations and uses an asymptotic approximation with skewness and kurtosis adjustments; it is not a calibrated probability of future profitability.</li>',
        '<li>PBO measures how often the strategy selected on one half of a date-block split ranks at or below the median on its complementary half. Here it assesses selection within saved fronts, rather than the complete adaptive search.</li>',
        '<li>Saved strategy-level outcomes reproduce these measurements without the licensed minute data. Rerunning the original search or replaying strategies requires the corresponding original data and fitted inputs.</li>',
        '<li>The archived training P&amp;L arrays lack date keys, and fitted embedding readout coefficients are not present in the strategy records.</li>' if archived else '',
        '</ul><p class="note">The metrics JSON contains per-strategy measurements, fold diagnostics, daily portfolio series, calculation settings and source checksums. The report is self-contained and opens offline.</p>'])
    style = '''body{margin:0;background:#f5f7fa;color:#263849;font:15px/1.6 system-ui,-apple-system,sans-serif}main{max-width:1160px;margin:32px auto;padding:44px 48px;background:#fff;border:1px solid #e0e6eb;border-radius:12px}h1{font-size:32px;line-height:1.2;margin:8px 0 15px}h2{font-size:23px;margin:40px 0 10px;border-top:1px solid #e3e9ee;padding-top:25px}h3{font-size:18px;margin-top:28px}.eyebrow{font-size:12px;letter-spacing:.13em;color:#2e8c86;font-weight:700}.meta,.note,figcaption{color:#627488;font-size:13px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;margin:14px 0 18px}th{text-align:left;background:#edf3f5;color:#405268;padding:11px 9px;white-space:normal}td{padding:9px;border-bottom:1px solid #e7ecf0;white-space:nowrap}tbody tr:nth-child(even){background:#fafcfd}.ci{display:block;color:#6b7888;font-size:12px}figure{margin:25px 0}img{width:100%;height:auto}figcaption{text-align:center}li{margin:7px 0}@media(max-width:750px){main{margin:0;padding:24px 16px;border:0}h1{font-size:26px}}'''
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>' + style + '</style></head><body><main>' + '\n'.join(parts) + '</main></body></html>\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='Metrics JSON produced by search evaluation.')
    parser.add_argument('--out', type=Path, required=True, help='One self-contained HTML report.')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing HTML report.')
    args = parser.parse_args(argv)
    if args.results.resolve() == args.out.resolve():
        parser.error('The report must not overwrite the metrics input.')
    if args.out.exists() and not args.overwrite:
        parser.error('The output already exists; choose a new path or add --overwrite.')
    try:
        record = json.loads(args.results.read_text())
        html = render(record)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(html)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f'{exc}. Create a metrics JSON with the evaluation command before rendering.')
    print(f'Report saved to {args.out}')


if __name__ == '__main__':
    main()
