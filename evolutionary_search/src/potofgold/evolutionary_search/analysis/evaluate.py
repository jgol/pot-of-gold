"""Recalculate search metrics from saved daily outcomes, without licensed minute data."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import ndtr
import scipy

from .archive import CONDITIONS, LABELS, load_archive, load_runs, write_json
from .metrics import psr_summary, cscv_front_pbo


def _ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.quantile(values, [.025, .975]).tolist() if len(values) else [None, None]


def _weights(n_days, replicates, block_length, seed):
    """One shared circular-block sample of the ordered market calendar per draw."""
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, n_days), dtype=np.int32)
    for b in range(replicates):
        starts = rng.integers(0, n_days, size=(n_days + block_length - 1) // block_length)
        selected = ((starts[:, None] + np.arange(block_length)) % n_days).ravel()[:n_days]
        weights[b] = np.bincount(selected, minlength=n_days)
    return weights


def _strategy_contrast(a, b, replicates=2000):
    """Retain the dissertation's independently resampled strategy-record interval."""
    if not len(a) or not len(b):
        return None
    rng = np.random.default_rng(0)  # Historical protocol resets the generator for each contrast.
    draws = np.array([rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
                      for _ in range(replicates)])
    return dict(difference_pp=float(100 * (a.mean() - b.mean())),
                ci_pp=(100 * np.quantile(draws, [.025, .975])).tolist())


def _risk(x):
    ps0, ps1 = psr_summary(x, 0), psr_summary(x, 1)
    curve = np.r_[0, np.cumsum(x)]
    gains = x[x > 0]
    tail = np.sort(x)[:max(1, int(np.ceil(.05 * len(x))))]
    return dict(mean_pnl=float(x.mean()), total_pnl=float(x.sum()),
                sharpe_annual=ps0['sharpe_annual'], psr0=ps0['psr'], psr1=ps1['psr'],
                cvar5_loss=float(max(0, -tail.mean())),
                max_drawdown=float(np.max(np.maximum.accumulate(curve) - curve)),
                largest_positive_day_share=float(gains.max() / gains.sum()) if len(gains) else None)


def evaluate(folds, source, *, replicates=10000, block_length=5, bootstrap_seed=20260917):
    """Evaluate saved fronts; additional champions remain available as separate files."""
    groups = defaultdict(list)
    for f in folds:
        groups[(f.condition, f.seed)].append(f)
    keys = sorted(groups, key=lambda x: (CONDITIONS.index(x[0]), x[1]))
    all_days = sorted({d for f in folds for d in f.days})
    member_rows = []; fold_rows = []; discovery = []; labels = {}; series = {}; portfolios = []
    print(f'Evaluating {len(folds)} saved folds and {sum(len(f.records) for f in folds)} front members.', flush=True)
    for key in keys:
        condition, seed = key
        dates = sorted({d for f in groups[key] for d in f.days})
        date_index = {d: i for i, d in enumerate(dates)}
        active_curve = np.zeros(len(dates)); fixed_curve = np.zeros(len(dates)); base_curve = np.full(len(dates), np.nan)
        activity = np.zeros(len(dates), dtype=int)
        group_members = []; group_pbo = []; indicator = []
        # A local full-calendar matrix supports control bootstraps. Zeros outside a
        # member's own fold are computational padding, not observations of that strategy.
        member_matrix = np.zeros((len(dates), sum(len(f.records) for f in groups[key])))
        train_positive = []; cursor = 0
        for f in groups[key]:
            indices = [date_index[d] for d in f.days]
            n_active = f.active.sum(1); total = f.pnl.sum(1); width = len(f.records)
            active_curve[indices] = np.divide(total, n_active, out=np.zeros_like(total), where=n_active > 0)
            fixed_curve[indices] = total / width
            base_curve[indices] = f.baseline.mean(1)
            activity[indices] = n_active
            member_matrix[np.ix_(indices, range(cursor, cursor + width))] = f.pnl
            fold_members = []
            for j, rec in enumerate(f.records):
                x = f.pnl[:, j]; n_traded = int(f.active[:, j].sum()); train = f.train[j]
                train_mean = float(train.mean()) if len(train) else np.nan
                both = bool(x.sum() > 0 and train_mean > 0)
                eligible = n_traded > 0 and np.isfinite(train_mean)
                if eligible:
                    indicator.append(float(both))
                p0, p1 = psr_summary(x, 0), psr_summary(x, 1)
                quality = n_traded >= 20
                genome = rec['genome']; shape = genome.get('shape', '')
                if str(genome.get('shape_tree', '')).startswith('Shape_') and '(' not in genome['shape_tree']:
                    shape = genome['shape_tree'][6:]
                row = dict(strategy_id=rec['strategy_id'], condition=condition, seed=seed, fold=f.fold,
                    shape=shape, n_days=len(x), n_traded_days=n_traded, trade_count=rec['trade_count'],
                    training_mean_pnl=train_mean, total_pnl=float(x.sum()),
                    positive_oos=bool(x.sum() > 0), positive_train_oos=both, discovery_eligible=bool(eligible),
                    quality_eligible=quality, sharpe_annual=p0['sharpe_annual'],
                    psr0=p0['psr'] if quality else None, psr1=p1['psr'] if quality else None,
                    psr0_pass=bool(quality and p0['psr'] >= .9), psr1_pass=bool(quality and p1['psr'] >= .9))
                member_rows.append(row); group_members.append(row); fold_members.append(row)
                train_positive.append(train_mean > 0)
            pbo = cscv_front_pbo(f.pnl)
            if np.isfinite(pbo['pbo']):
                group_pbo.append(pbo['pbo'])
            fold_rows.append(dict(condition=condition, seed=seed, fold=f.fold,
                saved_members=width, profitable_both=sum(r['positive_train_oos'] for r in fold_members),
                discovery_eligible=sum(r['discovery_eligible'] for r in fold_members),
                pbo={k: v for k, v in pbo.items() if k != 'partitions'}))
            cursor += width
        indicator = np.asarray(indicator); labels[key] = indicator
        discovery.append(dict(condition=condition, seed=seed, saved_members=len(group_members),
            eligible_members=len(indicator), profitable_both=int(indicator.sum()),
            discovery_yield=float(indicator.mean()) if len(indicator) else None,
            positive_oos=sum(r['positive_oos'] for r in group_members),
            quality_eligible=sum(r['quality_eligible'] for r in group_members),
            psr0_pass=sum(r['psr0_pass'] for r in group_members), psr1_pass=sum(r['psr1_pass'] for r in group_members),
            pbo_mean=float(np.mean(group_pbo)) if group_pbo else None,
            pbo_median=float(np.median(group_pbo)) if group_pbo else None,
            pbo_min=min(group_pbo) if group_pbo else None, pbo_max=max(group_pbo) if group_pbo else None))
        series[key] = dict(dates=dates, active=active_curve, fixed=fixed_curve,
                           matrix=member_matrix, train_positive=np.asarray(train_positive))
        for allocation, curve in [('active', active_curve), ('fixed', fixed_curve)]:
            portfolios.append(dict(condition=condition, seed=seed, allocation=allocation,
                n_days=len(dates), n_traded_days=int(np.sum(activity > 0)), **_risk(curve),
                baseline_mean_pnl=float(base_curve.mean()) if allocation == 'fixed' and np.isfinite(base_curve).all() else None,
                mean_baseline_excess=float((curve - base_curve).mean()) if allocation == 'fixed' and np.isfinite(base_curve).all() else None))
    contrasts = []
    for condition, seed in keys:
        if condition not in ('embeddings', 'combined') or ('indicators', seed) not in labels:
            continue
        # Compare only matched evaluation dates; a partial run is still reportable alone.
        if series[(condition, seed)]['dates'] != series[('indicators', seed)]['dates']:
            continue
        contrast = _strategy_contrast(labels[(condition, seed)], labels[('indicators', seed)])
        if contrast:
            contrasts.append(dict(condition=condition, reference='indicators', seed=seed, **contrast))
    print(f'Calculating {replicates:,} paired calendar bootstrap draws.', flush=True)
    weights_by_calendar = {}
    for row in portfolios:
        key = (row['condition'], row['seed']); s = series[key]; calendar = tuple(s['dates'])
        if calendar not in weights_by_calendar:
            weights_by_calendar[calendar] = _weights(len(calendar), replicates, block_length,
                                                    bootstrap_seed + block_length * 1009)
        w = weights_by_calendar[calendar]
        row['mean_pnl_ci'] = _ci(w @ s[row['allocation']] / len(calendar))
    control = []
    for condition, seed in keys:
        if condition != 'permuted_embeddings' or ('embeddings', seed) not in series:
            continue
        c = series[('embeddings', seed)]; p = series[(condition, seed)]
        if c['dates'] != p['dates']:
            continue
        w = weights_by_calendar[tuple(c['dates'])]; n = len(c['dates'])
        # Only the negative control requires day-resampled discovery rates; the
        # main discovery comparison above deliberately retains strategy resampling.
        rate_draws = []
        for s in (c, p):
            positive = csr_matrix(s['matrix'].T).dot(w.astype(float).T).T > 0
            rate_draws.append((positive & s['train_positive'][None, :]).mean(1))
        original = (c['matrix'].sum(0) > 0) & c['train_positive']
        perturbed = (p['matrix'].sum(0) > 0) & p['train_positive']
        control.append(dict(seed=seed, metric='discovery_yield', original=float(original.mean()),
            permuted=float(perturbed.mean()), difference=float(original.mean() - perturbed.mean()),
            ci=_ci(rate_draws[0] - rate_draws[1])))
        control.append(dict(seed=seed, metric='mean_pnl', original=float(c['active'].mean()),
            permuted=float(p['active'].mean()), difference=float(c['active'].mean() - p['active'].mean()),
            ci=_ci(w @ (c['active'] - p['active']) / n)))
        original_risk, permuted_risk = _risk(c['active']), _risk(p['active'])
        sampled = {k: [] for k in ('sharpe_annual', 'psr0', 'psr1')}
        for wb in w:
            original_psr = psr_summary(np.repeat(c['active'], wb), 0)
            permuted_psr = psr_summary(np.repeat(p['active'], wb), 0)
            sampled['sharpe_annual'].append(original_psr['sharpe_annual'] - permuted_psr['sharpe_annual'])
            sampled['psr0'].append(original_psr['psr'] - permuted_psr['psr'])
            # Reuse the sampled moments for the annual Sharpe-one benchmark.
            def psr_one(v):
                if not v['valid']:
                    return np.nan
                return ndtr((v['sharpe_daily'] - 1 / np.sqrt(252)) * np.sqrt(n - 1) / np.sqrt(v['variance_factor']))
            sampled['psr1'].append(psr_one(original_psr) - psr_one(permuted_psr))
        for metric in sampled:
            control.append(dict(seed=seed, metric=metric, original=original_risk[metric],
                permuted=permuted_risk[metric], difference=original_risk[metric] - permuted_risk[metric],
                ci=_ci(sampled[metric])))
        for metric in ('max_drawdown', 'cvar5_loss', 'largest_positive_day_share'):
            original_value, permuted_value = original_risk[metric], permuted_risk[metric]
            # Concentration is undefined when a portfolio has no positive days.
            # Preserve that absence instead of treating it as zero concentration.
            difference = (original_value - permuted_value
                          if original_value is not None and permuted_value is not None else None)
            control.append(dict(seed=seed, metric=metric, original=original_risk[metric],
                permuted=permuted_risk[metric], difference=difference, ci=None))
    methods = dict(discovery=dict(definition='Positive mean training and held-out traded-day P&L; denominator includes records with nonempty held-out outcomes and finite training means.',
        interval='Independent two-sample strategy-record percentile bootstrap', replicates=2000,
        random_seed_reset_per_contrast=0, quantiles=[.025, .975], units='percentage points',
        assumption='Strategy records are treated as independent; intervals are conditional on the recorded evaluation period.'),
        portfolio=dict(series='Actual executed daily P&L, with zero on nontrading dates.',
            active='Sum of executed strategy P&L divided by the number of active strategies that day; zero when none trade.',
            fixed='Sum of executed strategy P&L divided by all front members in that fold.',
            interpretation='Fold-stitched held-out summaries. Active-member averaging is retrospective, not a specified executable capital allocation.',
            interval='Paired circular moving-block percentile bootstrap', replicates=replicates,
            block_days=block_length, seed=bootstrap_seed, random_seed=bootstrap_seed + block_length * 1009),
        risk=dict(units='Native proxy P&L units; not dollar account returns or drawdown percentages.',
            cvar='Nonnegative loss magnitude from the mean of the worst ceil(5% × calendar days) daily outcomes.',
            max_drawdown='Largest peak-to-trough decline of cumulative arithmetic P&L, including initial zero.',
            psr='IID/asymptotic plug-in confidence statistic using sample standard deviation, skewness and Pearson kurtosis; annual Sharpe benchmarks 0 and 1.',
            quality='At least 20 distinct held-out trading days and PSR at least 0.90.',
            pbo='Conditional surviving-front CSCV: eight ordered date blocks, all 70 half/half splits; flat and duplicate return columns excluded. Full adaptive-search PBO is not reconstructed.'),
        baseline='Same-structure always-enter daily P&L recovered as actual strategy P&L minus saved excess, on the recorded union calendar.',
        negative_control='Original and permuted embeddings share calendar draws. Its discovery interval uses paired day resampling, distinct from the main strategy-record bootstrap.')
    return dict(schema_version=1, title='Genetic programming search results', source=source, methods=methods,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        calculation_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path(__file__).with_name('archive.py'), Path(__file__).with_name('metrics.py'))},
        summary=dict(conditions=[LABELS[c] for c in CONDITIONS if any(k[0] == c for k in keys)],
            seeds=sorted({k[1] for k in keys}), folds=len(folds), front_members=len(member_rows),
            calendar_days=len(all_days), date_start=all_days[0], date_end=all_days[-1]),
        discovery=discovery, discovery_contrasts=contrasts, portfolios=portfolios, negative_control=control,
        fold_metrics=fold_rows, members=member_rows,
        daily_portfolios=[dict(condition=c, seed=s, dates=v['dates'], active_pnl=v['active'], fixed_pnl=v['fixed'])
                          for (c, s), v in series.items()])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategies', type=Path, help='Directory of individual archived strategies and manifest.json.')
    parser.add_argument('--outcomes', type=Path, help='Corresponding numeric outcome NPZ.')
    parser.add_argument('--runs', type=Path, help='Folder of new per-fold search JSONs, or one fold JSON.')
    parser.add_argument('--out', type=Path, required=True, help='Destination metrics JSON.')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing metrics output.')
    parser.add_argument('--bootstrap', type=int, default=10000, help='Portfolio/control block-bootstrap draws (default: 10000).')
    parser.add_argument('--block-days', type=int, default=5)
    parser.add_argument('--bootstrap-seed', type=int, default=20260917)
    args = parser.parse_args(argv)
    if args.bootstrap < 100 or args.block_days < 1:
        parser.error('Use at least 100 bootstrap draws and a positive block length.')
    if args.runs:
        if args.strategies or args.outcomes:
            parser.error('Use --runs OR the pair --strategies/--outcomes.')
    elif not (args.strategies and args.outcomes):
        parser.error('Supply --runs, or both --strategies and --outcomes.')
    destination = args.out.resolve()
    if args.strategies and destination.is_relative_to(args.strategies.resolve()):
        parser.error('Save metrics outside the strategies directory to preserve the archive.')
    if any(p is not None and p.resolve() == destination for p in (args.strategies, args.outcomes, args.runs)):
        parser.error('The output must not overwrite an input artifact.')
    if args.runs and args.runs.is_dir() and args.out.exists():
        try:
            existing = json.loads(args.out.read_text())
        except (OSError, ValueError):
            existing = {}
        if isinstance(existing, dict) and 'surviving_set' in existing:
            parser.error('The output must not overwrite a saved search fold.')
    if args.out.exists() and not args.overwrite:
        parser.error('The output already exists; choose a new path or add --overwrite.')
    try:
        folds, source = load_runs(args.runs) if args.runs else load_archive(args.strategies, args.outcomes)
        result = evaluate(folds, source, replicates=args.bootstrap, block_length=args.block_days,
                          bootstrap_seed=args.bootstrap_seed)
        write_json(args.out, result)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f'Metrics saved to {args.out}')


if __name__ == '__main__':
    main()
