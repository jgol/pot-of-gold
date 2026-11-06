"""Recompute H3 portfolio levels and paired bootstrap comparisons."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from potofgold.multi_agent import runtime_source_paths
from potofgold.multi_agent.analysis.allocation import POLICIES, allocations
from .archive import load_inputs, load_new_inputs

def levels(books, tail_fraction=0.05):
    """Days × policies -> policies × (cumulative P&L, CVaR, maximum drawdown)."""
    k = max(1, math.ceil(tail_fraction * len(books)))
    equity = np.cumsum(books, axis=0)
    peak = np.maximum.accumulate(np.maximum(equity, 0), axis=0)
    return np.stack([books.sum(axis=0), np.sort(books, axis=0)[:k].mean(axis=0),
                     (peak - equity).max(axis=0)], axis=1)


def bootstrap(books, samples, seed, tail_fraction=0.05, block_length=10):
    # Apply each sampled date sequence to every allocation policy.
    """Resample paired policy returns using the published H3 conventions.

    Use independent days for cumulative P&L and CVaR, and circular blocks for
    drawdown. Each draw uses the same sampled dates across policies.
    """
    iid_rng, block_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    n, p = books.shape
    tail = max(1, math.ceil(tail_fraction * n))
    out = np.empty((samples, p, 3))
    for start in range(0, samples, 200):
        batch = min(200, samples - start)
        values = books[iid_rng.integers(0, n, (batch, n))]
        out[start:start + batch, :, 0] = values.sum(axis=1)
        out[start:start + batch, :, 1] = np.partition(values, tail - 1, axis=1)[:, :tail].mean(axis=1)
        starts = block_rng.integers(0, n, (batch, math.ceil(n / block_length)))
        order = ((starts[:, :, None] + np.arange(block_length)) % n).reshape(batch, -1)[:, :n]
        equity = np.cumsum(books[order], axis=1)
        peak = np.maximum.accumulate(np.maximum(equity, 0), axis=1)
        out[start:start + batch, :, 2] = (peak - equity).max(axis=1)
    return out


def comparisons(point, draws, court):
    """Compare court performance with each baseline, with positive differences favoring court.

    Scale each bootstrap difference by its own baseline magnitude. Omit draws
    with zero denominators from percentage intervals and report their counts.
    """
    result = {}
    for offset, label in enumerate(POLICIES[1:], 1):
        baseline = court + offset
        signed = np.array([1., 1., -1.])
        difference = (point[court] - point[baseline]) * signed
        sampled = (draws[:, court] - draws[:, baseline]) * signed
        if np.any(point[baseline] == 0):
            raise ValueError('Zero observed comparator denominator')
        percent = 100 * difference / np.abs(point[baseline])
        denominators = np.abs(draws[:, baseline])
        scaled = np.divide(100 * sampled, denominators,
                           out=np.full_like(sampled, np.nan), where=denominators > 0)
        ci = np.nanquantile(scaled, [0.025, 0.975], axis=0).T
        result[label] = {'difference': difference.tolist(), 'difference_ci95':
                        np.quantile(sampled, [0.025, 0.975], axis=0).T.tolist(),
                        'percent': percent.tolist(), 'percent_ci95': ci.tolist(),
                        'zero_denominator_draws': (denominators == 0).sum(axis=0).tolist(),
                        'meets_h3': bool(ci[1, 0] > 0 and ci[2, 0] > 0)}
    return result


def calculate(archive, config):
    return calculate_inputs(*load_inputs(archive), config)


def calculate_inputs(keys, cohort, pnl, meta, config):
    """Apply the same allocation and bootstrap calculations to an aligned sample."""
    weights = allocations(keys, cohort)
    books = pnl @ weights.T
    point = levels(books, config['tail_fraction'])
    draws = bootstrap(books, config['bootstrap_samples'], config['seed'],
                      config['tail_fraction'], config['block_length'])
    ci = np.quantile(draws, [0.025, 0.975], axis=0)
    levels_result = {
        p: {'pnl': float(point[i, 0]), 'cvar': float(point[i, 1]), 'maxdd': float(point[i, 2]),
            'pnl_ci95': ci[:, i, 0].tolist(), 'cvar_ci95': ci[:, i, 1].tolist(),
            'maxdd_ci95': ci[:, i, 2].tolist(), 'exposure': float(weights[i].mean()),
            'funded': int((weights[i] > 0).sum()), 'max_size': float(weights[i].max())}
        for i, p in enumerate(POLICIES)
    }
    return {
        'schema_version': 1, 'metadata': meta, 'samples': config['bootstrap_samples'],
        'seed': config['seed'], 'block_length': config['block_length'],
        'tail_fraction': config['tail_fraction'], 'confidence_level': config['confidence_level'],
        'code_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in runtime_source_paths().items()},
        'variants': {'score_only': {'levels': levels_result, 'comparisons': comparisons(point, draws, 0)}},
        'allocations': [{'champion_id': k, **{p: float(weights[j, i]) for j, p in enumerate(POLICIES)}}
                        for i, k in enumerate(keys)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('.'), help='Module directory containing the released archive')
    parser.add_argument('--config', type=Path, default=Path('configs/experiment.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--outcomes', type=Path, help='New dated outcomes in the documented NPZ layout')
    parser.add_argument('--court-decisions', type=Path)
    parser.add_argument('--single-decisions', type=Path)
    args = parser.parse_args(argv)
    new_paths = (args.outcomes, args.court_decisions, args.single_decisions)
    if any(new_paths) and not all(new_paths):
        parser.error('New-data evaluation requires --outcomes, --court-decisions and --single-decisions')
    config = yaml.safe_load(args.config.read_text())
    # Report labels and intervals assume these fixed tail and confidence levels.
    if config.get('tail_fraction') != 0.05 or config.get('confidence_level') != 0.95:
        parser.error('The published report uses CVaR at 5% and confidence intervals at 95%')
    for name, minimum in [('bootstrap_samples', 1000), ('seed', 0), ('block_length', 1)]:
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            parser.error(f'{name} must be an integer >= {minimum}')
    result = (calculate_inputs(*load_new_inputs(*new_paths), config) if all(new_paths)
              else calculate(args.archive, config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(f'Wrote {args.output}: {result["metadata"]["n_champions"]} strategies, '
          f'{result["metadata"]["n_days"]} held-out dates')


if __name__ == '__main__':
    main()
