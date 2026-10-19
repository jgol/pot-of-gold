"""Load individual archived strategies or new search results without replaying trades."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np

CONDITIONS = ('indicators', 'embeddings', 'combined', 'permuted_embeddings')
LABELS = dict(zip(CONDITIONS, ('Indicators', 'Embeddings', 'Combined', 'Permuted embeddings')))


@dataclass
class Fold:
    condition: str
    seed: int
    fold: int
    days: list[str]
    records: list[dict]
    pnl: np.ndarray                 # [day, strategy], with zeros on abstention days
    active: np.ndarray              # An executed zero-P&L day remains active.
    baseline: np.ndarray           # Same-structure always-enter reference; NaN if unavailable.
    train: list[np.ndarray]         # Archived training values do not have date keys.


def json_safe(value):
    """Use JSON null for unavailable results; never serialize NaN as valid data."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(record), indent=2, allow_nan=False) + '\n')


def _calendar(values):
    if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
        raise ValueError('A nonempty list of ISO test dates is required.')
    if values != sorted(set(values)):
        raise ValueError('Test dates must be sorted and unique.')
    if any(date.fromisoformat(v).isoformat() != v for v in values):
        raise ValueError('Test dates must use YYYY-MM-DD format.')
    return values


def _validate_folds(folds):
    seen = set()
    by_group = {}
    for f in folds:
        key = (f.condition, f.seed, f.fold)
        if key in seen:
            raise ValueError(f'Duplicate saved fold: {key}. Use one run per condition and seed.')
        seen.add(key)
        used = by_group.setdefault((f.condition, f.seed), set())
        if used.intersection(f.days):
            raise ValueError(f'Held-out dates overlap across folds for {f.condition}, seed {f.seed}.')
        used.update(f.days)
    if not folds:
        raise ValueError('No saved strategy fronts were found.')
    return sorted(folds, key=lambda f: (CONDITIONS.index(f.condition), f.seed, f.fold))


def _manifest(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError('Supply the strategies directory containing manifest.json.')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema_version') != 2:
        raise ValueError('Unsupported strategy archive schema.')
    _calendar(manifest['calendar'])
    paths = manifest['folds']
    if not isinstance(paths, list) or not paths or len(paths) != len(set(paths)):
        raise ValueError('Archive fold paths must be nonempty and unique.')
    return root, manifest


def _archive_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Archive path leaves its directory: {relative}')
    return path


def _read_fold(root, relative, calendar):
    path = _archive_path(root, relative)
    fold = json.loads(path.read_text())
    if fold.get('schema_version') != 2 or fold['condition'] not in CONDITIONS:
        raise ValueError(f'Unsupported fold metadata: {path}')
    expected = root / fold['condition'] / f"seed_{fold['seed']}" / f"fold_{fold['fold']}" / 'fold.json'
    if path != expected or fold['fold_id'] != f"{fold['condition']}-s{fold['seed']}-f{fold['fold']}":
        raise ValueError(f'Fold identity does not match its directory: {path}')
    days = _calendar(fold['test_days'])
    if not set(days).issubset(calendar) or fold['train_day_count'] != len(calendar) - len(days):
        raise ValueError(f'Fold dates do not match the archive calendar: {path}')
    names = fold['strategies']
    if not isinstance(names, list) or len(names) != len(set(names)):
        raise ValueError(f'Strategy filenames must be unique: {path}')
    actual = {p.name for p in path.parent.glob('*.json') if p.name != 'fold.json'}
    if set(names) != actual:
        raise ValueError(f'Strategy files do not match fold.json: {path.parent}')
    return path, fold


def _read_strategy(path, fold):
    record = json.loads(path.read_text())
    if record.get('schema_version') != 2:
        raise ValueError(f'Unsupported strategy schema: {path}')
    if any(record[key] != fold[key] for key in ('condition', 'seed', 'fold', 'fold_id')):
        raise ValueError(f'Strategy identity does not match its fold: {path}')
    if path.name != record['strategy_id'] + '.json' or path.name not in fold['strategies']:
        raise ValueError(f'Strategy ID does not match its filename: {path}')
    if record['role'] not in ('front', 'champion'):
        raise ValueError(f'Unknown strategy role: {path}')
    return record


def load_strategy(path):
    """Load one strategy and its shared fold context, without loading other genomes."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Strategy file does not exist: {path}')
    # strategies/<condition>/seed_N/fold_N/<strategy-id>.json
    if len(path.parents) < 4:
        raise ValueError('Keep the strategy in its condition/seed/fold archive directory.')
    root, manifest = _manifest(path.parents[3])
    relative = path.with_name('fold.json').relative_to(root).as_posix()
    if relative not in manifest['folds']:
        raise ValueError(f'Strategy fold is absent from manifest.json: {path}')
    _, fold = _read_fold(root, relative, manifest['calendar'])
    return _read_strategy(path, fold), fold, manifest['calendar']


def load_archive(strategies, outcomes):
    """Match individual strategy files to outcome rows by their preserved IDs."""
    root, manifest = _manifest(strategies)
    metadata, records_by_id = [], {}
    for relative in manifest['folds']:
        path, fold = _read_fold(root, relative, manifest['calendar'])
        metadata.append(fold)
        for filename in fold['strategies']:
            record = _read_strategy(_archive_path(path.parent, filename), fold)
            strategy_id = record['strategy_id']
            if strategy_id in records_by_id:
                raise ValueError(f'Duplicate strategy ID: {strategy_id}')
            records_by_id[strategy_id] = record
    if sorted({d for fold in metadata for d in fold['test_days']}) != manifest['calendar']:
        raise ValueError('Fold dates do not cover the archive calendar.')
    expected = manifest.get('outcomes_sha256')
    digest = hashlib.sha256(Path(outcomes).read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError('The outcomes archive does not match the strategy manifest checksum.')
    with np.load(outcomes, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    ids = arrays['strategy_ids'].tolist()
    if len(ids) != len(set(ids)) or set(ids) != set(records_by_id):
        raise ValueError('Strategy IDs are not unique or do not match the outcome rows.')
    records = [records_by_id[strategy_id] for strategy_id in ids]
    for role, key in (('front', 'front_count'), ('champion', 'extra_champion_count')):
        if sum(r['role'] == role for r in records) != manifest[key]:
            raise ValueError(f'Archive {key} does not match its strategy files.')
    count = len(records)
    pnl, active, baseline = (arrays[k] for k in ('oos_pnl', 'oos_active', 'baseline_pnl'))
    if pnl.ndim != 2 or pnl.shape[0] != count or active.shape != pnl.shape or baseline.shape != pnl.shape:
        raise ValueError('Invalid outcome array dimensions.')
    if active.dtype != np.bool_ or not np.isfinite(pnl).all() or not np.isfinite(baseline).all():
        raise ValueError('Invalid execution mask or nonfinite archived outcomes.')
    offsets, train = arrays['train_offsets'], arrays['train_values']
    if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(train) or np.any(np.diff(offsets) < 0):
        raise ValueError('Invalid training outcome offsets.')
    if not np.isfinite(train).all():
        raise ValueError('Nonfinite training outcomes.')
    folds = []
    for meta in metadata:
        days = _calendar(meta['test_days'])
        indices = [i for i, r in enumerate(records) if r['fold_id'] == meta['fold_id'] and r['role'] == 'front']
        if not indices:
            continue
        if np.any(arrays['n_days'][indices] != len(days)):
            raise ValueError(f"Outcome/calendar length mismatch for {meta['fold_id']}.")
        n = len(days)
        if np.any(pnl[indices, :n][~active[indices, :n]] != 0):
            raise ValueError('Nonzero P&L on an inactive date.')
        folds.append(Fold(meta['condition'], int(meta['seed']), int(meta['fold']), days,
                          [records[i] for i in indices], pnl[indices, :n].T,
                          active[indices, :n].T, baseline[indices, :n].T,
                          [train[offsets[i]:offsets[i+1]] for i in indices]))
    return _validate_folds(folds), dict(kind='archived experiment', strategy_count=len(records),
        front_count=sum(r['role'] == 'front' for r in records),
        extra_champion_count=sum(r['role'] == 'champion' for r in records),
        outcomes_sha256=digest, source_artifacts=manifest.get('source_artifacts', []),
        limitations=manifest.get('limitations', []))


def _runtime_member(record, days):
    index = {day: i for i, day in enumerate(days)}
    dates = record.get('held_out_traded_days')
    values = record.get('held_out_traded_day_pnl')
    if not isinstance(dates, list) or not isinstance(values, list) or len(dates) != len(values):
        raise ValueError('Missing or inconsistent dated executed outcomes.')
    if len(dates) != len(set(dates)) or not set(dates).issubset(index):
        raise ValueError('Repeated or out-of-fold executed dates.')
    trades = record.get('n_champion_trades')
    if isinstance(trades, bool) or not isinstance(trades, int) or trades < len(dates):
        raise ValueError('Missing or invalid trade count.')
    if not dates and trades != 0:
        raise ValueError('Empty outcomes do not establish a zero-trade strategy.')
    pnl = np.zeros(len(days)); active = np.zeros(len(days), dtype=bool)
    for day, value in zip(dates, values):
        pnl[index[day]] = value; active[index[day]] = True
    train = np.asarray(record.get('train_traded_day_pnl', []), dtype=np.float64)
    if train.ndim != 1 or not np.isfinite(train).all() or not np.isfinite(pnl).all():
        raise ValueError('Training and executed outcomes must be finite numeric arrays.')
    baseline = np.full(len(days), np.nan)
    if 'held_out_days' in record and 'held_out_day_excess' in record:
        union = record['held_out_days']; excess = record['held_out_day_excess']
        if len(union) != len(excess) or len(union) != len(set(union)) or not set(union).issubset(index) or not set(dates).issubset(union):
            raise ValueError('Invalid baseline union calendar.')
        baseline[:] = 0
        for day, value in zip(union, excess):
            baseline[index[day]] = pnl[index[day]] - value
        if not np.isfinite(baseline).all():
            raise ValueError('Nonfinite matched baseline outcomes.')
    return pnl, active, baseline, train


def load_runs(root):
    """Read new per-fold JSON outputs, ignoring unrelated configuration JSONs."""
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob('*.json'))
    folds = []; sources = []; synthetic_flags = []
    for path in paths:
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict) or 'surviving_set' not in doc or 'test_day_list' not in doc:
            continue
        prov = doc.get('provenance', {})
        synthetic = doc.get('synthetic_data')
        if synthetic is not None and type(synthetic) is not bool:
            raise ValueError(f'{path.name}: synthetic_data must be true, false, or null.')
        synthetic_flags.append(synthetic)
        condition = doc.get('condition', prov.get('condition'))
        if condition not in CONDITIONS:
            raise ValueError(f'{path.name}: unknown condition {condition!r}.')
        seed = doc.get('seed', prov.get('evolution_seed'))
        if seed is None:
            raise ValueError(f'{path.name}: missing evolution seed.')
        fold = int(doc['fold']); days = _calendar(doc['test_day_list']); rows = doc['surviving_set']
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'{path.name}: no saved front members; no portfolio can be evaluated.')
        data = [_runtime_member(r, days) for r in rows]
        records = [dict(strategy_id=r.get('strategy_id', f'{condition}-s{seed}-f{fold}-m{i:03d}'),
                        genome=r['genome'], trade_count=r['n_champion_trades']) for i, r in enumerate(rows)]
        folds.append(Fold(condition, int(seed), fold, days, records,
                          np.column_stack([v[0] for v in data]), np.column_stack([v[1] for v in data]),
                          np.column_stack([v[2] for v in data]), [v[3] for v in data]))
        sources.append(dict(condition=condition, seed=int(seed), fold=fold,
                            synthetic_data=synthetic, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    folds = _validate_folds(folds)
    all_synthetic = all(flag is True for flag in synthetic_flags)
    any_synthetic = any(flag is True for flag in synthetic_flags)
    all_real = all(flag is False for flag in synthetic_flags)
    kind = 'synthetic demonstration' if all_synthetic else ('mixed data sources' if any_synthetic else 'new search results')
    return folds, dict(kind=kind, synthetic_data=True if all_synthetic else (False if all_real else None),
        contains_synthetic_data=any_synthetic, source_artifacts=sources,
        front_count=sum(len(f.records) for f in folds), extra_champion_count=0)
