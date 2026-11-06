"""Load saved H3 decisions and outcomes, checking identities, hashes and dates."""
import hashlib
import gzip
import json
from pathlib import Path
from statistics import median

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def keyed(rows):
    result = {r['champion_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate champion IDs')
    return result


def load_new_inputs(outcomes, court_decisions, single_decisions):
    """Join new-run decisions to dated outcomes using the released NPZ layout."""
    documents = [read_json(p) for p in (court_decisions, single_decisions)]
    if any(doc.get('dry_run') for doc in documents):
        raise ValueError('Mock decisions cannot be used for a new-data evaluation')
    court, single = [keyed(doc['verdicts']) for doc in documents]
    with np.load(outcomes, allow_pickle=False) as saved:
        ids = saved['champion_ids'].tolist()
        dates = saved['dates'].tolist()
        pnl, mask = saved['held_out_pnl'].copy(), saved['held_out_mask'].copy()
        train, offsets = saved['train_pnl'].copy(), saved['train_offsets'].copy()
    if (not ids or len(ids) != len(set(ids)) or not all(isinstance(cid, str) for cid in ids)
            or set(ids) != set(court) or set(ids) != set(single)):
        raise ValueError('Outcomes and both decision files must cover the same unique strategy IDs')
    from datetime import date
    if (not dates or dates != sorted(set(dates))
            or any(not isinstance(d, str) or date.fromisoformat(d).isoformat() != d for d in dates)):
        raise ValueError('Outcome dates must be unique sorted ISO dates')
    if (pnl.shape != (len(dates), len(ids)) or mask.shape != pnl.shape or mask.dtype != bool
            or not np.isfinite(pnl).all() or np.any(pnl[~mask] != 0)):
        raise ValueError('Invalid held-out P&L or fold mask')
    if (train.ndim != 1 or not np.isfinite(train).all() or offsets.shape != (len(ids) + 1,)
            or offsets.dtype.kind not in 'iu' or offsets[0] != 0 or offsets[-1] != len(train)
            or np.any(np.diff(offsets) <= 0)):
        raise ValueError('Invalid training P&L or offsets')
    selected, keep, excluded = {}, [], []
    for i, cid in enumerate(ids):
        if (court[cid].get('disagreement') or {}).get('degraded'):
            excluded.append(cid)
            continue
        keep.append(i)
        selected[cid] = {'train_net': sum(train[offsets[i]:offsets[i + 1]].tolist()),
                         'court': court[cid], 'single': single[cid]}
    if not selected:
        raise ValueError('No strategies have a complete court assessment')
    paths = [Path(p) for p in (outcomes, court_decisions, single_decisions)]
    meta = {'mode': 'new_data', 'n_champions': len(keep), 'n_days': len(dates),
            'start': dates[0], 'end': dates[-1], 'excluded': excluded, 'joins': [],
            'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'source_sha256': {}, 'limitations': [
                'New user-supplied outcomes and decisions; these are not the published H3 sample.',
                'The caller is responsible for keeping scored outcomes out of the assessment inputs.']}
    return list(selected), selected, pnl[:, keep], meta


def load_inputs(archive):
    root = Path(archive).resolve()
    manifest = read_json(root / 'data/manifest.json')
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported archive schema')
    for relative, expected in manifest['archive_sha256'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Archive paths must stay within the module directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Archive checksum mismatch: {relative}')

    strategies = read_json(root / 'data/cohort.json')['strategies']
    cohort = keyed(strategies)
    court = keyed(read_json(root / 'results/court.json')['verdicts'])
    single = keyed(read_json(root / 'results/single_llm.json')['verdicts'])
    ids = list(cohort)
    if set(ids) != set(court) or set(ids) != set(single):
        raise ValueError('Cohort and decision IDs differ')
    audits = read_transcripts(root / 'results/transcripts.json.gz')
    court_audits = keyed([r for r in audits if r['outcome'] == 'verdict'])
    single_audits = keyed([r for r in audits if r['outcome'] == 'single_llm_baseline'])
    if set(court_audits) != set(ids) or set(single_audits) != set(ids):
        raise ValueError('Transcript coverage differs from the cohort')
    for cid in ids:
        actual = court_audits[cid]
        scores = [j['risk_score'] for j in actual['panel']]
        recorded = court[cid]['disagreement']['exploratory_ranking']['median_risk_score']
        if not scores or median(scores) != recorded or actual['decision'] != court[cid]['decision']:
            raise ValueError(f'Court decision differs from its archived panel: {cid}')
        baseline = single_audits[cid]
        if baseline['risk_score'] != single[cid]['risk_score']:
            raise ValueError(f'Single-LLM decision differs from its archived assessment: {cid}')

    with np.load(root / 'results/outcomes.npz', allow_pickle=False) as saved:
        if saved['champion_ids'].tolist() != ids:
            raise ValueError('Outcome rows do not match cohort order')
        dates = saved['dates'].tolist()
        pnl = saved['held_out_pnl'].copy()
        mask = saved['held_out_mask'].copy()
        train = saved['train_pnl'].copy()
        offsets = saved['train_offsets'].copy()
    if len(ids) != manifest['n_source_strategies'] or len(dates) != manifest['n_days']:
        raise ValueError('Archive counts differ from the manifest')
    if dates != sorted(set(dates)) or (dates[0], dates[-1]) != (manifest['start'], manifest['end']):
        raise ValueError('Invalid held-out calendar')
    if pnl.shape != (len(dates), len(ids)) or mask.shape != pnl.shape or mask.dtype != bool:
        raise ValueError('Invalid outcome or fold-mask dimensions')
    if not np.isfinite(pnl).all() or np.any(pnl[~mask] != 0):
        raise ValueError('Invalid P&L or values outside a strategy\'s held-out fold')
    if (offsets.shape != (len(ids) + 1,) or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or offsets[-1] != len(train) or np.any(np.diff(offsets) <= 0)
            or train.ndim != 1 or not np.isfinite(train).all()):
        raise ValueError('Invalid training P&L offsets or values')

    keep, excluded, selected = [], [], {}
    for i, cid in enumerate(ids):
        record = cohort[cid]
        degraded = bool((court[cid].get('disagreement') or {}).get('degraded'))
        if degraded != record['excluded']:
            raise ValueError(f'Exclusion differs from the saved panel status: {cid}')
        if degraded:
            excluded.append(cid)
            continue
        keep.append(i)
        # Preserve summation order because rounding can affect ties in the ranking.
        train_net = sum(train[offsets[i]:offsets[i + 1]].tolist())
        selected[cid] = {**record, 'train_net': train_net, 'court': court[cid], 'single': single[cid]}
    if len(keep) != manifest['n_evaluated']:
        raise ValueError('Evaluated cohort count differs from the manifest')
    meta = {'n_champions': len(keep), 'n_days': len(dates), 'start': dates[0], 'end': dates[-1],
            'excluded': excluded, 'joins': manifest['batches'],
            'input_sha256': manifest['archive_sha256'], 'source_sha256': manifest['source_sha256'],
            'limitations': manifest['limitations']}
    return list(selected), selected, pnl[:, keep], meta


def read_transcripts(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)['records']
