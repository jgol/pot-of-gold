"""Measure saved embeddings or recalculate summaries from archived per-seed metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..encoder.checkpoints import load_checkpoint, training_metadata
from ..encoder.config import canonical_tokenizer
from .metrics import (ic, metric_function, participation_ratio, r2, rank_ic,
                      rankme, seed_interval)

ROOT = Path(__file__).resolve().parents[3]
GEOMETRY = ('rankme', 'participation_ratio')
PREDICTION = ('rank_ic', 'r2')
WINDOWS = ('train', 'val', 'test')


# This order matches the first four forward-target columns in the corpus.
TARGETS = ("volatility_difference", "volatility_scaled_return", "upward_excursion", "downward_excursion")
ALPHAS = np.logspace(-4, 6, 11)

def probe_skill(Z_tr, T_tr, Vt_tr, Z_va, T_va, Vt_va, rank_ic, ic):
    """Fit one ridge probe per target on train and score held-out predictions.

    Z has shape (windows, embedding dimensions); T and its boolean validity
    mask Vt have shape (windows, targets). Scaling uses all training vectors,
    while each probe uses only valid, finite training targets. Return one
    metrics dict per target; sign agreement is a fraction, not a percentage.
    R^2 compares squared error with the held-out target mean and can be negative.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    rows = []
    n_targets = T_tr.shape[1]
    sc = StandardScaler().fit(np.nan_to_num(Z_tr))
    Ztr, Zva = sc.transform(np.nan_to_num(Z_tr)), sc.transform(np.nan_to_num(Z_va))
    for j in range(n_targets):
        mtr = Vt_tr[:, j] & np.isfinite(T_tr[:, j])
        if mtr.sum() < 50:
            rows.append({"rank_ic": float("nan"), "ic": float("nan"), "sign": float("nan"), "r2": float("nan")}); continue
        m = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(Ztr[mtr], T_tr[mtr, j])
        pv = m.predict(Zva)
        mva = Vt_va[:, j] & np.isfinite(T_va[:, j]) & np.isfinite(pv)
        rows.append({
            "rank_ic": float(rank_ic(pv, T_va[:, j], Vt_va[:, j])),
            "ic": float(ic(pv, T_va[:, j], Vt_va[:, j])),
            "sign": float(np.mean((np.sign(pv[Vt_va[:, j]]) == np.sign(T_va[Vt_va[:, j], j]))))
                     if Vt_va[:, j].sum() else float("nan"),
            "r2": float(r2_score(T_va[mva, j], pv[mva])) if mva.sum() >= 5 else float("nan"),
        })
    return rows

def shuffle_diagnostic(Z_tr, T_tr, Vt_tr, Z_va, T_va, Vt_va, rank_ic, ic, n_shuffle, seed0=7):
    """Break embedding-to-target pairings and refit probes as a diagnostic.

    Train and held-out rows are shuffled separately; targets and validity
    masks stay in place. Return each target's 95th percentile of absolute
    rank-IC and signed R^2 over the shuffled fits. These reference values are
    not confidence intervals or evidence of predictive usefulness: row
    shuffling does not preserve the full temporal dependence.
    """
    n_targets = T_tr.shape[1]
    ric = np.full((max(n_shuffle, 1), n_targets), np.nan)
    r2 = np.full((max(n_shuffle, 1), n_targets), np.nan)
    for s in range(n_shuffle):
        rng = np.random.default_rng(seed0 + s)
        Ztr = Z_tr[rng.permutation(len(Z_tr))]
        Zva = Z_va[rng.permutation(len(Z_va))]
        rows = probe_skill(Ztr, T_tr, Vt_tr, Zva, T_va, Vt_va, rank_ic, ic)
        ric[s] = [r["rank_ic"] for r in rows]
        r2[s] = [r["r2"] for r in rows]
    return {"rank_ic": np.nanpercentile(np.abs(ric), 95, axis=0).tolist(),
            "r2": np.nanpercentile(r2, 95, axis=0).tolist()}

def ckpt_zstats(ck):
    """Extract saved input means, scales, and preprocessing settings, if present."""
    if ck.get("z_mean") is None:
        return None
    return {"mean": ck["z_mean"], "std": ck["z_std"],
            "log1p_cols": ck.get("log1p_cols"), "min_std_floor": ck.get("min_std_floor", 0.01)}

def prep_eval_data_tvt(data_path, embargo_days=5, ckpt_zstats=None):
    """Load the corpus once and prepare train, validation, and test arrays.

    X_* arrays have shape (windows, time steps, features); T_* and boolean
    Vt_* arrays have shape (windows, four targets). Inputs and targets use
    training-fitted preprocessing. Both split calls fit the same training
    rows; only the held-out date mask changes, so retain train from the first.
    """
    from ..encoder import config as cfg
    from .forward_targets import load_evaluation_days
    from .target_processing import make_evaluation_splits
    settings = cfg.EXPERIMENT_PROFILE
    dov = {k: settings[k] for k in ("TRAIN_START", "TRAIN_END", "VAL_START", "VAL_END", "TEST_START", "TEST_END")}
    log1p_idx = tuple(settings["preprocess_log1p_indices"])
    ssl_data = load_evaluation_days(data_path, settings["TRAIN_START"], settings["TEST_END"], log1p_indices=log1p_idx)
    out = {"ssl_data": ssl_data}
    for win in ("val", "test"):
        splits = make_evaluation_splits(ssl_data, embargo_days=embargo_days,
                                       eval_window=win, date_overrides=dov, z_stats=ckpt_zstats)
        if "X_train" not in out:
            out["X_train"], out["T_train"], out["Vt_train"] = splits["train"]
        out[f"X_{win}"], out[f"T_{win}"], out[f"Vt_{win}"] = splits["val"]
    return out

def probe_multi(Z_tr, y_tr, v_tr, Z_by_window):
    """Fit one target's probe on train and reuse it for every named window.

    Z_by_window maps split names to (windows, dimensions) vectors; the result
    maps those names to prediction vectors. Scale all training vectors and
    fit only valid, finite training targets. Fewer than 50 usable targets
    returns NaNs. A training-window prediction is an in-sample diagnostic.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.nan_to_num(Z_tr))
    m = v_tr & np.isfinite(y_tr)
    if int(m.sum()) < 50:
        return {w: np.full(Z.shape[0], np.nan) for w, Z in Z_by_window.items()}
    est = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(sc.transform(np.nan_to_num(Z_tr))[m], y_tr[m])
    return {w: est.predict(sc.transform(np.nan_to_num(Z))) if len(Z) else np.empty(0)
            for w, Z in Z_by_window.items()}

def fit_predict(X_train, y_train, valid, dates, X_test):
    """Tune original-target ridge models on chronological training-day folds.

    X arrays are (windows, features); y_train and valid are (windows, targets),
    with one date per training window. Five expanding folds keep whole days
    together and fit scaling on each fold's training rows. Choose a penalty
    per target, then refit on all valid training rows and predict X_test.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    if not np.all(valid == valid[:, :1]):
        raise ValueError("Forward targets must share a validity mask")
    days = np.unique(dates)
    loss = np.zeros((len(ALPHAS), y_train.shape[1]))
    count = 0
    folds = []
    for train, validation in TimeSeriesSplit(n_splits=5).split(days):
        tm = np.isin(dates, days[train]) & valid[:, 0]
        vm = np.isin(dates, days[validation]) & valid[:, 0]
        if not tm.any() or not vm.any():
            raise ValueError("A training or validation fold has no valid targets")
        assert max(dates[tm]) < min(dates[vm])
        scaler = StandardScaler().fit(X_train[tm])
        xt, xv = scaler.transform(X_train[tm]), scaler.transform(X_train[vm])
        for i, alpha in enumerate(ALPHAS):
            model = Ridge(alpha=alpha, solver="cholesky").fit(xt, y_train[tm])
            loss[i] += ((model.predict(xv) - y_train[vm]) ** 2).sum(axis=0)
        count += int(vm.sum())
        folds.append({"train_last": max(dates[tm]),
                      "validation_first": min(dates[vm]),
                      "validation_last": max(dates[vm])})
    # Summed errors weight validation windows equally, not folds equally.
    penalties = ALPHAS[loss.argmin(axis=0)]
    tm = valid[:, 0]
    scaler = StandardScaler().fit(X_train[tm])
    model = Ridge(alpha=penalties, solver="cholesky").fit(
        scaler.transform(X_train[tm]), y_train[tm])
    predictions = model.predict(scaler.transform(X_test))
    return predictions, {"alpha": penalties.tolist(), "cv_mse": (loss / count).tolist(),
                         "prediction_std": predictions.std(axis=0).tolist()}, folds

def indicator_bootstrap(predictions, targets, valid, dates, n_seeds, repeats, seed, *, groups=None):
    """Bootstrap frozen original-target predictions over days and encoder seeds.

    predictions is (fits, windows, targets); targets and valid are (windows,
    targets). Fit zero is the indicator baseline. Each group lists fits for
    one tokenizer; the default groups assume three tokenizers with n_seeds
    fits each. Metrics are ordered as rank-IC, R^2.

    Every replicate draws the same whole days for all fits, then independently
    resamples fits within each tokenizer and averages their metrics. A second
    calculation shares seed positions across equal-sized groups; callers must
    check that those positions identify the same seeds. Requires repeats > 0.
    """
    days, ids = np.unique(dates, return_inverse=True)
    metrics = metric_function(predictions, targets, valid, ids)
    per_run = metrics(np.ones(len(days)))
    if groups is None:
        groups = [list(range(1 + a * n_seeds, 1 + (a + 1) * n_seeds)) for a in range(3)]
    if not groups or any(not group for group in groups):
        raise ValueError('Each tokenizer needs at least one checkpoint')
    point = np.concatenate([per_run[:1], np.stack([per_run[group].mean(axis=0) for group in groups])])
    # Axes: replicate, baseline/tokenizer, target, metric. Average fit scores,
    # not predictions, so this measures typical fitted models, not an ensemble.
    samples = np.empty((repeats, len(groups) + 1, targets.shape[1], 2))
    paired = np.empty_like(samples)
    rng = np.random.default_rng(seed)
    for b in range(repeats):
        # Day counts replicate all of a day's windows together without copying
        # prediction arrays. Encoders and ridge models are never refitted here.
        counts = np.bincount(rng.integers(len(days), size=len(days)), minlength=len(days))
        met = metrics(counts)
        samples[b, 0] = paired[b, 0] = met[0]
        equal_sizes = len({len(group) for group in groups}) == 1
        if equal_sizes:
            count = len(groups[0])
            tokenizer_metrics = np.stack([met[group] for group in groups])
            picks = rng.integers(count, size=(len(groups), count))
            samples[b, 1:] = np.stack([tokenizer_metrics[a, picks[a]].mean(axis=0) for a in range(len(groups))])
            # Shared positions make this a paired-seed sensitivity calculation.
            shared = rng.integers(count, size=count)
            paired[b, 1:] = tokenizer_metrics[:, shared].mean(axis=1)
        else:
            samples[b, 1:] = np.stack([met[np.asarray(group)[rng.integers(len(group), size=len(group))]].mean(axis=0)
                                      for group in groups])
            paired[b] = samples[b]
        if (b + 1) % 2500 == 0:
            print(f"Bootstrap: {b + 1}/{repeats}", flush=True)

    def summarize(values, quantiles):
        def ci(x):
            # Put lower/upper bounds on the last axis of the serialized arrays.
            return np.moveaxis(np.quantile(x, quantiles, axis=0), 0, -1).tolist()
        result = {"metric_ci": ci(values), "comparisons": {}}
        for a in range(1, len(groups) + 1):
            for baseline in range(a):
                x, ref = values[:, a], values[:, baseline]
                # R^2 shares the same resampled target variance across fits, so
                # (R^2 - ref)/(1 - ref) is the proportional error reduction.
                result["comparisons"][f"{a}_minus_{baseline}"] = {
                    "delta_ci": ci(x - ref),
                    "rank_ic_5pct_margin_ci": ci(x[:, :, 0] - 1.05 * ref[:, :, 0]),
                    "mse_reduction_pct_ci": ci(100 * (x[:, :, 1] - ref[:, :, 1]) / (1 - ref[:, :, 1])),
                }
        return result

    if not np.isfinite(samples).all() or not np.isfinite(paired).all():
        raise ValueError("Non-finite bootstrap metrics")
    # Central percentile bounds apply to each target/metric separately;
    # they do not adjust for the number of comparisons.
    return {"point": point.tolist(), "per_run": per_run.tolist(),
            "95": summarize(samples, [.025, .975]),
            "90": summarize(samples, [.05, .95]),
            "paired_seed_95": summarize(paired, [.025, .975]) if equal_sizes else None}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_embeddings(path):
    """Read finite exported vectors with unique (date, last_bar) row identities.

    Pooled exports are (windows, dimensions); token exports add a middle
    variate axis. Return one float64 vector per window for scoring, plus the
    metadata and row keys needed to align exports and targets.
    """
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(bytes(data['metadata']).decode('utf-8'))
        metadata['tokenizer'] = canonical_tokenizer(metadata.get('tokenizer'))
        z = np.asarray(data['embeddings'])
        dates = np.asarray(data['dates']).astype(str)
        bars = np.asarray(data['last_bar'])
    if z.ndim not in (2, 3) or not len(z) or dates.shape != (len(z),) or bars.shape != dates.shape:
        raise ValueError(f'Invalid embedding shapes: {path}')
    if not np.issubdtype(bars.dtype, np.integer) or not np.isfinite(z).all():
        raise ValueError(f'Embedding vectors must be finite and last_bar must contain integers: {path}')
    keys = list(zip(dates.tolist(), bars.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError(f'Duplicate (date, last_bar) rows in {path}')
    if metadata.get('readout') not in ('pooled', 'variates', 'multilayer'):
        raise ValueError(f'Export has no supported readout metadata: {path}')
    if (metadata['readout'] == 'pooled') != (z.ndim == 2):
        raise ValueError(f'Readout metadata does not match embedding dimensions: {path}')
    # Average variates for both token readouts. Multilayer exports retain their
    # concatenated layer width, so their geometry uses a wider feature space.
    pooled = z if z.ndim == 2 else z.mean(axis=1)
    return dict(path=Path(path), metadata=metadata, z=np.asarray(pooled, np.float64),
                dates=dates, bars=bars, keys=keys)


def split_masks(dates, split_dates, embargo_days=5):
    """Build ordered split masks and apply an embargo measured from train's end.

    The gap counts observed trading dates, not calendar days. Both held-out
    masks use the same training boundary; test has no extra gap after val.
    """
    from ..encoder.forward_split import apply_embargo
    required = [f'{window}_{edge}' for window in WINDOWS for edge in ('start', 'end')]
    if any(key not in split_dates for key in required):
        raise ValueError('Embedding metadata must provide all six split_dates boundaries; re-export embeddings')
    if not (split_dates['train_start'] <= split_dates['train_end'] < split_dates['val_start']
            <= split_dates['val_end'] < split_dates['test_start'] <= split_dates['test_end']):
        raise ValueError('Training, validation and test dates must be ordered and disjoint')
    masks = {window: (dates >= split_dates[f'{window}_start']) & (dates <= split_dates[f'{window}_end'])
             for window in WINDOWS}
    for window in ('val', 'test'):
        masks[window] = apply_embargo(dates, masks['train'], masks[window], embargo_days)
    return masks


def _align_exports(exports):
    """Require compatible exports and reorder their rows to match the first."""
    first = exports[0]
    for export in exports[1:]:
        for key in ('corpus_sha256', 'profile', 'stride', 'seq_len', 'split_dates', 'readout'):
            if export['metadata'].get(key) != first['metadata'].get(key):
                raise ValueError(f'Exports disagree on {key}; evaluate compatible exports together')
        if set(export['keys']) != set(first['keys']):
            raise ValueError('Exports do not contain the same (date, last_bar) rows')
        # Row identity, not file order, determines which vectors are compared.
        lookup = {key: index for index, key in enumerate(export['keys'])}
        order = [lookup[key] for key in first['keys']]
        for key in ('z', 'dates', 'bars'):
            export[key] = export[key][order]
        export['keys'] = first['keys']


def _estimate(value, samples=None):
    """Attach 95% percentile bounds when at least two bootstrap values are finite."""
    ci = None
    if samples is not None:
        finite = np.asarray(samples)[np.isfinite(samples)]
        if len(finite) >= 2:
            ci = np.percentile(finite, [2.5, 97.5]).tolist()
    return {'value': float(value), 'ci': ci}


def _geometry(z):
    if len(z) < 2:
        return {metric: _estimate(float('nan')) for metric in GEOMETRY}
    return {'rankme': _estimate(rankme(z)), 'participation_ratio': _estimate(participation_ratio(z))}


def _prediction_metrics(predictions, targets, valid):
    return {target: {'rank_ic': _estimate(rank_ic(predictions[:, j], targets[:, j], valid[:, j])),
                     'r2': _estimate(r2(predictions[:, j], targets[:, j], valid[:, j]))}
            for j, target in enumerate(TARGETS)}


def _row_keys(dates, seq_len=60, stride=30):
    """Rebuild window anchors in corpus order, restarting the stride each day."""
    counts = {}
    keys = []
    for date in dates:
        index = counts.get(date, 0)
        keys.append((str(date), seq_len - 1 + index * stride))
        counts[date] = index + 1
    return keys


def _full_data(exports, data_path, checkpoint_paths, embargo_days):
    """Verify corpus/checkpoint provenance and align vectors with prepared targets.

    Prediction scoring reconstructs the recorded experiment's window layout.
    Export hashes must match the supplied files, and all checkpoints must
    share input normalization so target residualization is identical.
    Reorder exports in place to the complete corpus's window order.
    """
    from ..encoder.config import ACTIVE_SSL_FEATURES, EXPERIMENT_PROFILE
    first = exports[0]['metadata']
    if first.get('readout') != 'pooled' or first.get('stride') != 30 or first.get('seq_len') != 60:
        raise ValueError('Prediction evaluation requires pooled embeddings with stride 30 and sequence length 60; re-export that readout')
    if first.get('profile') != 'experiment':
        raise ValueError('Prediction evaluation requires exports using the recorded experiment preprocessing')
    expected_splits = {key.lower(): EXPERIMENT_PROFILE[key] for key in EXPERIMENT_PROFILE if key.endswith(('_START', '_END'))}
    if first.get('split_dates') != expected_splits:
        raise ValueError('Prediction export split dates differ from the recorded experiment')
    digest = sha256(data_path)
    if any(export['metadata'].get('corpus_sha256') != digest for export in exports):
        raise ValueError('Corpus SHA-256 does not match the exported embeddings')
    # Hashes bind exports to exact files, independently of paths or filenames.
    checkpoints = {}
    for path in checkpoint_paths:
        checkpoint_hash = sha256(path)
        if checkpoint_hash in checkpoints:
            raise ValueError('Duplicate checkpoint supplied')
        checkpoints[checkpoint_hash] = (Path(path), load_checkpoint(path))
    required = {export['metadata'].get('checkpoint_sha256') for export in exports}
    if required != set(checkpoints):
        raise ValueError('Provide exactly the checkpoints matching the embedding checkpoint SHA-256 values')
    normalization = None
    for export in exports:
        path, checkpoint = checkpoints[export['metadata']['checkpoint_sha256']]
        if checkpoint['ssl_features'] != ACTIVE_SSL_FEATURES:
            raise ValueError('Prediction evaluation requires the original 147-feature layout')
        if checkpoint['ssl_hyperparams']['d_model'] != export['z'].shape[1]:
            raise ValueError('Embedding width differs from its checkpoint')
        try:
            saved = training_metadata(path)
        except ValueError:
            saved = {}
        for key in ('tokenizer', 'seed'):
            value = saved.get(key, checkpoint.get(key))
            if value is not None and str(value) != str(export['metadata'].get(key)):
                raise ValueError(f'Embedding {key} does not match its checkpoint')
        export['training'] = saved
        # Residualizer scalars come from normalized input windows. Different
        # normalization could make the runs predict different residual targets.
        current = ckpt_zstats(checkpoint)
        if current is None:
            raise ValueError('Checkpoint must contain fitted normalization')
        if normalization is None:
            normalization = current
        elif any((normalization[key] is not current[key]) if normalization[key] is None or current[key] is None
                 else not np.array_equal(normalization[key], current[key], equal_nan=True)
                 for key in normalization):
            raise ValueError('Compared checkpoints must share normalization so residual targets are identical')
    prepared = prep_eval_data_tvt(str(data_path), embargo_days=embargo_days,
                                 ckpt_zstats=normalization)
    # Matching complete row sets lets split masks select the same ordered rows
    # from embeddings and prepared targets, even if exports were saved shuffled.
    raw_keys = _row_keys(prepared['ssl_data']['window_dates'])
    if set(raw_keys) != set(exports[0]['keys']):
        raise ValueError('Exported (date, last_bar) rows do not match corpus windows; export the complete corpus')
    order = {key: index for index, key in enumerate(exports[0]['keys'])}
    order = [order[key] for key in raw_keys]
    for export in exports:
        for key in ('z', 'dates', 'bars'):
            export[key] = export[key][order]
        export['keys'] = raw_keys
    return prepared


def _day_bootstrap(exports, masks, prepared, predictions, window, repeats, seed):
    """Resample complete evaluation days with selected encoders and probes fixed.

    Reuse each day's draw across runs so differences are paired by day. This
    varies evaluation data only: no model refitting or encoder-seed resampling.
    Return (repeats, exports, metrics), with geometry first, followed by each
    target's rank-IC and R^2. Return None when day uncertainty is unavailable.
    """
    if not repeats or masks[window].sum() < 2:
        return None
    dates = exports[0]['dates'][masks[window]]
    days = np.unique(dates)
    if len(days) < 2:
        return None
    day_rows = [np.flatnonzero(dates == day) for day in days]
    n_metrics = 2 + (2 * len(TARGETS) if prepared is not None else 0)
    samples = np.full((repeats, len(exports), n_metrics), np.nan)
    rng = np.random.default_rng(seed)
    vectors = [export['z'][masks[window]] for export in exports]
    for b in range(repeats):
        drawn = rng.integers(0, len(days), size=len(days))
        # Repeat all rows for a selected day together, preserving within-day
        # dependence among overlapping windows. Metrics pool the sampled rows.
        rows = np.concatenate([day_rows[day] for day in drawn])
        for i, z in enumerate(vectors):
            samples[b, i, :2] = rankme(z[rows]), participation_ratio(z[rows])
            if prepared is not None:
                for j in range(len(TARGETS)):
                    pred = predictions[i][window][rows, j]
                    target = prepared[f'T_{window}'][rows, j]
                    valid = prepared[f'Vt_{window}'][rows, j]
                    samples[b, i, 2 + 2*j:4 + 2*j] = rank_ic(pred, target, valid), r2(pred, target, valid)
    return samples


def _add_intervals(runs, samples, window):
    """Add bootstrap bounds only to the chosen evaluation window's metrics."""
    if samples is None:
        return
    for i, run in enumerate(runs):
        metrics = run['windows'][window]
        for k, name in enumerate(GEOMETRY):
            metrics['geometry'][name]['ci'] = _estimate(0, samples[:, i, k])['ci']
        for j, target in enumerate(TARGETS):
            for k, name in enumerate(PREDICTION):
                if 'residual_predictions' in metrics:
                    metrics['residual_predictions'][target][name]['ci'] = _estimate(0, samples[:, i, 2 + 2*j + k])['ci']


def _seed_summary(values):
    """Summarize encoder-seed variation at fixed dates with a Student t interval.

    Day-bootstrap uncertainty is separate. A single finite seed supplies a
    mean but no estimate of across-seed uncertainty.
    """
    mean, ci, n = seed_interval(values)
    return dict(mean=mean, ci=ci if n > 1 else None, n=n, per_seed=values,
                unit='encoder seed; fixed evaluation dates')


def _aggregates(runs):
    output = {}
    for tokenizer in dict.fromkeys(run['tokenizer'] for run in runs):
        members = [run for run in runs if run['tokenizer'] == tokenizer]
        windows = {}
        for window in WINDOWS:
            records = [run['windows'][window] for run in members]
            summary = {'geometry': {metric: _seed_summary([r['geometry'][metric]['value'] for r in records])
                                    for metric in GEOMETRY}}
            if 'residual_predictions' in records[0]:
                summary['residual_predictions'] = {
                    target: {metric: _seed_summary([r['residual_predictions'][target][metric]['value'] for r in records])
                             for metric in PREDICTION} for target in TARGETS}
            windows[window] = summary
        output[tokenizer] = dict(n_seeds=len(members), seeds=[r['seed'] for r in members], windows=windows)
    return output


def _paired_deltas(runs, samples, window):
    """Compare tokenizer means using paired days, with selected seeds fixed.

    Average run metrics within each tokenizer, then subtract on every shared
    day draw. Seed identities need not match: the pairing here is by day only.
    """
    groups = {tokenizer: [i for i, run in enumerate(runs) if run['tokenizer'] == tokenizer]
              for tokenizer in dict.fromkeys(run['tokenizer'] for run in runs)}
    output = []
    labels = list(groups)
    for a, left in enumerate(labels):
        for right in labels[:a]:
            li, ri = groups[left], groups[right]
            delta_samples = samples[:, li].mean(axis=1) - samples[:, ri].mean(axis=1) if samples is not None else None
            def delta(section, metric, index, target=None):
                def value(i):
                    row = runs[i]['windows'][window][section]
                    return (row[target] if target else row)[metric]['value']
                point = np.mean([value(i) for i in li]) - np.mean([value(i) for i in ri])
                return _estimate(point, delta_samples[:, index] if delta_samples is not None else None)
            entry = dict(left=left, right=right, window=window,
                         unit='paired evaluation days; selected encoder seeds fixed',
                         geometry={metric: delta('geometry', metric, i) for i, metric in enumerate(GEOMETRY)})
            if 'residual_predictions' in runs[0]['windows'][window]:
                entry['residual_predictions'] = {target: {
                    metric: delta('residual_predictions', metric, 2 + 2*j + k, target)
                    for k, metric in enumerate(PREDICTION)} for j, target in enumerate(TARGETS)}
            output.append(entry)
    return output


def seed_comparisons(runs):
    """Relative RankMe versus the best other tokenizer, paired by model seed.

    Use only seeds present in every selected tokenizer. Choose the best other
    tokenizer separately for each seed, then average percentage differences.
    These ratios compare matched seeds at fixed dates, not bootstrap samples.
    """
    groups = {tokenizer: {run['seed']: run for run in runs if run['tokenizer'] == tokenizer and run['seed'] is not None}
              for tokenizer in dict.fromkeys(run['tokenizer'] for run in runs)}
    if len(groups) < 2:
        return []
    shared = set.intersection(*(set(group) for group in groups.values()))
    output = []
    for tokenizer, members in groups.items():
        for window in WINDOWS:
            values, seeds, next_best = [], [], {}
            for seed in sorted(shared, key=str):
                other = {name: group[seed]['windows'][window]['geometry']['rankme']['value']
                         for name, group in groups.items() if name != tokenizer}
                if not all(np.isfinite(value) and value > 0 for value in other.values()):
                    continue
                best = max(other, key=other.get)
                value = members[seed]['windows'][window]['geometry']['rankme']['value']
                if not np.isfinite(value):
                    continue
                values.append((value / other[best] - 1.0) * 100.0)
                seeds.append(seed)
                next_best[best] = next_best.get(best, 0) + 1
            if values:
                output.append(dict(tokenizer=tokenizer, window=window, metric='relative_rankme_pct',
                                   comparator='best other selected tokenizer within each matched seed',
                                   seeds=seeds, next_best_counts=next_best, **_seed_summary(values)))
    return output


def recalculate_archive(path, *, eval_window='test'):
    """Recalculate geometry summaries and the 5% milestone from per-seed values.

    Saved means, confidence bounds and comparison results are not calculation
    inputs. This aggregates recorded measurements; it does not reevaluate models.
    """
    path = Path(path)
    payload = path.read_bytes()
    record = json.loads(payload)
    if not isinstance(record, dict) or record.get('schema_version') != 1:
        raise ValueError('--archive requires a schema-1 experiment.json with per-seed measurements')
    summary = record.get('across_seed_results', {})
    if not isinstance(summary, dict):
        raise ValueError('The archive must contain across_seed_results with per-seed measurements')
    seeds = summary.get('seeds', [])
    tokenizers = summary.get('tokenizers', [])
    if not isinstance(seeds, list) or not seeds or any(not isinstance(s, (str, int)) or isinstance(s, bool) for s in seeds):
        raise ValueError('The archive must identify each encoder seed')
    seeds = [str(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Archive seed identifiers must be unique')
    if not isinstance(tokenizers, list) or not tokenizers or any(not isinstance(t, str) for t in tokenizers):
        raise ValueError('The archive must list its tokenizer variants')
    if len(set(tokenizers)) != len(tokenizers) or 'convolutional' not in tokenizers or len(tokenizers) < 2:
        raise ValueError('The milestone requires convolutional and at least one other distinct tokenizer')
    values = {}
    for tokenizer in tokenizers:
        for window in WINDOWS:
            for metric, archived_name in (('rankme', 'rankme'), ('participation_ratio', 'pr')):
                try:
                    raw = summary['per_tokenizer'][tokenizer]['windows'][window][archived_name]['per_seed']
                    array = np.asarray(raw, dtype=np.float64)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f'Missing or invalid per-seed {metric}: {tokenizer}, {window}') from error
                if array.shape != (len(seeds),) or not np.isfinite(array).all():
                    raise ValueError(f'Expected one finite {metric} value per seed: {tokenizer}, {window}')
                values[tokenizer, window, metric] = array
    runs = []
    for tokenizer in tokenizers:
        for i, seed in enumerate(seeds):
            windows = {}
            for window in WINDOWS:
                result = dict(geometry={metric: dict(value=float(values[tokenizer, window, metric][i])) for metric in GEOMETRY})
                count = summary.get('n_eval_by_window', {}).get(window)
                if count is not None:
                    result['n_windows'] = count
                if window == summary.get('eval_window') and summary.get('n_days') is not None:
                    result['n_days'] = summary['n_days']
                windows[window] = result
            runs.append(dict(tokenizer=tokenizer, seed=seed, d_model=summary.get('d_model'), windows=windows))
    comparisons = [row for row in seed_comparisons(runs)
                   if row['tokenizer'] == 'convolutional' and row['window'] == 'test']
    if not comparisons:
        raise ValueError('The archive has no assessable matched-seed convolutional test comparison')
    for comparison in comparisons:
        comparison['threshold_pct'] = 5.0
        comparison['lower_ci_exceeds_threshold'] = (bool(comparison['ci'][0] > 5.0)
                                                   if comparison['ci'] is not None else None)
    corpus = record.get('corpus', {})
    split_dates = {}
    date_range = corpus.get('date_range')
    if isinstance(date_range, list) and len(date_range) == 2:
        split_dates['train_start'] = date_range[0]
    if corpus.get('splits', {}).get('train_end'):
        split_dates['train_end'] = corpus['splits']['train_end']
    for window in WINDOWS:
        bounds = corpus.get('splits', {}).get(window)
        if isinstance(bounds, list) and len(bounds) == 2:
            split_dates.update({window + '_start': bounds[0], window + '_end': bounds[1]})
    return dict(schema_version=2, source_kind='archived_measurements',
                source=dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest()),
                corpus=dict(sha256=corpus.get('corpus_sha256'), split_dates=split_dates),
                evaluation=dict(eval_window=eval_window,
                    ci_method='95% Student-t intervals across encoder seeds at fixed evaluation dates; mean ± t(0.975, n−1) × sample standard deviation / sqrt(n).',
                    notes=['Means and confidence intervals are recalculated from the full-precision per-seed geometry measurements.',
                           'Relative RankMe improvements are computed within matched seeds before averaging; the test milestone requires a lower 95% confidence bound above 5%.',
                           'Underlying encoder measurements are retained from the archive. Recomputing them requires the original corpus and encoder checkpoints.']),
                calculation_code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                         (Path(__file__), Path(__file__).with_name('metrics.py'))},
                runs=runs, seed_aggregates=_aggregates(runs), seed_comparisons=comparisons)


def _indicator_comparison(exports, prepared, masks, window, repeats, seed):
    """Test whether saved embeddings add to indicators on original targets.

    Compare one indicators-only fit with indicators-plus-embeddings fits for
    each encoder. Unlike the residual probes, these models predict unscaled,
    unresidualized corpus targets and use chronological training-day folds.
    Bootstrap scores reuse fitted predictions and pair evaluation days;
    encoder seeds are resampled independently within each tokenizer.
    """
    from ..encoder.config import ACTIVE_SSL_FEATURES
    from ..encoder.forward_split import build_residualizer_scalars
    raw = prepared['ssl_data']
    # Take anchor indicators before encoder normalization and targets before
    # residualization. fit_predict fits each model's scaling on training rows.
    indicators, names = build_residualizer_scalars(raw['X_windows'], ACTIVE_SSL_FEATURES)
    indicators = np.asarray(indicators, dtype=float)
    y = np.asarray(raw['fwd_predictive'][:, :4], dtype=float)
    valid = np.repeat(raw['fwd_predictive_valid'][:, None], 4, axis=1) & np.isfinite(y)
    train, test = masks['train'], masks[window]
    dates = exports[0]['dates']
    if len(np.unique(dates[train])) < 6:
        raise ValueError('Original-target indicator comparison needs at least six training days for five chronological folds')
    predictions, fits, folds = [], {}, None
    def fit(label, x):
        nonlocal folds
        pred, summary, folds = fit_predict(x[train], y[train], valid[train], dates[train], x[test])
        predictions.append(pred)
        fits[label] = summary
    fit('indicators', indicators)
    tokenizers = list(dict.fromkeys(e['metadata']['tokenizer'] for e in exports))
    groups = []
    seeds = {}
    for tokenizer in tokenizers:
        group = []
        members = [e for e in exports if e['metadata']['tokenizer'] == tokenizer]
        seeds[tokenizer] = [e['metadata'].get('seed') for e in members]
        for e in members:
            group.append(len(predictions))
            fit(e['id'], np.column_stack([indicators, e['z']]))
        groups.append(group)
    # Keep individual fitted models: (baseline plus encoder runs, rows, targets).
    predictions = np.stack(predictions)
    if repeats:
        result = indicator_bootstrap(predictions, y[test], valid[test], dates[test], None, repeats, seed, groups=groups)
        point = np.asarray(result['point'])
        intervals = np.asarray(result['95']['metric_ci'])
    else:
        days, ids = np.unique(dates[test], return_inverse=True)
        per = metric_function(predictions, y[test], valid[test], ids)(np.ones(len(days)))
        point = np.concatenate([per[:1], np.stack([per[g].mean(axis=0) for g in groups])])
        result, intervals = None, None
    labels = ['indicators', *tokenizers]
    models = {}
    for a, label in enumerate(labels):
        models[label] = {'targets': {target: {metric: {
            'value': float(point[a, j, k]), 'ci': intervals[a, j, k].tolist() if intervals is not None else None}
            for k, metric in enumerate(PREDICTION)} for j, target in enumerate(TARGETS)}}
    deltas = []
    for a, left in enumerate(labels):
        for b, right in enumerate(labels[:a]):
            ci = result['95']['comparisons'][f'{a}_minus_{b}']['delta_ci'] if result else None
            deltas.append(dict(left=left, right=right, targets={target: {metric: {
                'value': float(point[a, j, k] - point[b, j, k]), 'ci': ci[j][k] if ci else None}
                for k, metric in enumerate(PREDICTION)} for j, target in enumerate(TARGETS)}))
    sensitivity = None
    # Equal group sizes alone do not establish pairing; seed identities must
    # match in the same order before exposing the paired-seed sensitivity bounds.
    if result and result['paired_seed_95'] and len({tuple(v) for v in seeds.values()}) == 1:
        sensitivity = result['paired_seed_95']
    return dict(targets=list(TARGETS), models=models, paired_deltas=deltas, seeds=seeds,
                n_days=len(np.unique(dates[test])), n_windows=int(test.sum()), eval_window=window,
                bootstrap_repeats=repeats, bootstrap_seed=seed,
                indicator_names=names, fits=fits, folds=folds, paired_seed_sensitivity=sensitivity,
                method='Original targets; raw anchor indicators; train-only scaling; five chronological whole-day ridge folds; paired whole-day percentile bootstrap with independent encoder-seed resampling within each tokenizer.',
                limitations='Conditional on the training dataset and fitted encoders. Target-wise intervals are not multiplicity-adjusted. A single encoder seed does not estimate training-seed uncertainty.')


def evaluate(embedding_paths, *, data_path=None, checkpoint_paths=None, indicators=False,
             bootstrap=1000, bootstrap_seed=42, embargo_days=5, eval_window='test', shuffle_n=0,
             indicator_bootstrap_repeats=None, indicator_bootstrap_seed=None):
    """Evaluate saved vectors and return a schema-version-2 metrics record.

    Exports alone support geometry. A matching corpus and checkpoints also
    enable training-fitted residual probes; indicators adds a separate
    original-target comparison. Score all date splits, but bootstrap only
    eval_window. Encoders remain frozen throughout evaluation.
    """
    if indicator_bootstrap_repeats is None:
        indicator_bootstrap_repeats = bootstrap
    if indicator_bootstrap_seed is None:
        indicator_bootstrap_seed = bootstrap_seed
    if bootstrap < 0 or indicator_bootstrap_repeats < 0 or embargo_days < 0 or shuffle_n < 0 or eval_window not in ('val', 'test'):
        raise ValueError('Invalid bootstrap, embargo, shuffle count or evaluation window')
    if not embedding_paths:
        raise ValueError('Supply at least one embedding export')
    if bool(data_path) != bool(checkpoint_paths):
        raise ValueError('Full prediction evaluation requires both --data and --checkpoints')
    if (indicators or shuffle_n) and not data_path:
        raise ValueError('Indicator and shuffle diagnostics require --data and --checkpoints')
    exports = [load_embeddings(path) for path in embedding_paths]
    for export in exports:
        metadata = export['metadata']
        if metadata.get('seed') is not None:
            metadata['seed'] = int(metadata['seed'])
        if not metadata.get('tokenizer'):
            raise ValueError('Export metadata must identify its tokenizer; re-export embeddings')
    # Stable tokenizer/seed order makes random draws independent of input-file
    # order and aligns matched seed positions for the sensitivity calculation.
    tokenizer_order = {'linear': 0, 'patch': 1, 'convolutional': 2}
    exports.sort(key=lambda e: (tokenizer_order.get(e['metadata']['tokenizer'], 3),
                               e['metadata']['tokenizer'],
                               e['metadata']['seed'] if e['metadata'].get('seed') is not None else -1))
    _align_exports(exports)
    for export in exports:
        if not export['metadata'].get('corpus_sha256'):
            raise ValueError('Embedding export must include corpus_sha256; re-export with potofgold-embed')
        metadata = export['metadata']
        export['id'] = f"{metadata.get('tokenizer', export['path'].stem)}_s{metadata.get('seed', 'unknown')}"
    identities = [(e['metadata'].get('tokenizer'), e['metadata'].get('seed')) for e in exports]
    if len(set(identities)) != len(identities):
        raise ValueError('Repeated tokenizer/seed identity; evaluate one export per checkpoint')
    prepared = _full_data(exports, data_path, checkpoint_paths, embargo_days) if data_path else None
    first = exports[0]['metadata']
    masks = split_masks(exports[0]['dates'], first.get('split_dates', {}), embargo_days)
    if not masks[eval_window].any():
        raise ValueError(f'No {eval_window} windows remain after applying split dates and embargo')
    predictions, runs = [], []
    for export in exports:
        metadata, z = export['metadata'], export['z']
        prediction = None
        if prepared is not None:
            # _full_data established row identity and order; verify split sizes
            # before pairing each window's vectors with its prepared targets.
            for window in WINDOWS:
                if int(masks[window].sum()) != len(prepared[f'T_{window}']):
                    raise ValueError(f'{window} target and embedding rows do not align')
            by_window = {window: z[masks[window]] for window in WINDOWS}
            # Fit each target once on train and retain predictions for scoring
            # and bootstrap reuse. No validation/test targets enter these fits.
            predictions_by_target = [probe_multi(by_window['train'], prepared['T_train'][:, j],
                                                prepared['Vt_train'][:, j], by_window) for j in range(len(TARGETS))]
            prediction = {window: np.column_stack([p[window] for p in predictions_by_target]) for window in WINDOWS}
        predictions.append(prediction)
        windows = {}
        for window in WINDOWS:
            mask = masks[window]
            row = dict(n_windows=int(mask.sum()), n_days=len(np.unique(export['dates'][mask])), geometry=_geometry(z[mask]))
            if prediction is not None:
                row['residual_predictions'] = _prediction_metrics(prediction[window], prepared[f'T_{window}'], prepared[f'Vt_{window}'])
            windows[window] = row
        run = dict(id=export['id'], tokenizer=metadata.get('tokenizer', export['path'].stem), seed=metadata.get('seed'),
                   d_model=int(z.shape[1]), embedding_file=str(export['path']), checkpoint_sha256=metadata.get('checkpoint_sha256'),
                   readout=metadata['readout'], stride=metadata['stride'], training=export.get('training', {}), windows=windows)
        if shuffle_n:
            run['shuffle_diagnostic'] = shuffle_diagnostic(z[masks['train']], prepared['T_train'], prepared['Vt_train'],
                                                          z[masks[eval_window]], prepared[f'T_{eval_window}'], prepared[f'Vt_{eval_window}'],
                                                          rank_ic, ic, shuffle_n)
        runs.append(run)
    samples = _day_bootstrap(exports, masks, prepared, predictions, eval_window, bootstrap, bootstrap_seed)
    _add_intervals(runs, samples, eval_window)
    # Use the archive only to label provenance; all scores above were computed
    # from the supplied exports, including when their hashes match the archive.
    archived_path = ROOT / 'results/experiment.json'
    archived = json.loads(archived_path.read_text()) if archived_path.exists() else {}
    original_hashes = set(archived.get('indicator_comparison', {}).get('checkpoint_sha256', {}).values())
    original = (first['corpus_sha256'] == archived.get('corpus', {}).get('corpus_sha256')
                and all(e['metadata'].get('checkpoint_sha256') in original_hashes for e in exports))
    synthetic = bool(first.get('synthetic_data'))
    comparison = _indicator_comparison(exports, prepared, masks, eval_window,
                                       indicator_bootstrap_repeats, indicator_bootstrap_seed) if indicators else None
    return dict(schema_version=2, source_kind='synthetic' if synthetic else ('original' if original else 'new'),
                corpus=dict(path=str(data_path) if data_path else None, sha256=first['corpus_sha256'], synthetic_data=synthetic,
                            n_features=first.get('n_features'), split_dates=first['split_dates']),
                evaluation=dict(eval_window=eval_window, embargo_days=embargo_days, bootstrap_repeats=bootstrap,
                                bootstrap_seed=bootstrap_seed, ci_method='95% whole-day percentile bootstrap; frozen encoders and probes; same resampled days across runs. Across-seed summaries use a Student t interval over encoder seeds.',
                                residual_probe_method='Training-fitted scalar residualization and target scaling; StandardScaler and RidgeCV(0.1,1,10,100) fitted on training representations only; float64 pooled embeddings, as in the original across-seed evaluation.',
                                notes=['Training scores are in-sample diagnostics.', 'Residual-target and original-target predictions answer different questions.',
                                       'Geometry averages over variates for token exports; multilayer readouts have a different dimension.',
                                       'Undefined metrics or unavailable uncertainty are null; a single seed has no across-seed confidence interval.']),
                runs=runs, seed_aggregates=_aggregates(runs), seed_comparisons=seed_comparisons(runs),
                paired_deltas=_paired_deltas(runs, samples, eval_window),
                indicator_comparison=comparison)


def _finite_json(value):
    """Convert NumPy values for JSON and represent undefined numbers as null."""
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--embeddings', nargs='+', type=Path,
                        help='One or more saved embedding files; a single encoder is sufficient')
    source.add_argument('--archive', type=Path,
                        help='Recalculate geometry means, seed confidence intervals and the 5%% milestone from archived per-seed measurements')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--checkpoints', nargs='+', type=Path)
    parser.add_argument('--indicators', action='store_true', help='Also compare indicators with indicators plus embeddings on original targets')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--eval-window', choices=('val', 'test'), default='test')
    parser.add_argument('--embargo-days', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=1000, help='Whole-day bootstrap repetitions for --embeddings; 0 omits day-bootstrap intervals')
    parser.add_argument('--bootstrap-seed', type=int, default=42)
    parser.add_argument('--indicator-bootstrap', type=int, default=None,
                        help='Original-target bootstrap repetitions; defaults to --bootstrap')
    parser.add_argument('--indicator-bootstrap-seed', type=int, default=None,
                        help='Original-target resampling seed; defaults to --bootstrap-seed')
    parser.add_argument('--shuffle-n', type=int, default=0, help='Optional shuffled-input probe diagnostic repetitions')
    parser.add_argument('--max-cores', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.out.suffix != '.json':
        parser.error('--out must end with .json')
    inputs = [*(args.embeddings or []), *(args.checkpoints or []),
              *([args.data] if args.data else []), *([args.archive] if args.archive else [])]
    if args.out.resolve() in {path.resolve() for path in inputs}:
        parser.error('--out must differ from every input file, including when --overwrite is set')
    if args.out.resolve() == (ROOT / 'results/experiment.json').resolve():
        parser.error('Preserve the archived experiment record; write new metrics under runs/')
    if args.out.exists() and not args.overwrite:
        parser.error(f'Output already exists: {args.out}; choose another path or use --overwrite')
    if args.archive:
        if args.data or args.checkpoints or args.indicators or args.shuffle_n:
            parser.error('--archive uses saved per-seed geometry only; omit --data, --checkpoints, --indicators and --shuffle-n')
        try:
            record = recalculate_archive(args.archive, eval_window=args.eval_window)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_finite_json(record), indent=2, allow_nan=False) + '\n')
        milestone = record['seed_comparisons'][0]
        print(f'Wrote {args.out}: recalculated geometry summaries for {len(record["runs"])} archived encoder runs')
        print('Convolutional test RankMe improvement: '
              f'{milestone["mean"]:.2f}%; 95% CI {milestone["ci"]}; '
              f'lower bound > 5%: {milestone["lower_ci_exceeds_threshold"]}')
        return
    if bool(args.data) != bool(args.checkpoints):
        parser.error('Prediction metrics require both --data and --checkpoints. Supply the checkpoint(s) for the chosen embeddings, or omit both arguments for geometry only.')
    missing = [str(path) for path in args.embeddings if not path.is_file()]
    if missing:
        parser.error('Embedding file(s) not found: ' + ', '.join(missing) +
                     '. Export the chosen encoder(s) with potofgold-embed first. '
                     'One encoder is sufficient: list only its existing embedding file and matching checkpoint; all three encoders are not required.')
    missing = [str(path) for path in args.checkpoints or [] if not path.is_file()]
    if missing:
        parser.error('Checkpoint file(s) not found: ' + ', '.join(missing) +
                     '. Train the chosen encoder(s) first, or correct --checkpoints to match the exported embeddings. One encoder is sufficient.')
    if args.data and not args.data.is_file():
        parser.error(f'Corpus not found: {args.data}. Run potofgold-pack-data first, or correct --data to the corpus used for these embeddings.')
    from ..encoder.train import _cap_cores
    _cap_cores(args.max_cores)
    import torch
    torch.set_num_threads(max(1, args.max_cores))
    try:
        record = evaluate(args.embeddings, data_path=args.data, checkpoint_paths=args.checkpoints,
                          indicators=args.indicators, bootstrap=args.bootstrap, bootstrap_seed=args.bootstrap_seed,
                          embargo_days=args.embargo_days, eval_window=args.eval_window, shuffle_n=args.shuffle_n,
                          indicator_bootstrap_repeats=args.indicator_bootstrap,
                          indicator_bootstrap_seed=args.indicator_bootstrap_seed)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_finite_json(record), indent=2, allow_nan=False) + '\n')
    print(f'Wrote {args.out}: {len(record["runs"])} saved embedding runs ({record["source_kind"]})')


if __name__ == '__main__':
    main()
