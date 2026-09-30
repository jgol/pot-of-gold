"""Train an encoder to predict a future window's residualized teacher embedding.

The student and an exponential-moving-average (EMA) teacher encode paired
same-day windows. Teacher windows begin at or after the student's final bar.
The teacher target is detached after removing a ridge-fitted projection onto
indicator variables from the student's final bar. Training and validation fit
this projection within each batch. Evaluation probes instead fit their target
residualization on training data and apply that fit to other splits.

The objective combines latent prediction, embedding variance and covariance
penalties, linear decorrelation, and a gradient-reversal adversary. The variance
and covariance penalties discourage collapsed embeddings; checkpoint selection
requires validation participation ratio >= 0.20 * d_model and minimizes the
combined validation loss among qualifying epochs. If no epoch qualifies, the
final encoder is saved with its threshold status recorded.

The YAML supplies shared architecture and training settings; --tokenizer selects
linear, patch, or convolutional tokenization. The experiment date splits and
preprocessing settings come from config.EXPERIMENT_PROFILE. Checkpoints contain
encoder and prediction-head weights, normalization statistics, and run metadata,
serialized as base64-encoded torch pickles.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _cap_cores(n: int) -> None:
    """Set numerical-library thread limits before importing NumPy and PyTorch.

    main() also applies PyTorch's in-process thread limits after importing it.
    """
    n = max(int(n), 1)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


def _reconstruct_last_bar_pos(window_dates, seq_len, stride):
    """Recover each window's final-bar position within its trading day.

    load_ssl_days emits windows in date order, then increasing start position.
    Within each contiguous date group, window k ends at seq_len - 1 + k*stride.
    """
    import numpy as np
    n = len(window_dates)
    last_bar_pos = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        j = i
        while j < n and window_dates[j] == window_dates[i]:
            last_bar_pos[j] = (seq_len - 1) + (j - i) * stride
            j += 1
        i = j
    return last_bar_pos


def _build_pairs(window_dates, last_bar_pos, seq_len, horizon_bars, tol=2):
    """Align each student window with the nearest qualifying teacher window.

    Both windows must be in the same trading day. Teacher ends must be later
    than student ends by at least horizon_bars - tol bars. A separate assertion
    rejects pairs whose teacher starts before the student's final bar.
    Windows without a qualifying future retain a self-index and valid=False;
    callers exclude them before constructing training and validation batches.
    """
    import numpy as np
    n = len(window_dates)
    teacher_idx = np.arange(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    thresh = horizon_bars - tol
    i = 0
    while i < n:
        j = i
        while j < n and window_dates[j] == window_dates[i]:
            j += 1
        # [i, j) is one day's block, ordered by increasing last_bar_pos.
        day = slice(i, j)
        lbp = last_bar_pos[day]
        for k in range(i, j):
            p = last_bar_pos[k]
            # Search only this day, requiring a later end and the requested gap.
            ahead = lbp - p
            cand = np.where((ahead > 0) & (ahead >= thresh))[0]
            if cand.size:
                best = cand[np.argmin(ahead[cand])]
                teacher_idx[k] = i + int(best)
                valid[k] = True
        i = j
    # A teacher may share the student's final bar, but cannot include earlier bars.
    v = valid
    t_first_bar = last_bar_pos[teacher_idx] - (seq_len - 1)
    s_last_bar = last_bar_pos
    bad = v & (t_first_bar < s_last_bar)
    assert not bad.any(), (
        f"{int(bad.sum())} teacher windows begin before the student's final bar. "
        f"Increase horizon_bars={horizon_bars} for seq_len={seq_len}; "
        f"minimum start-to-end gap is {int((t_first_bar - s_last_bar)[v].min())} bars.")
    # Check that the paired indices stay within their trading day.
    same_day = np.array([window_dates[teacher_idx[k]] == window_dates[k] for k in range(n) if valid[k]])
    assert same_day.all() if same_day.size else True, "pairing crossed a day boundary"
    return teacher_idx, valid


def main():
    ap = argparse.ArgumentParser(description="Train an encoder with paired past and future windows.")
    ap.add_argument("--config", required=True, type=Path,
                    help="Shared YAML with top-level `ssl_hyperparams:` + `training:` sections.")
    ap.add_argument("--tokenizer", choices=("linear", "patch", "convolutional"), default=None,
                    help="Tokenizer architecture. Overrides the config; required "
                         "unless the config sets ssl_hyperparams.tokenizer.")
    ap.add_argument("--data", required=True, type=Path,
                    help="Corpus bundle .npz (see data/DATA.md).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Write the checkpoint to this file (base64 torch pickle).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override num_epochs (default: ssl_hyperparams.num_epochs from the config).")
    ap.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--max-cores", type=int, default=8,
                    help="Cap OMP/MKL/BLAS env AND torch.set_num_threads to this many cores.")
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing run output.")
    args = ap.parse_args()
    if args.out.exists() and not args.overwrite:
        ap.error(f"Output already exists: {args.out}. Choose a new --out or use --overwrite.")

    # Read architecture settings and training settings from the shared YAML.
    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    ssl_hp_cfg = dict(cfg.get("ssl_hyperparams", {}))
    training_cfg = dict(cfg.get("training", {}))
    if training_cfg.get("use_v3_features", True) is not True:
        ap.error("training.use_v3_features must be true; training uses the 147-feature v3 layout.")

    tokenizer = args.tokenizer or ssl_hp_cfg.get("tokenizer")
    if tokenizer not in ("linear", "patch", "convolutional"):
        ap.error(f"tokenizer must be linear|patch|convolutional (set via --tokenizer or the config), got {tokenizer!r}")

    seed = args.seed if args.seed is not None else int(training_cfg.get("seed", 4395))
    ema_decay = float(training_cfg.get("ema_decay", 0.99))
    horizon_bars = int(training_cfg.get("horizon_bars", 60))
    embargo_days = int(training_cfg.get("embargo_days", 5))
    loss_weights = dict(training_cfg.get("loss_weights", {}))

    seq_len_cfg = int(ssl_hp_cfg.get("seq_len", 60))
    if horizon_bars < seq_len_cfg:
        ap.error(f"training.horizon_bars={horizon_bars} must be >= seq_len ({seq_len_cfg}). "
                 f"Pair construction also checks that teacher windows start at or after the student's final bar.")

    # Set thread limits and MPS fallback before numerical libraries initialize.
    _cap_cores(args.max_cores)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    with np.load(args.data, allow_pickle=False) as bundle:
        data_manifest = json.loads(bytes(bundle["_manifest"]).decode("utf-8"))
    synthetic_data = bool(data_manifest.get("synthetic", False))
    print(f"[training] data: {'synthetic demonstration' if synthetic_data else 'user-supplied corpus'}")

    import torch
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(max(int(args.max_cores), 1))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # parallel engine already initialized; the env caps still apply
    print(f"[training] threads: torch intra-op={torch.get_num_threads()} | "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    from potofgold.encoder import objective
    from potofgold.encoder.anticollapse import compute_participation_ratio
    from potofgold.encoder.forward_split import build_residualizer_scalars, apply_embargo

    from potofgold.encoder import config as encoder_config
    from potofgold.encoder.data import load_ssl_days, prepare_input_splits
    from potofgold.encoder.model import iTransformerEncoder, _decay_split

    hp = dict(encoder_config.SSL_HYPERPARAMS)
    # Apply this run's YAML settings to a copy of the encoder defaults.
    hp.update(ssl_hp_cfg)
    if hp.get("quantile_flow", False):
        ap.error("Self-supervised paired-window preprocessing requires quantile_flow=false.")
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    profile = encoder_config.EXPERIMENT_PROFILE
    date_overrides = {key: profile[key] for key in
                      ("TRAIN_START", "TRAIN_END", "VAL_START", "VAL_END", "TEST_START", "TEST_END")}
    preprocess_indices = profile["preprocess_log1p_indices"]

    # Resolve the tokenizer and dimensions before loading windows.
    hp["tokenizer"] = tokenizer
    D = hp["d_model"]
    seq_len = hp["seq_len"]
    stride = int(training_cfg.get("window_stride", encoder_config.WINDOW_STRIDE))
    V = int(training_cfg.get("n_ssl_features", encoder_config.ACTIVE_N_SSL_FEATURES))
    if V != encoder_config.ACTIVE_N_SSL_FEATURES:
        ap.error(f"n_ssl_features must match the active {encoder_config.ACTIVE_N_SSL_FEATURES}-channel layout")
    if stride <= 0:
        ap.error("window_stride must be positive")

    epochs = args.epochs if args.epochs is not None else int(hp["num_epochs"])
    if epochs < 1:
        ap.error("epochs must be at least 1")
    hp["num_epochs"] = epochs

    print(f"[training] tokenizer={tokenizer}  device={device}  d_model={D}  seq_len={seq_len}  "
          f"stride={stride}  n_variates={V}  horizon_bars={horizon_bars}  epochs={epochs}")
    print(f"[training] checkpoint out: {args.out}")

    # Fit input normalization while retaining window order. Pairing needs dates,
    # bar positions, and indicator rows to stay aligned.
    print(f"[training] loading SSL days {profile['TRAIN_START']}..{profile['TEST_END']} …")
    ssl_data = load_ssl_days(args.data, profile['TRAIN_START'], profile['TEST_END'], T=seq_len, stride=stride,
                             log1p_indices=tuple(preprocess_indices))
    inputs = prepare_input_splits(ssl_data, embargo_days=embargo_days,
                                  date_overrides=date_overrides)
    z_stats = inputs["z_stats"]  # {"mean","std","log1p_cols","min_std_floor"}
    del inputs  # Pair construction below retains its own normalized windows.

    X_windows = ssl_data["X_windows"]                      # (N, T, V), before z-scoring
    window_dates = np.asarray(ssl_data["window_dates"])    # (N,) ISO date strings

    # Apply the signed-log transform, fitted normalization, and clipping without
    # reordering windows. Quantile preprocessing is rejected above because these
    # saved mean/std statistics are not sufficient to reproduce that transform.
    mean = np.asarray(z_stats["mean"], dtype=np.float32)
    std = np.asarray(z_stats["std"], dtype=np.float32)
    log1p_cols = list(z_stats.get("log1p_cols", []))
    Xn = X_windows.astype(np.float32).copy()
    if log1p_cols:
        Xn[:, :, log1p_cols] = np.sign(Xn[:, :, log1p_cols]) * np.log1p(np.abs(Xn[:, :, log1p_cols]))
    Xn = (Xn - mean) / std
    Xn = np.clip(Xn, -5.0, 5.0)
    Xn = np.nan_to_num(Xn, nan=0.0).astype(np.float32)     # Replace missing normalized inputs with zero before encoding.

    # Extract (N, S) indicator variables from each preprocessed window's final
    # bar. Student-row indicators condition both the future target and adversary.
    scalars_np, reg_names = build_residualizer_scalars(Xn, encoder_config.ACTIVE_SSL_FEATURES)
    scalars_np = np.nan_to_num(scalars_np, nan=0.0).astype(np.float32)
    n_scalars = scalars_np.shape[1]
    print(f"[training] indicator variables: n={n_scalars} {reg_names}")

    # Match the shared date splits and validation embargo while keeping
    # boolean masks available for indexing the paired window arrays.
    train_mask = window_dates <= profile['TRAIN_END']
    val_mask = (window_dates >= profile['VAL_START']) & (window_dates <= profile['VAL_END'])
    val_mask = apply_embargo(window_dates, train_mask, val_mask, embargo_days)

    # Reconstruct within-day positions and pair student rows with future rows.
    last_bar_pos = _reconstruct_last_bar_pos(window_dates, seq_len, stride)
    teacher_idx, valid = _build_pairs(window_dates, last_bar_pos, seq_len, horizon_bars)
    n_tr_pair = int((train_mask & valid).sum())
    n_va_pair = int((val_mask & valid).sum())
    print(f"[training] pairs: train valid={n_tr_pair}/{int(train_mask.sum())}  "
          f"val valid={n_va_pair}/{int(val_mask.sum())}  (horizon_bars={horizon_bars})")
    if n_tr_pair < 8:
        raise SystemExit(f"[training] too few valid train pairs ({n_tr_pair}); increase the date span "
                         f"or lower horizon_bars (still >= {seq_len}).")
    if n_va_pair < 2:
        raise SystemExit("[training] no usable validation pairs; check the corpus dates against the experiment splits.")

    # Torch tensors (kept on CPU; batches moved to device). Xn/scalars are float32.
    Xn_t = torch.from_numpy(Xn)
    sc_t = torch.from_numpy(scalars_np)
    teacher_idx_t = torch.from_numpy(teacher_idx)
    valid_t = torch.from_numpy(valid.astype(np.float32))
    tr_ids = np.nonzero(train_mask & valid)[0]
    va_ids = np.nonzero(val_mask & valid)[0]

    # Initialize the teacher from the student, then update it only by EMA.
    student = iTransformerEncoder(hp, n_variates=V).to(device)
    import copy
    teacher = copy.deepcopy(student).to(device)
    teacher.requires_grad_(False)
    # Evaluation mode disables teacher dropout. LayerNorm remains independent
    # of batch statistics; teacher weights never receive optimizer gradients.
    teacher.eval()
    heads = objective.PredictionHeads(D, n_scalars).to(device)   # mean-pool over variates
    # Use configured loss weights, or SelfSupervisedLoss defaults when absent.
    loss_fn = objective.SelfSupervisedLoss(**loss_weights)

    participation_ratio_threshold = 0.20 * D
    lr = float(hp["learning_rate"])
    warmup_epochs = min(int(hp.get("warmup_epochs", 10)), max(epochs - 1, 1))
    BS = int(hp["batch_size"])

    # Exclude normalization parameters, biases, and embeddings from weight decay.
    fwd_wd = float(hp.get("forward_weight_decay", 1e-2))
    decay_params, no_decay_params = _decay_split([student, heads])
    all_params = decay_params + no_decay_params
    opt = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": fwd_wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[warmup_epochs])
    print(f"[training] optimizer AdamW lr={lr} wd={fwd_wd} (decay {len(decay_params)} / "
          f"no-decay {len(no_decay_params)}) | ema_decay={ema_decay} | participation_ratio_threshold={participation_ratio_threshold:.1f}")

    def _batch(student_ids):
        """Yield aligned device tensors: two (B, T, V) windows, (B, S) indicators, and (B,) validity.

        Use the same seeded order on each call; retain only batches with B >= 2
        for variance and covariance estimates.
        """
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(student_ids))
        for s in range(0, len(order), BS):
            sel = student_ids[order[s:s + BS]]
            if len(sel) < 2:
                continue  # VICReg/decorr need B>=2
            tids = teacher_idx_t[sel]
            xb = Xn_t[sel].to(device)
            xf = Xn_t[tids].to(device)
            scb = sc_t[sel].to(device)
            vb = valid_t[sel].to(device)
            yield xb, xf, scb, vb

    best_val = float("inf")
    best_state = None
    best_participation_ratio = 0.0

    for epoch in range(1, epochs + 1):
        student.train(); heads.train()
        ep_loss = 0.0; ep_n = 0; last_info = None
        for xb, xf, scb, vb in _batch(tr_ids):
            B = xb.shape[0]
            student_tokens = student.encode_variates(xb)   # (B, V, D)
            with torch.no_grad():
                teacher_tokens = teacher.encode_variates(xf)  # (B, V, D)
                teacher_pooled = heads.pool_tokens(teacher_tokens)                          # (B, D)
                # Fit and subtract the indicator projection within this training
                # batch. Detach the target so only the student and heads receive gradients.
                teacher_target = objective.batch_residualize(teacher_pooled, scb).detach()
            loss, info = loss_fn(heads, student_tokens, teacher_target, scb, vb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            # Move the teacher toward the newly updated student parameters.
            objective.ema_update(student, teacher, ema_decay)
            ep_loss += float(loss.item()) * B; ep_n += B
            last_info = info
        scheduler.step()
        train_loss = ep_loss / max(ep_n, 1)

        # Validation repeats the batch-fitted target residualization with no
        # gradients or EMA updates. This combined objective is distinct from the
        # evaluation probes, which use residualization fitted on training data.
        student.eval(); heads.eval()
        v_loss = 0.0; v_n = 0; pooled_chunks = []
        with torch.no_grad():
            for xb, xf, scb, vb in _batch(va_ids):
                st = student.encode_variates(xb)
                tt = teacher.encode_variates(xf)
                tgt = objective.batch_residualize(heads.pool_tokens(tt), scb).detach()
                l, _ = loss_fn(heads, st, tgt, scb, vb)
                v_loss += float(l.item()) * xb.shape[0]; v_n += xb.shape[0]
                pooled_chunks.append(heads.pool_tokens(st).detach().cpu())
        val_loss = v_loss / max(v_n, 1)
        pooled = torch.cat(pooled_chunks, dim=0) if pooled_chunks else torch.zeros(2, D)
        # Compute participation ratio over all retained validation embeddings,
        # rather than averaging batch ratios. This is distinct from RankMe.
        participation_ratio = float(compute_participation_ratio(pooled))

        below = "  below threshold" if participation_ratio < participation_ratio_threshold else ""
        info_str = (f" l_pred={last_info['l_pred']:.4f} l_vicreg={last_info['l_vicreg']:.4f} "
                    f"l_decorr={last_info['l_decorr']:.4f} l_adv={last_info['l_adv']:.4f}"
                    if last_info else "")
        print(f"[training:{tokenizer}] epoch {epoch}/{epochs}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  participation_ratio {participation_ratio:.1f}/{D} (threshold {participation_ratio_threshold:.1f}){below}"
              f"{info_str}")

        # Among epochs meeting the participation-ratio threshold, keep the
        # encoder and heads with the lowest combined validation objective.
        if participation_ratio >= participation_ratio_threshold and val_loss < best_val:
            best_val = val_loss
            best_participation_ratio = participation_ratio
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            best_heads = {k: v.cpu().clone() for k, v in heads.state_dict().items()}

    if best_state is None:
        # If no epoch meets the threshold, retain the final state and record that status.
        print(f"[training] No epoch met the participation-ratio threshold {participation_ratio_threshold:.1f}; "
              f"saving the final checkpoint with participation ratio {participation_ratio:.1f}.")
        best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
        best_heads = {k: v.cpu().clone() for k, v in heads.state_dict().items()}
        best_participation_ratio = participation_ratio
        best_val = val_loss

    # Save the selected encoder and heads with normalization and run metadata
    # so inference can reproduce the input transform and architecture.
    training_metadata = {
        "checkpoint_out": str(args.out),
        "checkpoint_key": args.out.name,
        "synthetic_data": synthetic_data,
        "data_generator": data_manifest.get("generator"),
        "profile": "experiment",
        "objective": "future_latent_prediction",
        "tokenizer": tokenizer,
        "horizon_bars": horizon_bars,
        "ema_decay": ema_decay,
        "epochs": epochs,
        "git_sha": _git_sha(),
        "validation_participation_ratio_at_selection": best_participation_ratio,
        "participation_ratio_threshold": participation_ratio_threshold,
        "participation_ratio_threshold_met": bool(best_participation_ratio >= participation_ratio_threshold),
        "best_val_loss": best_val,
        "d_model": D,
        "seq_len": seq_len,
        "window_stride": stride,
        "n_variates": V,
        "n_indicator_variables": n_scalars,
        "indicator_variables": reg_names,
        "embargo_days": embargo_days,
        "train_pairs": n_tr_pair,
        "val_pairs": n_va_pair,
        "date_span": {"train_start": profile['TRAIN_START'], "train_end": profile['TRAIN_END'],
                      "val_start": profile['VAL_START'], "val_end": profile['VAL_END'], "test_end": profile['TEST_END']},
        "loss_weights": {
            "w_vicreg": loss_fn.w_vicreg, "w_decorr": loss_fn.w_decorr,
            "adv_lambda": loss_fn.adv_lambda, "w_var": loss_fn.w_var, "w_cov": loss_fn.w_cov,
            "vicreg_gamma": loss_fn.vicreg_gamma, "huber_delta": loss_fn.huber_delta},
        "seed": seed,
        "residualization": "Fitted within each training and validation batch; evaluation probes use training-fitted residualization.",
    }

    buf = io.BytesIO()
    torch.save({
        "training_metadata": training_metadata,
        "encoder_state_dict": best_state,
        "heads_state_dict": best_heads,
        "objective": "future_latent_prediction",
        "tokenizer": tokenizer,
        "ssl_hyperparams": dict(hp),
        "ssl_features": encoder_config.ACTIVE_SSL_FEATURES,
        "n_ssl_features": V,
        "use_v3_features": True,
        "z_mean": z_stats.get("mean"),
        "z_std": z_stats.get("std"),
        "log1p_cols": z_stats.get("log1p_cols", []),
        "min_std_floor": z_stats.get("min_std_floor", 0.01),
        "residualizer_scalars": reg_names,
        "horizon_bars": horizon_bars,
        "ema_decay": ema_decay,
        "synthetic_data": synthetic_data,
        "data_generator": data_manifest.get("generator"),
        "profile": "experiment",
        "seed": seed,
    }, buf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(base64.b64encode(buf.getvalue()).decode())
    print(f"[training] saved checkpoint -> {args.out}")

    print(f"[training] DONE. participation_ratio {best_participation_ratio:.1f}/{D} "
          f"({'meets' if best_participation_ratio >= participation_ratio_threshold else 'below'} threshold {participation_ratio_threshold:.1f}); "
          f"best_val {best_val:.4f}")


if __name__ == "__main__":
    main()
