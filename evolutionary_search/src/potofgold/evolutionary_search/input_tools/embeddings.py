"""Stream the shared encoder's multilayer readout directly into thirteen GP vectors."""
import numpy as np

from .schema import GROUP_RANGES, VARIATE_READOUTS, EMBEDDING_COLUMNS, repair_raw_iv


def pool_readouts(tokens, ssl_features):
    """Mean pools and three selected tokens in the archived GP's exact column order."""
    from potofgold.encoder.config import ACTIVE_SSL_FEATURES
    if list(ssl_features) != list(ACTIVE_SSL_FEATURES):
        raise ValueError("GP pooling requires the exact 147-feature v3 order")
    if tokens.ndim != 3 or tokens.shape[1:] != (147, 384):
        raise ValueError(f"Expected multilayer tokens (N,147,384), received {tokens.shape}")
    out = {"EMB_SHARED": tokens.mean(axis=1)}
    out.update({name: tokens[:, start:end].mean(axis=1) for name, (start, end) in GROUP_RANGES.items()})
    out.update({name: tokens[:, list(ssl_features).index(raw_index)].copy()
                for name, raw_index in VARIATE_READOUTS.items()})
    return {name: np.asarray(out[name], dtype=np.float32) for name in EMBEDDING_COLUMNS}


def encode_readouts(model, checkpoint, raw, *, preprocessing, cold_start="masked", iv_repair_mult=4.0,
                    batch_size=32):
    """Encode batches of windows, pool immediately, and retain only (N,13,384).

    Historical GP preprocessing excludes a second raw-column transform of v132.
    The transformer-study profile retains its published v132 transform. Both use
    shared cleaning/VIX helpers and checkpoint-fitted log1p/z-score statistics.
    """
    import torch
    from potofgold.encoder.config import PREPROCESS_LOG1P_INDICES
    from potofgold.encoder.data import preprocess_day, apply_vix_forwardfill, should_skip_day

    if preprocessing not in ("historical-gp", "transformer"):
        raise ValueError("Unknown preprocessing profile")
    if cold_start not in ("masked", "omit") or batch_size < 1:
        raise ValueError("Invalid cold-start mode or batch size")
    hp = checkpoint["ssl_hyperparams"]
    if hp.get("quantile_flow", False):
        raise ValueError("Quantile-flow checkpoints require a saved transform and are unsupported")
    if hp["d_model"] * hp["n_layers"] != 384:
        raise ValueError("The archived GP grammar requires 384-dimensional multilayer vectors")
    if cold_start == "masked" and not hp.get("use_mask_indicator", False):
        raise ValueError("Masked prefixes require a checkpoint trained with a mask indicator")
    if should_skip_day(raw):
        raise ValueError("Corpus day fails the encoder's dead-feature check; rows cannot be silently dropped")
    log_indices = (101, 102, 104, 139) if preprocessing == "historical-gp" else PREPROCESS_LOG1P_INDICES
    day = preprocess_day(raw, log1p_indices=log_indices)
    apply_vix_forwardfill(day)
    repaired = repair_raw_iv(day, iv_repair_mult)
    features = list(checkpoint["ssl_features"])
    selected = day[:, features].copy()
    log_cols = checkpoint.get("log1p_cols", [])
    if log_cols:
        selected[:, log_cols] = np.sign(selected[:, log_cols]) * np.log1p(np.abs(selected[:, log_cols]))
    selected = (selected - np.asarray(checkpoint["z_mean"])) / np.asarray(checkpoint["z_std"])
    selected = np.clip(selected, -5.0, 5.0).astype(np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Non-finite encoder input after saved normalization")
    seq_len = int(hp["seq_len"])
    bars = np.arange(0 if cold_start == "masked" else seq_len - 1, len(day), dtype=np.int64)
    if len(bars) == 0:
        raise ValueError("No complete encoder windows")
    out = {name: np.empty((len(bars), 384), dtype=np.float32) for name in EMBEDDING_COLUMNS}
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(bars), batch_size):
            batch_bars = bars[start:start + batch_size]
            windows = np.full((len(batch_bars), seq_len, len(features)),
                              hp.get("mask_fill_value", -8.0), dtype=np.float32)
            mask = np.ones_like(windows, dtype=np.float32)
            for row, end in enumerate(batch_bars):
                lo = max(0, int(end) - seq_len + 1)
                width = int(end) - lo + 1
                windows[row, -width:] = selected[lo:end + 1]
                mask[row, -width:] = 0.0
            tokens = model.encode_variates(torch.from_numpy(windows).to(device), multi_layer=True,
                                           mask=torch.from_numpy(mask).to(device)).cpu().numpy()
            pooled = pool_readouts(tokens, features)
            for name, values in pooled.items():
                if not np.isfinite(values).all():
                    raise ValueError(f"Encoder emitted non-finite {name}")
                out[name][start:start + len(batch_bars)] = values
            del tokens, pooled, windows, mask
    return out, bars, repaired
