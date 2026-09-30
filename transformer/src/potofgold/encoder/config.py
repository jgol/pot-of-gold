"""Feature indices, preprocessing constants, encoder defaults, and date splits."""
from importlib.resources import files
import json


# Historical checkpoints retain their original bytes; normalize names when read.
TOKENIZER_ALIASES = {"cnn": "convolutional"}


def canonical_tokenizer(name):
    return TOKENIZER_ALIASES.get(name, name)


def canonical_tokenizer_hyperparams(params):
    result = dict(params)
    if "tokenizer" in result:
        result["tokenizer"] = canonical_tokenizer(result["tokenizer"])
    if "cnn_c_dw" in result:
        width = result.pop("cnn_c_dw")
        if "convolutional_c_dw" in result and result["convolutional_c_dw"] != width:
            raise ValueError("Conflicting convolutional channel widths in checkpoint or configuration")
        result["convolutional_c_dw"] = width
    return result


# Feature-column layout and preprocessing constants. A variate is one input
# feature measured across time; raw indices refer to columns before selection.


def feature_names():
    """Name raw columns in their original order, including unused placeholders."""
    resource = files(__package__).joinpath("variate_names.json")
    named = {int(row["v"]): row["name"] for row in json.loads(resource.read_text(encoding="utf-8"))}
    if set(named) != set(range(159)):
        raise ValueError("Expected names for raw feature columns 0 through 158")
    return [f"v{i:03d}_{named.get(i, 'unused')}" for i in range(248)]


TOTAL_VARIATES = 141

# Daily preprocessing constants.
GAMMA_INDICES = [i * 8 + 3 for i in range(11)]
OPTIONS_GRID_END = 88
RELIABILITY_END = 99

# The 147 self-supervised learning (SSL) features in the supported v3 layout.
# Grid reliability (88-98) and unused raw column 140 are excluded.
ACTIVE_SSL_FEATURES = (
    list(range(0, 88)) +       # Options grid (mid, iv, theta, gamma, qliq, qimb, spread, spread_chg) x 11
    list(range(99, 105)) +     # Strike aggregates
    list(range(105, 119)) +    # SPX-derived features, including minutes to close
    list(range(119, 131)) +    # VIX term structure
    list(range(131, 140)) +    # Order flow
    list(range(141, 150)) +    # 3-bar trailing means of raw order-flow columns 131-139
    list(range(150, 159))      # 15-bar trailing means of the same order-flow columns
)
ACTIVE_N_SSL_FEATURES = len(ACTIVE_SSL_FEATURES)

# Heavy-tailed features compressed after feature selection, before z-scoring.
# prepare_input_splits maps these raw column numbers to selected-input
# positions and applies sign(x)*log1p(|x|), then fits normalization on training rows.
SSL_LOG1P_VARIATES = frozenset({
    111, 112, 113,  # Parkinson RV (15m, 30m, 60m) — heavy right tail during volatility spikes
    131,            # put/call liquidity ratio — unbounded ratio, right-skewed
})

# Raw columns compressed inside preprocess_day, before feature selection and
# normalization. Column 132 (net-premium flow) is already stored with signed log1p;
# applying it again preserves the preprocessing used by the recorded experiment.
PREPROCESS_LOG1P_INDICES = [101, 102, 104, 132, 139]


# Default training hyperparameters

# Used when the encoder receives no explicit hyperparameters.
SSL_HYPERPARAMS = dict(
    seq_len=60,
    d_model=128,
    n_heads=4,
    n_layers=3,
    d_ff=512,             # 4x d_model
    dropout=0.1,
    learning_rate=1e-4,
    batch_size=64,
    num_epochs=100,
    warmup_epochs=10,
    tokenizer="linear",
    patch_size=12,
    use_mask_indicator=True,
)


# Raw-column positions used by forward targets. ATM means at-the-money and
# IV means implied volatility.

RAW_SESSION_LOG_RET_INDEX = 105   # raw column 105: ln(close_t / close_open), for forward return delta
RAW_ATM_IV_INDEX = 41

# Advance 30 bars between input windows; consecutive 60-bar windows overlap.

WINDOW_STRIDE = 30

# Date boundaries from the recorded experiment; split preparation also applies embargoes.
TRAIN_START = "2022-09-19"
TRAIN_END = "2025-09-30"
VAL_START = "2025-10-08"
VAL_END = "2026-01-31"
TEST_START = "2026-02-08"
TEST_END = "2026-09-01"


# Training, inference, evaluation, and synthetic examples share this one setup.
# The name describes experiment settings; the input feature layout remains v3.
EXPERIMENT_PROFILE = dict(
    TRAIN_START=TRAIN_START, TRAIN_END=TRAIN_END,
    VAL_START=VAL_START, VAL_END=VAL_END,
    TEST_START=TEST_START, TEST_END=TEST_END,
    preprocess_log1p_indices=PREPROCESS_LOG1P_INDICES,
)
