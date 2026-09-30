"""Load saved weights and recover their tokenizer, seed, and training details.

New checkpoints include their training metadata. Original released checkpoints
are identified by file hashes recorded in results/experiment.json.
"""
import base64
import hashlib
import io
import json
from pathlib import Path

from .config import canonical_tokenizer, canonical_tokenizer_hyperparams

ROOT = Path(__file__).resolve().parents[3]


def load_checkpoint(path):
    """Read a trusted local checkpoint in released base64 or standard torch format.

    The original files contain numpy normalization arrays, so their deserialization
    requires the same unrestricted torch loader used by the experiment.
    """
    import torch

    raw = Path(path).read_bytes()
    # PyTorch files may be ZIP archives or pickle streams. Released checkpoints
    # instead wrap those bytes in base64 text; decoding does not change weights.
    if raw.startswith(b"PK\x03\x04") or raw.startswith(b"\x80"):
        blob = raw
    else:
        blob = base64.b64decode(b"".join(raw.split()), validate=True)
    checkpoint = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        # Normalize historical tokenizer labels in memory only. Preserve the
        # original files and state-dictionary keys used to reconstruct the model.
        if 'tokenizer' in checkpoint:
            checkpoint['tokenizer'] = canonical_tokenizer(checkpoint['tokenizer'])
        if isinstance(checkpoint.get('ssl_hyperparams'), dict):
            checkpoint['ssl_hyperparams'] = canonical_tokenizer_hyperparams(checkpoint['ssl_hyperparams'])
        if isinstance(checkpoint.get('training_metadata'), dict):
            metadata = checkpoint['training_metadata']
            if 'tokenizer' in metadata:
                metadata['tokenizer'] = canonical_tokenizer(metadata['tokenizer'])
    return checkpoint


def _experiment():
    """Read the released measurement archive when available alongside the source."""
    path = ROOT / 'results/experiment.json'
    if not path.is_file():
        return {}
    record = json.loads(path.read_text())
    if record.get('schema_version') != 1:
        raise ValueError(f'Unsupported original experiment record: {path}')
    return record


def _original_identity(path, experiment):
    """Match exact released file bytes to a tokenizer and seed in the archive."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    hashes = experiment.get('indicator_comparison', {}).get('checkpoint_sha256', {})
    matches = [key for key, value in hashes.items() if value == digest]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous original checkpoint hash: {path}')
    if matches:
        tokenizer, seed = matches[0].rsplit('_', 1)
        return dict(tokenizer=tokenizer, seed=seed)
    return {}


def _embedded_identity(path, checkpoint):
    """Combine checkpoint identities, rejecting conflicting embedded metadata."""
    if not isinstance(checkpoint, dict):
        raise ValueError(f'Expected a checkpoint mapping: {path}')
    metadata = checkpoint.get('training_metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError(f'Expected embedded training metadata to be a mapping: {path}')
    identity = {}
    # A run may record its identity both at the top level and inside training
    # metadata. Neither location should silently override a contradictory value.
    for source in (checkpoint, metadata):
        for key in ('tokenizer', 'seed'):
            if key in source:
                value = str(source[key])
                if key in identity and identity[key] != value:
                    raise ValueError(f'Conflicting checkpoint {key}: {path}')
                identity[key] = value
    return identity


def _validate_requested(path, identity, tokenizer, seed):
    """Check known identity fields against the requested run; allow missing fields."""
    for key, value in (('tokenizer', tokenizer), ('seed', seed)):
        if key in identity and str(identity[key]) != str(value):
            raise ValueError(f'Checkpoint {key} does not match {value}: {path}')


def training_metadata(path):
    """Return embedded training details or the exact original run's saved summary.

    A released file without embedded metadata must match an archived hash.
    Its date ranges come from the recorded corpus, not its filename.
    """
    checkpoint = load_checkpoint(path)
    embedded_identity = _embedded_identity(path, checkpoint)
    if 'training_metadata' in checkpoint:
        return checkpoint['training_metadata']
    # Original checkpoints predate embedded training summaries. Join the archive
    # by content identity so moving or renaming a checkpoint remains safe.
    experiment = _experiment()
    original = _original_identity(path, experiment)
    if not original:
        raise ValueError(f'No embedded training metadata or matching original checkpoint hash: {path}')
    _validate_requested(path, embedded_identity, original['tokenizer'], original['seed'])
    records = [record for record in experiment['training']['records']
               if record['tokenizer'] == original['tokenizer']
               and str(record['seed']) == original['seed']]
    if len(records) != 1:
        raise ValueError(f'Expected one original training summary for {path}; found {len(records)}')
    corpus = experiment['corpus']
    splits = corpus['splits']
    return dict(records[0],
                date_span=dict(train_start=corpus['date_range'][0], train_end=splits['train_end'],
                               val_start=splits['val'][0], val_end=splits['val'][1],
                               test_end=splits['test'][1]))
