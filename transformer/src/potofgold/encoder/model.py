"""Encode each input feature's time history as a variate token.

Shapes use B for batch size, T for bars per window, V for input features, and D
for embedding dimensions. Tokenizers map (B, V, T) to (B, V, D); encoder blocks
then attend across variates. Normalization of raw inputs happens outside this module.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import SSL_HYPERPARAMS, ACTIVE_N_SSL_FEATURES, canonical_tokenizer_hyperparams


class iTransformerEncoderBlock(nn.Module):
    """Pre-norm self-attention and feed-forward block over variate tokens."""
    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        z2 = self.norm1(z)
        a, w = self.attn(z2, z2, z2, need_weights=True)
        z = z + self.drop(a)
        z = z + self.ffn(self.norm2(z))
        return z, w


class ConcatProjectPatchTokenizer(nn.Module):
    """Project non-overlapping patches, add learned positions, then concatenate and project.

    Each variate produces one D-dimensional token from its full T-bar history.
    Patch projection and final readout weights are shared across variates.
    """
    def __init__(self, seq_len, d_model, patch_size=12, dropout=0.2):
        super().__init__()
        assert seq_len % patch_size == 0, \
            f"seq_len={seq_len} must be divisible by patch_size={patch_size}"
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size

        # Per-patch projection.
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.patch_norm = nn.LayerNorm(d_model)

        # Learned patch positions.
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.1)

        # Project concatenated patches to one token.
        self.concat_proj = nn.Linear(self.n_patches * d_model, d_model)
        self.concat_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_vt):
        """Map (B, V, T) input to (B, V, D) variate tokens."""
        B, V, _ = x_vt.shape
        # Reshape into patches: (B*V, n_patches, patch_size)
        x = x_vt.reshape(B * V, self.n_patches, self.patch_size)

        # Project each patch: (B*V, n_patches, d_model)
        z = self.patch_proj(x)
        z = self.patch_norm(z)
        z = torch.nn.functional.gelu(z)

        # Learned positions preserve patch order in the concatenated readout.
        z = z + self.pos_embed

        # Concatenate all patches: (B*V, n_patches * d_model) -> (B*V, d_model)
        z = z.reshape(B * V, self.n_patches * z.shape[-1])
        z = self.concat_proj(z)
        z = self.concat_norm(z)
        z = self.drop(z)

        return z.reshape(B, V, -1)  # (B, V, D)


class DWSepMixTokenizer(nn.Module):
    """Apply per-variate temporal kernels, a full-window readout, and residual variate mixing.

    Maps (B, V, T) to (B, V, D). Each variate has c_dw learned temporal kernels;
    a shared linear readout combines their c_dw*T responses. A 1x1 convolution
    then mixes variates independently at each embedding coordinate. Its zero
    initialization preserves the readout before the final normalization.
    """
    def __init__(self, n_variates, seq_len, d_model, c_dw=4, dropout=0.2):
        super().__init__()
        self.c = c_dw
        self.dw = nn.Conv1d(n_variates, n_variates * c_dw, kernel_size=5,
                            padding=2, groups=n_variates)   # per-variate temporal kernels
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.read = nn.Linear(c_dw * seq_len, d_model)      # Read out all temporal positions.
        self.read_norm = nn.LayerNorm(d_model)
        self.mix = nn.Conv1d(n_variates, n_variates, kernel_size=1)  # 1x1 cross-variate mix
        self.mix_norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.mix.weight)                     # Start the residual mixer at zero.
        nn.init.zeros_(self.mix.bias)                       

    def forward(self, x_vt):                                # (B, V, T)
        B, V, T = x_vt.shape
        # Apply the depthwise kernels with unfold and einsum for faster CPU execution.
        xu = nn.functional.pad(x_vt, (2, 2)).unfold(2, 5, 1)          # (B, V, T, 5) same-pad, k=5
        z = (torch.einsum('vck,bvtk->bvct', self.dw.weight.view(V, self.c, 5), xu)
             + self.dw.bias.view(V, self.c)[None, :, :, None])       # (B, V, c, T)
        z = self.drop(self.act(z.reshape(B, V * self.c, T)))         # (B, V*c, T) Conv1d-output order
        z = self.read_norm(self.read(z.reshape(B, V, self.c * T)))   # (B, V, D)   full-T readout
        # Apply the 1x1 variate convolution as a matrix multiplication.
        z = z + (torch.einsum('ow,bwd->bod', self.mix.weight[:, :, 0], z) + self.mix.bias[:, None])
        return self.mix_norm(z)


class iTransformerEncoder(nn.Module):
    """Encode preprocessed (B, T, V) windows as variate tokens.

    Linear tokenization projects each feature's full T-bar history to D values.
    Patch tokenization projects and concatenates temporal patches. Convolutional
    tokenization combines temporal kernels with a full-window readout and variate
    mixing. All three then add learned variate identities and use the same
    attention blocks across variates. The caller normalizes inputs and fills masked cells.
    """
    def __init__(self, hp=None, n_variates=None):
        super().__init__()
        p = canonical_tokenizer_hyperparams(hp or SSL_HYPERPARAMS)
        V = n_variates or ACTIVE_N_SSL_FEATURES
        T = p["seq_len"]
        D = p["d_model"]
        # Classifier and pooled-projection dimensions are stored in checkpoints.
        R = p.get("num_regimes", 4)
        E = p.get("embedding_dim", 32)

        # The mask indicator distinguishes hidden cells from observed zeros.
        self.use_mask_indicator = p.get("use_mask_indicator", False)
        if self.use_mask_indicator:
            self.mask_proj = nn.Linear(T, D)
            # The paired-window trainer passes no mask, so only the bias contributes.
            print(f"  Mask-indicator module: ENABLED ({T*D + D:,} params)")

        if p.get("use_temporal_attn", False):
            raise ValueError("The released encoders require use_temporal_attn=False.")

        # Select the temporal tokenizer.
        tok_type = p.get("tokenizer", "linear")
        if tok_type == "patch":
            patch_size = p.get("patch_size", 12)  
            self.tok = ConcatProjectPatchTokenizer(T, D, patch_size, p["dropout"])
        elif tok_type == "convolutional":
            # This tokenizer handles temporal aggregation internally.
            self.tok = DWSepMixTokenizer(V, T, D, c_dw=p.get("convolutional_c_dw", 4), dropout=p["dropout"])
        else:
            # One shared projection maps each variate's complete history to a token.
            self.tok = nn.Linear(T, D)
        print(f"  Tokenizer: {tok_type}")

        # Learned variate identity.
        self.variate_embed = nn.Parameter(torch.randn(V, D) * 0.02)

        # Post-embedding dropout.
        self.embed_drop = nn.Dropout(p["dropout"])

        self.blocks = nn.ModuleList([
            iTransformerEncoderBlock(D, p["n_heads"], p["d_ff"], p["dropout"])
            for _ in range(p["n_layers"])
        ])

        # Normalize encoder outputs before pooling or exporting variate tokens.
        self.final_norm = nn.LayerNorm(D)

        # Preserve these checkpoint parameters and their initialization order.
        # The paired-window objective uses mean-pooled variate tokens instead.
        self.pool_q = nn.Parameter(torch.randn(1, 1, D))
        self.pool_attn = nn.MultiheadAttention(D, num_heads=1, batch_first=True)
        self.regime_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.ReLU(), nn.Dropout(p["dropout"]),
            nn.Linear(D // 2, R),
        )
        self.emb_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.ReLU(), nn.Linear(D // 2, E),
        )

    def forward(self, x, mask=None):
        """Return logits, probabilities, pooled embeddings, and attention weights.
        
        x is (B, T, V), already z-scored and filled at masked cells by the caller.
        mask is an optional (B, T, V) mask with True for hidden cells. If the mask
        indicator is enabled, a missing mask uses zeros and retains mask_proj.bias.
        """
        z, attn_weights_all = self.encode_variates(x, return_attn=True, mask=mask)

        # One learned query pools all variate tokens into a window vector.
        pq = self.pool_q.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool_attn(pq, z, z)
        pooled = pooled.squeeze(1)

        logits = self.regime_head(pooled)
        probs = torch.softmax(logits, dim=-1)
        emb = self.emb_head(pooled)
        return logits, probs, emb, attn_weights_all

    def encode_variates(self, x, multi_layer=False, return_attn=False, mask=None):
        """Return per-variate embeddings from (B, T, V) preprocessed input.
        
        The caller fills masked cells and may pass a (B, T, V) mask (True = hidden).
        A missing mask uses zeros when the mask indicator is enabled.
        With multi_layer=True, concatenate layer outputs to (B, V, D*n_layers);
        otherwise return (B, V, D). return_attn also returns one (B, V, V) attention
        matrix per layer, averaged over heads.
        """
        x_vt = x.transpose(1, 2)                  # (B, V, T)
        z = self.tok(x_vt)                               # (B, V, D)
        if self.use_mask_indicator:
            if mask is None:
                # Use a zero mask so the projection bias still contributes.
                mask = torch.zeros(x.shape[0], x.shape[1], x.shape[2],
                                    dtype=torch.float32, device=x.device)
            else:
                # Shape contract: mask must be (B, T, V) bool/float.
                assert mask.shape == x.shape, \
                    f"mask shape {tuple(mask.shape)} must equal input x shape {tuple(x.shape)}"
            z = z + self.mask_proj(mask.transpose(1, 2).float())
        z = z + self.variate_embed.unsqueeze(0)
        z = self.embed_drop(z)
        attn_weights_all = [] if return_attn else None
        if multi_layer:
            layer_outputs = []
            for block in self.blocks:
                z, w = block(z)
                if return_attn:
                    attn_weights_all.append(w)
                # Normalize saved outputs without changing the next block's input.
                layer_outputs.append(self.final_norm(z))
            out = torch.cat(layer_outputs, dim=-1)  # (B, V, D*n_layers)
            return (out, attn_weights_all) if return_attn else out
        for block in self.blocks:
            z, w = block(z)
            if return_attn:
                attn_weights_all.append(w)
        z = self.final_norm(z)
        return (z, attn_weights_all) if return_attn else z


def _decay_split(modules):
    """Split module parameters into decay and no-decay groups.
    
    Exclude normalization parameters, biases, and embeddings. Identify
    normalization layers by type so unnamed Sequential layers are covered.
    """
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)
    decay, no_decay, seen = [], [], set()
    for top in modules:
        for m in top.modules():
            is_norm = isinstance(m, norm_types)
            is_embed = isinstance(m, nn.Embedding)
            # Inspect parameters with their owning layer; shared tensors are added once.
            for pname, prm in m.named_parameters(recurse=False):
                if not prm.requires_grad or id(prm) in seen:
                    continue
                seen.add(id(prm))
                exclude = (is_norm or is_embed or pname.endswith("bias")
                           or "embed" in pname)
                (no_decay if exclude else decay).append(prm)
    return decay, no_decay
