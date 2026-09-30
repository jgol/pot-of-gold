"""Future-latent prediction with variance, covariance, and indicator-decorrelation losses.

The caller encodes paired future windows, residualizes their teacher embeddings,
and updates the teacher by exponential moving average (EMA). Prediction targets
are detached. A gradient-reversal adversary learns to predict indicator variables
while reversing only its gradient to the student encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .anticollapse import vicreg_var_cov_penalty


# EMA teacher update
@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """Move teacher parameters and floating buffers toward the student by EMA.
    
    Copy non-floating buffers directly. decay controls the teacher weight retained.
    """
    d = float(decay)
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.mul_(d).add_(ps.detach(), alpha=1.0 - d)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        if bt.dtype.is_floating_point:
            bt.mul_(d).add_(bs.detach().to(bt.dtype), alpha=1.0 - d)
        else:
            bt.copy_(bs)                                   # int buffers (num_batches_tracked): copy


# Gradient reversal
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)                                # identity on the forward pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None         # reversed (and scaled) gradient to the input


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Pass x through unchanged; multiply its backward gradient by -lambd.
    
    Adversary parameters receive normal gradients; only the input gradient reverses.
    """
    return _GradReverse.apply(x, lambd)


# Remove the teacher target's linear dependence on indicator variables.
def batch_residualize(target: torch.Tensor, scalars: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Remove a ridge-fitted indicator projection from (B, D) targets.

    scalars has shape (B, S); a constant column adds an intercept. The ridge
    penalty applies to all fitted coefficients, including that intercept.
    Fit and subtract on the same rows. The trainer does this separately for
    every training and validation batch, making validation batch-dependent.
    Evaluation probes instead fit residualization on training data only.
    Non-finite scalars become zero. Rows with non-finite targets are excluded
    from fitting and returned unchanged.
    """
    B = target.shape[0]
    ridge = max(float(ridge), 1e-8)
    s = torch.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)
    Z = torch.cat([torch.ones(B, 1, dtype=target.dtype, device=target.device), s], dim=1)  # (B, 1+S)
    row_ok = torch.isfinite(target).all(dim=-1)                                             # (B,)
    Zf, tf = Z[row_ok], target[row_ok]
    eye = torch.eye(Z.shape[1], dtype=target.dtype, device=target.device)
    if Zf.shape[0] >= 1:
        # Ridge stabilizes the solve when scalar columns are linearly dependent.
        beta = torch.linalg.solve(Zf.transpose(0, 1) @ Zf + ridge * eye, Zf.transpose(0, 1) @ tf)  # (1+S, D)
        out = target - Z @ beta
    else:
        out = target
    return torch.where(row_ok.unsqueeze(-1), out, target)   # non-finite target rows returned unchanged


# loss terms
def latent_predict_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None,
                        huber_delta: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked Huber loss and the number of usable (B, D) rows.

    Sum errors over embedding dimensions, then average valid rows. A nonpositive
    huber_delta selects squared error. Detach targets and exclude non-finite
    prediction or target rows. An all-invalid batch returns an autograd-connected zero.
    """
    t = target.detach()
    finite = torch.isfinite(t).all(dim=-1) & torch.isfinite(pred).all(dim=-1)   # (B,) usable rows
    # Sanitize before computing errors: multiplying NaN by a zero mask is still NaN.
    ps = torch.nan_to_num(pred)
    resid = ps - torch.nan_to_num(t)
    d = float(huber_delta)
    per = (torch.where(resid.abs() <= d, 0.5 * resid ** 2, d * (resid.abs() - 0.5 * d)).sum(dim=-1)
           if d > 0 else (resid ** 2).sum(dim=-1))         # (B,)
    per = torch.nan_to_num(per, nan=0.0)
    vb = finite if valid is None else \
        (valid.bool() if valid.dtype == torch.bool else (valid > 0.5)) & finite
    n = vb.float().sum()
    loss = (per * vb.float()).sum() / n if float(n) > 0 else (per * vb.float()).sum()
    return loss, n


def linear_decorr_loss(z: torch.Tensor, scalars: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Return mean squared correlation between (B, D) embeddings and (B, S) scalars.
    
    Replace NaNs and infinities with finite values. Batches smaller than two return zero.
    """
    if z.shape[0] < 2:
        return z.sum() * 0.0
    zz = torch.nan_to_num(z)
    s = torch.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)
    zc = zz - zz.mean(dim=0, keepdim=True)
    sc = s - s.mean(dim=0, keepdim=True)
    zn = zc / zc.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    sn = sc / sc.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    cross = (zn.transpose(0, 1) @ sn) / z.shape[0]         # (D, S) Pearson correlation (population)
    return (cross ** 2).mean()


def _standardize(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Z-score each column using detached statistics after replacing NaNs and infinities."""
    xz = torch.nan_to_num(x)
    # Keep gradients through values, but not through the batch statistics.
    m = xz.mean(dim=0, keepdim=True).detach()
    sd = xz.std(dim=0, unbiased=False, keepdim=True).detach().clamp_min(eps)
    return (xz - m) / sd


# trainable heads
class _MLP(nn.Module):
    """MLP with layer normalization, supporting single-sample batches."""
    def __init__(self, d_in, d_out, hidden=384, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class PredictionHeads(nn.Module):
    """Pool student tokens and predict future latents and indicator variables.

    Mean-pool (B, V, D) tokens to (B, D). The adversary predicts (B, S) indicator values;
    SelfSupervisedLoss reverses its gradient to the encoder.
    """
    def __init__(self, d_model: int, n_scalars: int, hidden: int = 384, dropout: float = 0.1):
        super().__init__()
        assert int(n_scalars) >= 1, ("The indicator adversary requires at least one input variable; "
                                     f"got n_scalars={n_scalars}")
        self.predictor = _MLP(d_model, d_model, hidden=hidden, dropout=dropout)
        self.adversary = _MLP(d_model, n_scalars, hidden=hidden, dropout=dropout)

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:      # (B, V, D) -> (B, D)
        return tokens.mean(dim=1)


class SelfSupervisedLoss(nn.Module):
    """Combine latent prediction, variance, covariance, and indicator-decorrelation losses.

    student_tokens is (B, V, D); teacher_target is a detached, residualized (B, D)
    future latent supplied by the caller. scalars is (B, S). valid is (B,) and
    masks only the prediction term; regularizers use every supplied row.
    Trainable heads are optimized alongside the encoder. The adversarial loss
    has full weight for the adversary; adv_lambda scales the reversed encoder gradient.
    """
    def __init__(self, w_vicreg: float = 0.5, w_decorr: float = 0.2, adv_lambda: float = 0.1,
                 w_var: float = 25.0, w_cov: float = 1.0, vicreg_gamma: float = 1.0,
                 huber_delta: float = 1.0):
        super().__init__()
        self.w_vicreg, self.w_decorr = float(w_vicreg), float(w_decorr)
        self.adv_lambda = float(adv_lambda)                # Encoder-side adversarial weight
        self.w_var, self.w_cov = float(w_var), float(w_cov)
        self.vicreg_gamma, self.huber_delta = float(vicreg_gamma), float(huber_delta)

    def forward(self, heads: PredictionHeads, student_tokens: torch.Tensor,
                teacher_target: torch.Tensor, scalars: torch.Tensor, valid: torch.Tensor):
        assert scalars.shape[-1] >= 1, "The indicator adversary requires at least one input variable; got S=0"
        pooled = heads.pool_tokens(student_tokens)                    # (B, D) — student window embedding
        pred = heads.predictor(pooled)                               # (B, D) — predicted future latent

        l_pred, n_eff = latent_predict_loss(pred, teacher_target, valid, huber_delta=self.huber_delta)

        # Normalize by the mean embedding standard deviation, holding that
        # scale fixed in backpropagation. The hinge discourages small dimension
        # variances; the covariance term discourages redundant directions.
        scale = pooled.detach().std(dim=0).mean().clamp_min(1e-4)
        var, cov = vicreg_var_cov_penalty(pooled / scale, gamma=self.vicreg_gamma)
        l_vicreg = self.w_var * var + self.w_cov * cov
        l_decorr = self.w_decorr * linear_decorr_loss(pooled, scalars)

        # Train the indicator adversary at full weight; adv_lambda scales only
        # the reversed gradient to the encoder.
        s_std = _standardize(scalars)
        adv_pred = heads.adversary(grad_reverse(pooled, self.adv_lambda))
        resid = adv_pred - s_std
        d = self.huber_delta
        l_adv = (torch.where(resid.abs() <= d, 0.5 * resid ** 2, d * (resid.abs() - 0.5 * d))
                 if d > 0 else resid ** 2).mean()

        loss = l_pred + self.w_vicreg * l_vicreg + l_decorr + l_adv
        info = {"l_pred": float(l_pred.detach()), "l_vicreg": float(l_vicreg.detach()),
                "l_var": float(var.detach()), "l_cov": float(cov.detach()),
                "l_decorr": float(l_decorr.detach()), "l_adv": float(l_adv.detach()),
                "loss": float(loss.detach()), "n_effective": float(n_eff.detach()),
                "n_batch": int(student_tokens.shape[0])}
        return loss, info
