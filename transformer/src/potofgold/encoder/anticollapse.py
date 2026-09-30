"""Measure or penalize collapsed embedding directions."""
import torch


def compute_participation_ratio(embeddings, eps=1e-8):
    """Measure covariance participation ratio: (sum(eigenvalues))² / sum(eigenvalues²).

    embeddings is (N, D). Center samples before forming covariance. The ratio
    describes how broadly variance is distributed across embedding directions;
    training uses it as a checkpoint-selection threshold. It is distinct from
    the RankMe effective-rank metric reported during evaluation.
    Clamp negative eigenvalues from numerical error to zero; eps stabilizes
    the denominator for constant embeddings.
    """
    X = embeddings - embeddings.mean(dim=0, keepdim=True)
    n = max(X.shape[0] - 1, 1)
    cov = (X.transpose(0, 1) @ X) / n
    lam = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    s = lam.sum()
    return (s * s) / (lam.pow(2).sum() + eps)


def vicreg_var_cov_penalty(z, gamma: float = 1.0, eps: float = 1e-4):
    """Penalize small dimension variances and off-diagonal covariance in (N, D) embeddings.

    Return (var_loss, cov_loss). The hinge applies a minimum standard-deviation
    target gamma to each dimension; the covariance term discourages redundant
    directions. The variance estimate divides by N and includes eps under the
    square root. Covariance divides by N-1; its squared off-diagonal entries
    sum and divide by D. Batches smaller than two return autograd-connected zeros.
    """
    if z.shape[0] < 2:                                            # Need two samples for covariance
        zero = z.sum() * 0.0                                      # Zero connected to autograd
        return zero, zero
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)          # (D,)
    var_loss = torch.relu(gamma - std).mean()
    n = max(z.shape[0] - 1, 1)
    cov = (z.transpose(0, 1) @ z) / n                             # (D, D)
    d = z.shape[1]
    # Leave each dimension's own variance to the hinge term above.
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / d
    return var_loss, cov_loss
