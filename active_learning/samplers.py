import torch
import numpy as np
import torch.nn.functional as F

from app.schemas.pam_active_learning import ALSingleSampleScore



def uncertainty(P: torch.Tensor) -> torch.Tensor:
    """
    Multi-label uncertainty: mean binary entropy across classes.
    P: [N, C] sigmoid probabilities
    """
    return -(P * torch.log(P + 1e-12) + (1 - P) * torch.log(1 - P + 1e-12)).mean(dim=1)

def diversity(Z_u: torch.Tensor, Z_l: torch.Tensor) -> torch.Tensor:
    """
    d(i) = min_{l in L} ||z_i - z_l||, i over unlabeled.
    Z_u: [N_u, d], Z_l: [N_l, d]
    """
    if Z_l.numel() == 0 or Z_u.numel() == 0:
        return torch.zeros(Z_u.shape[0])
    Z_u = F.normalize(Z_u, p=2, dim=1)
    Z_l = F.normalize(Z_l, p=2, dim=1)
    dist = torch.cdist(Z_u, Z_l)              # [N_u, N_l]
    return dist.min(dim=1).values             # [N_u]

def density(Z_u: torch.Tensor, k: int = 15) -> torch.Tensor:
    """
    Local density among unlabeled: rho(i) ≈ 1 / avg kNN distance.
    Z_u: [N_u, d]
    """
    N_u = Z_u.shape[0]
    if N_u <= 1:
        return torch.zeros(N_u)

    dist = torch.cdist(Z_u, Z_u)              # [N_u, N_u]
    # nearest k+1 (including self), drop self
    knn = dist.topk(k=min(k+1, N_u), largest=False).values[:, 1:]
    avg = knn.mean(dim=1)
    return 1.0 / (avg + 1e-8)

def random(n: int, device: str = "cpu") -> torch.Tensor:
    return torch.rand(n, device=device)


def composite(
    uncertainty_scores: torch.Tensor,
    diversity_scores: torch.Tensor,
    density_scores: torch.Tensor,
    wu: float = 0.5,
    wd: float = 0.25,
    wr: float = 0.25,
) -> torch.Tensor:
    def normalize(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        x_min = x.min()
        x_max = x.max()
        if torch.isclose(x_min, x_max):
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    u = normalize(uncertainty_scores)
    d = normalize(diversity_scores)
    r = normalize(density_scores)

    return wu * u + wd * d + wr * r

def calculate_single_sample_scores(
    probs_row: torch.Tensor,
    sample_embedding: torch.Tensor,
    unlabeled_embeddings: torch.Tensor,
    labeled_embeddings: torch.Tensor,
    density_k: int,
    wu: float,
    wd: float,
    wr: float,
) -> ALSingleSampleScore:
    """
    Compute all acquisition values for one sample.

    probs_row: [C]
    sample_embedding: [D]
    unlabeled_embeddings: [N_u, D]
    labeled_embeddings: [N_l, D]
    """
    u_score = uncertainty(probs_row.unsqueeze(0)).squeeze(0)

    if labeled_embeddings.numel() == 0:
        d_score = None
    else:
        d_score = torch.cdist(
            sample_embedding.unsqueeze(0),
            labeled_embeddings,
        ).min(dim=1).values.squeeze(0)

    if unlabeled_embeddings.shape[0] <= 1:
        rho_score = None
    else:
        dist = torch.cdist(
            sample_embedding.unsqueeze(0),
            unlabeled_embeddings,
        ).squeeze(0)

        # remove exact self distance if present
        sorted_vals = torch.sort(dist).values
        neigh = sorted_vals[1:min(density_k + 1, sorted_vals.shape[0])]
        if neigh.numel() == 0:
            rho_score = None
        else:
            rho_score = 1.0 / (neigh.mean() + 1e-8)

    if d_score is None or rho_score is None:
        c_score = None
    else:
        # single-sample composite without global normalization is not very meaningful;
        # use raw weighted sum here, mainly for convenience
        c_score = wu * u_score + wd * d_score + wr * rho_score

    return ALSingleSampleScore(
        uncertainty=float(u_score.item()),
        diversity=None if d_score is None else float(d_score.item()),
        density=None if rho_score is None else float(rho_score.item()),
        composite=None if c_score is None else float(c_score.item()),
    )


