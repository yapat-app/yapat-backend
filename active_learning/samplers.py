import torch
import numpy as np



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


