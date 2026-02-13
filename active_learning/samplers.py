import torch
import numpy as np



def entropy(P: torch.Tensor) -> torch.Tensor:
    logP = torch.log(P + 1e-12)
    return -(P * logP).sum(dim=1)

def diversity_to_labeled(Z_u: torch.Tensor, Z_l: torch.Tensor) -> torch.Tensor:
    """
    d(i) = min_{l in L} ||z_i - z_l||, i over unlabeled.
    Z_u: [N_u, d], Z_l: [N_l, d]
    """
    if Z_l.numel() == 0 or Z_u.numel() == 0:
        return torch.zeros(Z_u.shape[0])
    dist = torch.cdist(Z_u, Z_l)              # [N_u, N_l]
    return dist.min(dim=1).values             # [N_u]

def knn_density(Z_u: torch.Tensor, k: int = 15) -> torch.Tensor:
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


