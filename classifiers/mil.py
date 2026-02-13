from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

class JointMilModel(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, pooling: str = "lin") -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
        self.pooling = pooling

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # embeddings: [n_instances, in_dim] OR [B, in_dim] -> treat B as n_instances
        inst_logits = self.linear(embeddings)  # [n_instances, n_classes]
        inst_probs = torch.sigmoid(inst_logits)

        if self.pooling == "max":
            bag_probs = torch.max(inst_probs, dim=0).values
        elif self.pooling == "avg":
            bag_probs = torch.mean(inst_probs, dim=0)
        elif self.pooling == "exp":
            numerator = torch.sum(inst_probs * torch.exp(inst_probs), dim=0)
            denominator = torch.sum(torch.exp(inst_probs), dim=0) + 1e-8
            bag_probs = numerator / denominator
        else:  # lin
            numerator = torch.sum(inst_probs * inst_probs, dim=0)
            denominator = torch.sum(inst_probs, dim=0) + 1e-8
            bag_probs = numerator / denominator

        bag_logits = torch.logit(torch.clamp(bag_probs, min=1e-6, max=1 - 1e-6))
        return bag_logits, inst_logits


