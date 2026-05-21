import torch
import torch.nn as nn
from mil import JointMilModel

class SpeciesInstanceModel(nn.Module):

    def __init__(self, in_dim: int):
        super().__init__()
        self.mil = JointMilModel(in_dim=in_dim, n_classes=1, pooling="lin")

    def forward(self, emb_batch: torch.Tensor) -> torch.Tensor:
        # emb_batch: [B, in_dim]
        _, inst_logits = self.mil(emb_batch)      # [B, 1]
        return inst_logits.squeeze(-1)            # [B]

    def train_step(
        self,
        x: torch.Tensor,                  # [B, D] float32
        y: torch.Tensor,                  # [B] float32 in {0,1}
        optimizer: torch.optim.Optimizer,
        pos_weight: float | None = None,
    ):
        self.train()

        # --- ensure shapes/dtypes are right ---
        if y.dim() == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)
        y = y.to(dtype=torch.float32, device=x.device)

        logits = self(x)                  # [B]

        # pos_weight must be a scalar tensor on the right device
        if pos_weight is not None:
            pos_weight_t = torch.tensor(pos_weight, device=x.device, dtype=torch.float32)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == y).float().mean().item()

        return loss.item(), acc