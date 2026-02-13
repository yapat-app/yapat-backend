import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from samplers import entropy, diversity_to_labeled, knn_density

class _LabeledDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i], dtype=torch.float32)


class ActiveLearning:
    def __init__(self, X_pool: np.ndarray, Z_pool: np.ndarray | None = None):
        """
        X_pool: [N, D] embeddings used for the classifier training/inference
        Z_pool: [N, d] embeddings used for diversity/density (can be 2D UMAP or PCA)
        """
        self.X_pool = X_pool.astype(np.float32)
        self.Z_pool = None if Z_pool is None else Z_pool.astype(np.float32)

        # labels: -1 = unlabeled, 0 = reject, 1 = accept
        self.y = np.full((len(self.X_pool),), -1, dtype=np.int8)

    @torch.no_grad()
    def select_topk(
            strategy: str,
            k: int,
            is_labeled_np: np.ndarray,  # [N] bool
            p_np: np.ndarray | None = None,  # [N] prob present (needed for uncertainty)
            Z_np: np.ndarray | None = None,  # [N,d] embedding (needed for diversity/density)
            k_density: int = 15,
            seed: int = 0,
    ) -> list[int]:
        """
        Returns dataset indices of the top-k unlabeled samples according to strategy.

        strategy:
          - "uncertainty": uses entropy on P=[1-p, p]
          - "diversity": uses min distance to labeled points
          - "density": uses kNN density among unlabeled
          - "random": random unlabeled
        """
        is_labeled = torch.from_numpy(is_labeled_np.astype(bool))
        mask_u = ~is_labeled
        idx_u = torch.where(mask_u)[0]
        if idx_u.numel() == 0:
            return []

        if strategy == "uncertainty":
            if p_np is None:
                raise ValueError("p_np is required for strategy='uncertainty'")
            p = torch.from_numpy(p_np).float()  # [N]
            P = torch.stack([1.0 - p, p], dim=1)  # [N,2]
            scores = entropy(P)  # [N]
            scores_u = scores[mask_u]  # [N_u]

        elif strategy == "diversity":
            if Z_np is None:
                raise ValueError("Z_np is required for strategy='diversity'")
            Z = torch.from_numpy(Z_np).float()
            Z_u = Z[mask_u]
            Z_l = Z[is_labeled]
            scores_u = diversity_to_labeled(Z_u, Z_l)  # [N_u]

        elif strategy == "density":
            if Z_np is None:
                raise ValueError("Z_np is required for strategy='density'")
            Z = torch.from_numpy(Z_np).float()
            Z_u = Z[mask_u]
            scores_u = knn_density(Z_u, k=k_density)  # [N_u]

        elif strategy == "random":
            g = torch.Generator().manual_seed(int(seed))
            scores_u = torch.rand(idx_u.numel(), generator=g)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        k_eff = min(k, int(scores_u.numel()))
        top_rel = torch.topk(scores_u, k_eff, largest=True).indices  # indices into unlabeled
        chosen = idx_u[top_rel].cpu().numpy().tolist()  # indices into full dataset
        return chosen

    # ---------- inference / scoring ----------
    @torch.no_grad()
    def predict_probs(model, X_np, device="cpu", batch_size=2048):
        model.eval().to(device)
        out = []
        for i in range(0, len(X_np), batch_size):
            x = torch.from_numpy(X_np[i:i+batch_size]).to(device)
            p = torch.sigmoid(model(x)).detach().cpu().numpy()
            out.append(p)
        return np.concatenate(out, axis=0)

        # ---------- retraining ----------

    def retrain(
            self,
            model,
            device="cpu",
            epochs=5,
            lr=1e-3,
            batch_size=128,
            weight_decay=0.0,
    ) -> dict:
        labeled_idx = np.where(self.y != -1)[0]
        if len(labeled_idx) < 2:
            return {"status": "skip", "reason": "too_few_labels", "n_labeled": int(len(labeled_idx))}

        X = self.X_pool[labeled_idx]
        y = self.y[labeled_idx].astype(np.float32)

        # optional imbalance handling
        pos = float(y.sum())
        neg = float(len(y) - pos)
        pos_weight = (neg / max(pos, 1.0)) if pos > 0 else None

        ds = _LabeledDataset(X, y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model.train().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if pos_weight is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        last_loss = None
        for _ in range(epochs):
            tot, n = 0.0, 0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()

                tot += float(loss.item()) * xb.size(0)
                n += xb.size(0)
            last_loss = tot / max(n, 1)

        return {"status": "trained", "n_labeled": int(len(labeled_idx)), "loss": float(last_loss)}

    def apply_new_annotations(self, idx_to_label: dict[int, int]) -> int:
        """
        idx_to_label: {index: 0/1}
        returns: number of newly-labeled points added
        """
        added = 0
        for idx, lab in idx_to_label.items():
            lab = int(lab)
            if lab not in (0, 1):
                raise ValueError("label must be 0 or 1")
            if self.y[idx] == -1:
                added += 1
            self.y[idx] = lab
        return added

    # ---------- one active learning round ----------
    def step(
            self,
            model,
            strategy: str,
            k: int,
            device="cpu",
            retrain_every: int = 20,
            seed: int = 0,
            k_density: int = 15,
    ) -> dict:
        """
        Returns a dict with suggested indices and (optionally) probabilities.
        You call this to get suggestions; then you collect user labels and call apply_new_annotations + retrain.
        """
        is_labeled_np = self.is_labeled_mask()

        p_np = None
        Z_np = None

        if strategy == "uncertainty":
            p_np = self.predict_probs(model, self.X_pool, device=device)
        if strategy in ("diversity", "density"):
            if self.Z_pool is None:
                raise ValueError("Z_pool is required for diversity/density strategies")
            Z_np = self.Z_pool

        chosen = self.select_topk(
            strategy=strategy,
            k=k,
            is_labeled_np=is_labeled_np,
            p_np=p_np,
            Z_np=Z_np,
            k_density=k_density,
            seed=seed,
        )

        return {
            "chosen_indices": chosen,
            "probs": None if p_np is None else p_np[chosen].tolist(),
            "n_labeled": int(is_labeled_np.sum()),
            "n_unlabeled": int((~is_labeled_np).sum()),
        }

    def retrain(
            self,
            model,
            device="cpu",
            epochs=5,
            lr=1e-3,
            batch_size=128,
            weight_decay=0.0,
    ) -> dict:
        labeled_idx = np.where(self.y != -1)[0]
        if len(labeled_idx) < 2:
            return {"status": "skip", "reason": "too_few_labels", "n_labeled": int(len(labeled_idx))}

        X = self.X_pool[labeled_idx]
        y = self.y[labeled_idx].astype(np.float32)

        # imbalance handling
        pos = float(y.sum())
        neg = float(len(y) - pos)
        pos_weight = (neg / max(pos, 1.0)) if pos > 0 else None

        #ds = _LabeledDataset(X, y)
        #dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        last_loss, last_acc = None, None
        for _ in range(epochs):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)

                # <-- THIS is the integration point
                last_loss, last_acc = model.train_step(
                    xb, yb, optimizer=opt, pos_weight=pos_weight
                )

        return {
            "status": "trained",
            "n_labeled": int(len(labeled_idx)),
            "loss": float(last_loss) if last_loss is not None else None,
            "acc": float(last_acc) if last_acc is not None else None,
            "pos_weight": float(pos_weight) if pos_weight is not None else None,
        }
