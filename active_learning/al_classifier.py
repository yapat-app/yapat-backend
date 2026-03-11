from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiLabelMLPClassifier(nn.Module):
    """
    Multi-label MLP classifier with:
    - one hidden layer
    - dropout for regularization
    - prediction method returning both probabilities and binary predictions
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.n_dim = None
        self.num_classes = None
        self.hidden_dim = None
        self.dropout = None

    def create_classifier(
        self,
        n_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        """
        Create the classifier architecture.

        Parameters
        ----------
        n_dim : int
            Input embedding dimension.
        num_classes : int
            Number of species / labels.
        hidden_dim : int
            Size of the single hidden layer.
        dropout : float
            Dropout probability.
        """
        self.n_dim = n_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.model = nn.Sequential(
            nn.Linear(n_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return raw logits.
        """
        if self.model is None:
            raise ValueError("Classifier has not been created yet. Call create_classifier() first.")
        return self.model(x)

    def predict(
        self,
        x: torch.Tensor,
        threshold: Union[float, torch.Tensor] = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return both probabilities and binary predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, n_dim].
        threshold : float or torch.Tensor
            Either:
            - a single global threshold
            - a tensor of shape [num_classes] for per-class thresholds

        Returns
        -------
        probs : torch.Tensor
            Sigmoid probabilities of shape [batch_size, num_classes].
        preds : torch.Tensor
            Binary multi-label predictions of shape [batch_size, num_classes].
        """
        self.eval()

        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)

            if isinstance(threshold, (float, int)):
                preds = (probs >= threshold).int()
            else:
                threshold = threshold.to(probs.device)
                preds = (probs >= threshold).int()

        return probs, preds

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        device: str,
    ) -> Dict[str, float]:
        """
        Train the classifier on multi-label targets.
        """
        if self.model is None:
            raise ValueError("Classifier has not been created yet. Call create_classifier() first.")

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        pos_counts = y.sum(axis=0)
        neg_counts = y.shape[0] - pos_counts
        pos_weight = np.where(pos_counts > 0, neg_counts / np.maximum(pos_counts, 1), 1.0)
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        epoch_losses: List[float] = []

        for epoch in range(epochs):
            running_loss = 0.0
            num_batches = 0

            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                num_batches += 1

            avg_loss = running_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)
            logger.info(
                "Cold-start train epoch %d/%d - loss=%.6f",
                epoch + 1,
                epochs,
                avg_loss,
            )

        return {
            "final_train_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
            "best_train_loss": float(min(epoch_losses)) if epoch_losses else 0.0,
            "epochs": int(epochs),
        }

    def filter_and_balance_classes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        species_list: List[str],
        min_samples_per_class: int,
        max_samples_per_class: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, List[str], List[str], Dict[str, int]]:
        """
        Filter out under-supported classes and optionally cap samples per class.

        Notes
        -----
        In multi-label data, balancing is approximate because a single sample
        can belong to multiple classes.
        """
        class_support = y.sum(axis=0).astype(int)

        keep_class_indices = [
            i for i, count in enumerate(class_support)
            if count >= min_samples_per_class
        ]
        excluded_class_indices = [
            i for i, count in enumerate(class_support)
            if count < min_samples_per_class
        ]

        used_species = [species_list[i] for i in keep_class_indices]
        excluded_species = [species_list[i] for i in excluded_class_indices]

        if not keep_class_indices:
            return (
                np.empty((0, X.shape[1]), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
                [],
                excluded_species,
                {},
            )

        y = y[:, keep_class_indices]

        keep_rows = y.sum(axis=1) > 0
        X = X[keep_rows]
        y = y[keep_rows]

        if X.shape[0] == 0:
            return (
                np.empty((0, X.shape[1]), dtype=np.float32),
                np.empty((0, len(used_species)), dtype=np.float32),
                used_species,
                excluded_species,
                {},
            )

        if max_samples_per_class is not None:
            selected_indices: List[int] = []
            per_class_counts = np.zeros(y.shape[1], dtype=int)
            row_order = np.random.permutation(y.shape[0])

            for idx in row_order:
                labels = np.where(y[idx] > 0)[0]
                if len(labels) == 0:
                    continue

                if any(per_class_counts[c] < max_samples_per_class for c in labels):
                    selected_indices.append(idx)
                    for c in labels:
                        if per_class_counts[c] < max_samples_per_class:
                            per_class_counts[c] += 1

            if selected_indices:
                X = X[selected_indices]
                y = y[selected_indices]

        final_counts = y.sum(axis=0).astype(int)
        class_counts = {
            used_species[i]: int(final_counts[i])
            for i in range(len(used_species))
        }

        return X, y, used_species, excluded_species, class_counts

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = cls()
        model.create_classifier(
            n_dim=checkpoint["n_dim"],
            num_classes=checkpoint["num_classes"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        return model