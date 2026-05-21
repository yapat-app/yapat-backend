"""
PyTorch model architectures for WSSED Active Learning

Simple linear classifier for embeddings-based active learning.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_2d_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalize binary and multi-class logits to [batch, classes]."""
    if logits.ndim == 1:
        return logits.unsqueeze(-1)
    return logits


def _load_state_dict(checkpoint_path: str, device: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


class SimpleLinearClassifier(nn.Module):
    """
    Single linear layer classifier (simplest model).
    
    Used when model files contain just "linear.weight" and "linear.bias".
    """
    
    def __init__(self, input_dim: int = 1024, num_classes: int = 1):
        """
        Args:
            input_dim: Dimension of input embeddings
            num_classes: Number of output classes (1 for binary, >1 for multi-class)
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_dim = input_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes)

    def create_classifier(
        self,
        n_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        dropout: float | None = None,
    ) -> None:
        self.input_dim = n_dim
        self.n_dim = n_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(n_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            logits: [batch_size] or [batch_size, num_classes] - raw logits
        """
        out = self.linear(x)
        if self.num_classes == 1:
            return out.squeeze(-1)  # [batch_size] for binary
        return out  # [batch_size, num_classes] for multi-class

    def predict(
        self,
        x: torch.Tensor,
        threshold: Union[float, torch.Tensor] = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            probs = torch.sigmoid(_as_2d_logits(self.forward(x)))
            if isinstance(threshold, (float, int)):
                preds = (probs >= float(threshold)).int()
            elif isinstance(threshold, torch.Tensor):
                preds = (probs >= threshold.to(probs.device)).int()
            else:
                raise TypeError(f"Unsupported threshold type: {type(threshold).__name__}")
        return probs, preds

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        state_dict = _load_state_dict(checkpoint_path, device)
        weight = state_dict.get("linear.weight")
        if weight is None:
            raise ValueError(f"Checkpoint does not contain linear.weight: {checkpoint_path}")

        model = cls(input_dim=int(weight.shape[1]), num_classes=int(weight.shape[0]))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    
    def train_step(self, x, y, optimizer, pos_weight=None):
        """
        Single training step.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            y: Labels [batch_size] (0 or 1 for binary, class index for multi-class)
            optimizer: PyTorch optimizer
            pos_weight: Weight for positive class (for imbalanced data, binary only)
            
        Returns:
            loss: Scalar loss value
            acc: Accuracy
        """
        self.train()
        optimizer.zero_grad()
        
        logits = self.forward(x)
        
        if self.num_classes == 1:
            # Binary classification
            if pos_weight is not None:
                pos_weight_tensor = torch.tensor([pos_weight], device=x.device)
                loss = F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight_tensor
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y)
            
            # Calculate accuracy
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == y).float().mean()
        else:
            # Multi-class classification
            loss = F.cross_entropy(logits, y.long())
            
            # Calculate accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y.long()).float().mean()
        
        loss.backward()
        optimizer.step()
        
        return loss.item(), acc.item()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        device: str,
    ) -> Dict[str, float]:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        pos_counts = y.sum(axis=0)
        neg_counts = y.shape[0] - pos_counts
        pos_weight = np.where(pos_counts > 0, neg_counts / np.maximum(pos_counts, 1), 1.0)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        epoch_losses: List[float] = []
        for _ in range(epochs):
            running_loss = 0.0
            num_batches = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = _as_2d_logits(self.forward(xb))
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                num_batches += 1
            epoch_losses.append(running_loss / max(num_batches, 1))

        return {
            "final_train_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
            "best_train_loss": float(min(epoch_losses)) if epoch_losses else 0.0,
            "epochs": int(epochs),
        }

    def filter_and_balance_classes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        snippet_ids: List[int],
        species_list: List[str],
        min_samples_per_class: int,
        max_samples_per_class: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, List[int], List[str], List[str], Dict[str, int]]:
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
                [],
                excluded_species,
                {},
            )

        y = y[:, keep_class_indices]
        keep_rows = y.sum(axis=1) > 0
        X = X[keep_rows]
        y = y[keep_rows]
        snippet_ids = [sid for sid, keep in zip(snippet_ids, keep_rows) if keep]

        if max_samples_per_class is not None and X.shape[0] > 0:
            selected_indices: List[int] = []
            per_class_counts = np.zeros(y.shape[1], dtype=int)
            for idx in np.random.permutation(y.shape[0]):
                labels = np.where(y[idx] > 0)[0]
                if any(per_class_counts[c] < max_samples_per_class for c in labels):
                    selected_indices.append(idx)
                    for c in labels:
                        if per_class_counts[c] < max_samples_per_class:
                            per_class_counts[c] += 1
            if selected_indices:
                X = X[selected_indices]
                y = y[selected_indices]
                snippet_ids = [snippet_ids[i] for i in selected_indices]

        final_counts = y.sum(axis=0).astype(int) if y.size else np.zeros(len(used_species), dtype=int)
        class_counts = {
            used_species[i]: int(final_counts[i])
            for i in range(len(used_species))
        }
        return X, y, snippet_ids, used_species, excluded_species, class_counts


class LinearClassifier(nn.Module):
    """
    Simple linear classifier for embedding-based active learning.
    
    Takes pre-computed embeddings as input and outputs binary classification.
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256):
        """
        Args:
            input_dim: Dimension of input embeddings (default: 1024 for BirdNET)
            hidden_dim: Dimension of hidden layer
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple 2-layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary classification
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            logits: [batch_size] - raw logits (use sigmoid for probabilities)
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # [batch_size]

    def predict(
        self,
        x: torch.Tensor,
        threshold: Union[float, torch.Tensor] = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            probs = torch.sigmoid(_as_2d_logits(self.forward(x)))
            if isinstance(threshold, (float, int)):
                preds = (probs >= float(threshold)).int()
            elif isinstance(threshold, torch.Tensor):
                preds = (probs >= threshold.to(probs.device)).int()
            else:
                raise TypeError(f"Unsupported threshold type: {type(threshold).__name__}")
        return probs, preds

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.relu(self.fc1(x))
    
    def train_step(self, x, y, optimizer, pos_weight=None):
        """
        Single training step.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            y: Labels [batch_size] (0 or 1)
            optimizer: PyTorch optimizer
            pos_weight: Weight for positive class (for imbalanced data)
            
        Returns:
            loss: Scalar loss value
            acc: Accuracy
        """
        self.train()
        optimizer.zero_grad()
        
        logits = self.forward(x)
        
        # Binary cross-entropy with logits
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], device=x.device)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight_tensor
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y)
        
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean()
        
        return loss.item(), acc.item()


class DeepClassifier(nn.Module):
    """
    Deeper classifier with batch normalization for more complex patterns.
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dims: list = [512, 256, 128]):
        """
        Args:
            input_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.input_dim = input_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            logits: [batch_size] - raw logits
        """
        return self.model(x).squeeze(-1)

    def predict(
        self,
        x: torch.Tensor,
        threshold: Union[float, torch.Tensor] = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            probs = torch.sigmoid(_as_2d_logits(self.forward(x)))
            if isinstance(threshold, (float, int)):
                preds = (probs >= float(threshold)).int()
            elif isinstance(threshold, torch.Tensor):
                preds = (probs >= threshold.to(probs.device)).int()
            else:
                raise TypeError(f"Unsupported threshold type: {type(threshold).__name__}")
        return probs, preds

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            for layer in list(self.model.children())[:-1]:
                x = layer(x)
            return x
    
    def train_step(self, x, y, optimizer, pos_weight=None):
        """Single training step."""
        self.train()
        optimizer.zero_grad()
        
        logits = self.forward(x)
        
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], device=x.device)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight_tensor
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean()
        
        return loss.item(), acc.item()


def create_model(input_dim: int = 1024, model_type: str = "linear") -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        input_dim: Dimension of input embeddings
        model_type: "simple", "linear", or "deep"
        
    Returns:
        PyTorch model
    """
    if model_type == "simple":
        return SimpleLinearClassifier(input_dim=input_dim)
    elif model_type == "linear":
        return LinearClassifier(input_dim=input_dim)
    elif model_type == "deep":
        return DeepClassifier(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
