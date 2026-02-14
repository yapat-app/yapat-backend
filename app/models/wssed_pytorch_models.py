"""
PyTorch model architectures for WSSED Active Learning

Simple linear classifier for embeddings-based active learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes)
        
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
