"""
Species Model Store for Active Learning

Handles loading and managing species-specific PyTorch models.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SpeciesModelStore:
    """
    Manages loading and caching of species-specific PyTorch models.
    """

    def __init__(self):
        self._cache: Dict[str, nn.Module] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SpeciesModelStore initialized with device: {self.device}")

    def get_model_path(
        self,
        model_directory: str,
        metric_type: str = "macro",
        prediction_level: str = "segment"
    ) -> Path:
        """
        Construct the full path to the model file.

        Args:
            model_directory: Base directory containing the models
            metric_type: "macro" or "micro"
            prediction_level: "segment" or "clip"

        Returns:
            Path object pointing to the model file
        """
        # Construct filename: best_{metric}_model[_segment].pt
        filename = f"best_{metric_type}_model"
        if prediction_level == "segment":
            filename += "_segment"
        filename += ".pt"

        model_path = Path(model_directory) / filename
        return model_path

    def load_model(
        self,
        model_directory: str,
        metric_type: str = "macro",
        prediction_level: str = "segment",
        force_reload: bool = False
    ) -> nn.Module:
        """
        Load a species model from disk.

        Args:
            model_directory: Base directory containing the models
            metric_type: "macro" or "micro"
            prediction_level: "segment" or "clip"
            force_reload: Force reloading even if cached

        Returns:
            Loaded PyTorch model

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        cache_key = f"{model_directory}_{metric_type}_{prediction_level}"

        # Return cached model if available
        if not force_reload and cache_key in self._cache:
            logger.info(f"Returning cached model for {cache_key}")
            return self._cache[cache_key]

        # Get model path
        model_path = self.get_model_path(model_directory, metric_type, prediction_level)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Expected structure: {model_directory}/best_{metric_type}_model{'_segment' if prediction_level == 'segment' else ''}.pt"
            )

        try:
            logger.info(f"Loading model from {model_path}")
            
            # Load the model weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # The checkpoint might be just weights or a dict with 'model_state_dict'
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Create model instance
            # Note: This assumes the model architecture is available
            # You'll need to import or define the actual model class
            model = self._create_model_instance(state_dict)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            # Cache the model
            self._cache[cache_key] = model
            logger.info(f"Model loaded and cached successfully: {cache_key}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _create_model_instance(self, state_dict: Dict) -> nn.Module:
        """
        Create a model instance based on the state dict.

        This infers the model architecture from the state dict keys.
        
        Args:
            state_dict: Model state dictionary

        Returns:
            Model instance
        """
        from app.models.wssed_pytorch_models import SimpleLinearClassifier, LinearClassifier, DeepClassifier
        
        # Infer input dimension, output dimension, and model type from state dict keys
        first_layer_key = None
        input_dim = None
        num_classes = 1  # Default to binary classification
        
        # Check for different model architectures
        if any('linear.weight' in key for key in state_dict.keys()):
            # SimpleLinearClassifier (single linear layer)
            first_layer_key = 'linear.weight'
            logger.info("Detected SimpleLinearClassifier architecture")
        elif any('fc1.weight' in key for key in state_dict.keys()):
            # LinearClassifier (2-layer MLP)
            first_layer_key = 'fc1.weight'
            logger.info("Detected LinearClassifier architecture")
        elif any('model.0.weight' in key for key in state_dict.keys()):
            # DeepClassifier (Sequential model)
            first_layer_key = 'model.0.weight'
            logger.info("Detected DeepClassifier architecture")
        
        if first_layer_key and first_layer_key in state_dict:
            # Get input and output dimensions from weight shape
            weight_shape = state_dict[first_layer_key].shape
            input_dim = weight_shape[1]  # [out_features, in_features]
            
            # For SimpleLinearClassifier, also get num_classes
            if first_layer_key == 'linear.weight':
                num_classes = weight_shape[0]  # output dimension
                logger.info(f"Inferred num_classes: {num_classes}")
            
            logger.info(f"Inferred input dimension: {input_dim}")
        else:
            # Fallback: assume BirdNET embedding dimension
            logger.warning("Could not infer input dimension from state dict, using default 1024")
            input_dim = 1024
        
        # Create appropriate model based on detected architecture
        if any('linear.weight' in key for key in state_dict.keys()):
            model = SimpleLinearClassifier(input_dim=input_dim, num_classes=num_classes)
            logger.info(f"Created SimpleLinearClassifier with input_dim={input_dim}, num_classes={num_classes}")
        elif any('model.0.weight' in key for key in state_dict.keys()):
            model = DeepClassifier(input_dim=input_dim)
            logger.info(f"Created DeepClassifier with input_dim={input_dim}")
        else:
            # Default to LinearClassifier (2-layer MLP)
            model = LinearClassifier(input_dim=input_dim)
            logger.info(f"Created LinearClassifier with input_dim={input_dim}")
        
        return model

    def clear_cache(self, model_directory: Optional[str] = None):
        """
        Clear cached models.

        Args:
            model_directory: If provided, clear only models from this directory.
                           If None, clear all cached models.
        """
        if model_directory is None:
            self._cache.clear()
            logger.info("Cleared all cached models")
        else:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(model_directory)]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared {len(keys_to_remove)} cached models from {model_directory}")

    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cached models.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "num_cached_models": len(self._cache),
            "cached_keys": list(self._cache.keys())
        }


# Global instance
_species_model_store: Optional[SpeciesModelStore] = None


def get_species_model_store() -> SpeciesModelStore:
    """
    Get or create the global SpeciesModelStore instance.

    Returns:
        SpeciesModelStore instance
    """
    global _species_model_store
    if _species_model_store is None:
        _species_model_store = SpeciesModelStore()
    return _species_model_store
