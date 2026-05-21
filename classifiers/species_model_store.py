import os
import torch
from single_class_classifier import SpeciesInstanceModel

class SpeciesModelStore:
    """
    Simple: one folder, one file per species name.
    base checkpoint is copied once to create species checkpoints.
    """
    def __init__(self, root="/app/assets/models/single_species_models/species",
                 base_ckpt="/app/assets/models/single_species_models/base_segment_model.pt"):
        self.root = root
        self.base_ckpt = base_ckpt
        os.makedirs(root, exist_ok=True)

    def ckpt_path(self, species_name: str) -> str:
        safe = species_name.lower().replace(" ", "_").replace("/", "_")
        return os.path.join(self.root, f"{safe}.pt")

    def ensure_species_ckpt_exists(self, species_name: str):
        """
        If species checkpoint doesn't exist yet, copy from base.
        """
        path = self.ckpt_path(species_name)
        if not os.path.exists(path):
            if not os.path.exists(self.base_ckpt):
                raise FileNotFoundError(f"Base checkpoint not found: {self.base_ckpt}")
            state = torch.load(self.base_ckpt, map_location="cpu")
            torch.save(state, path)
        return path

    def load(self, in_dim: int, species_name: str, device="cpu") -> SpeciesInstanceModel:
        self.ensure_species_ckpt_exists(species_name)
        model = SpeciesInstanceModel(in_dim=in_dim)
        path = self.ckpt_path(species_name)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        return model

    def save(self, model: SpeciesInstanceModel, species_name: str) -> str:
        path = self.ckpt_path(species_name)
        torch.save(model.state_dict(), path)
        return path
