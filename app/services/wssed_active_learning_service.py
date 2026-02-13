

## STEPS TO BE FOLLOWED ##
## 1) Loads the right model (species_model_store.py SpeciesModelStore.load )
## 2) Loads the embedding pool (plugging in DB)
## 3) Loads labels from DB
## runs ActiveLearning.step() to suggest
## 5) applies labels and retrains, then saves the species-specific .pt


from typing import Dict, List, Any

class ActiveLearningService:
    def __init__(self, db, model_store: SpeciesModelStore):
        self.db = db
        self.model_store = model_store

    # ---- you implement these for your DB schema ----
    def _load_embedding_pool(self, snippet_set_id: int):
        """
        Must return:
          X_pool: np.ndarray [N,D]
          Z_pool: np.ndarray [N,d] or None
          snippet_ids: List[int] length N  (maps pool index -> snippet_id)
        """
        raise NotImplementedError

    def _load_labels(self, snippet_set_id: int, species_name: str) -> Dict[int, int]:

        return {}

    def _save_labels(self, snippet_set_id: int, species_name: str, snippet_id_to_label: Dict[int, int]):
        """
        Persist labels to DB.
        """
        pass

    # ---- API methods ----
    def get_suggestions(
        self,
        snippet_set_id: int,
        species_name: str,
        strategy: str = "uncertainty",
        k: int = 20,
        device: str = "cpu",
        seed: int = 0,
    ) -> Dict[str, Any]:
        X_pool, Z_pool, snippet_ids = self._load_embedding_pool(snippet_set_id)
        in_dim = X_pool.shape[1]

        # 1) load correct model checkpoint for this species
        model = self.model_store.load(in_dim=in_dim, species_name=species_name, device="cpu")

        # 2) create AL object
        al = ActiveLearning(X_pool=X_pool, Z_pool=Z_pool)

        # 3) load labels from DB into al.y
        labels = self._load_labels(snippet_set_id, species_name)   # {snippet_id:0/1}
        sid_to_idx = {sid: i for i, sid in enumerate(snippet_ids)}
        idx_to_label = {sid_to_idx[sid]: lab for sid, lab in labels.items() if sid in sid_to_idx}
        al.apply_new_annotations(idx_to_label)

        # 4) suggest
        out = al.step(model=model, strategy=strategy, k=k, device=device, seed=seed)

        chosen_idx = out["chosen_indices"]
        chosen_snippet_ids = [snippet_ids[i] for i in chosen_idx]

        return {
            "snippet_ids": chosen_snippet_ids,
            "probs": out["probs"],
            "n_labeled": out["n_labeled"],
            "checkpoint": self.model_store.ckpt_path(species_name),
        }

    def submit_labels_and_maybe_retrain(
        self,
        snippet_set_id: int,
        species_name: str,
        snippet_id_to_label: Dict[int, int],  # {snippet_id:0/1}
        retrain_every: int = 20,
        device: str = "cuda",
        epochs: int = 5,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        X_pool, Z_pool, snippet_ids = self._load_embedding_pool(snippet_set_id)
        in_dim = X_pool.shape[1]

        # 1) load species model
        model = self.model_store.load(in_dim=in_dim, species_name=species_name, device="cpu")
        al = ActiveLearning(X_pool=X_pool, Z_pool=Z_pool)

        # 2) load existing labels
        existing = self._load_labels(snippet_set_id, species_name)
        sid_to_idx = {sid: i for i, sid in enumerate(snippet_ids)}
        al.apply_new_annotations({sid_to_idx[sid]: lab for sid, lab in existing.items() if sid in sid_to_idx})

        # 3) apply new labels (both to AL memory and DB)
        idx_to_label = {sid_to_idx[sid]: int(lab) for sid, lab in snippet_id_to_label.items() if sid in sid_to_idx}
        added = al.apply_new_annotations(idx_to_label)
        self._save_labels(snippet_set_id, species_name, snippet_id_to_label)

        # 4) retrain trigger
        labeled_count = int(al.is_labeled_mask().sum())
        do_retrain = (added > 0) and (labeled_count % retrain_every == 0)

        result = {"added": added, "labeled_count": labeled_count, "retrained": False}

        if do_retrain:
            stats = al.retrain(model, device=device, epochs=epochs, lr=lr)
            ckpt_path = self.model_store.save(model, species_name)
            result["retrained"] = True
            result["train_stats"] = stats
            result["checkpoint"] = ckpt_path

        return result
