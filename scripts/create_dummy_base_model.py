"""
Create a dummy base model file for PAM Active Learning.

This generates a minimal PyTorch-compatible state_dict placeholder
at models_AL/pam/base/base_pam_model.pt so the pipeline has a physical
file to resolve for first-time inference.

Run once:
    python scripts/create_dummy_base_model.py
"""

import os
import sys

# Ensure repo root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:
    # Fallback: write a pickle-compatible bytes file that torch.load can
    # at least *open* (empty ordered-dict state_dict).
    import pickle, collections

    dst = os.path.join(ROOT, "models_AL", "pam", "base", "base_pam_model.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    state_dict = collections.OrderedDict()
    with open(dst, "wb") as f:
        pickle.dump(state_dict, f)
    print(f"[fallback/pickle] Dummy base model written to {dst}")
    sys.exit(0)

# ---------- torch available ----------
dst = os.path.join(ROOT, "models_AL", "pam", "base", "base_pam_model.pt")
os.makedirs(os.path.dirname(dst), exist_ok=True)

# Minimal dummy state_dict (single linear layer placeholder)
import torch.nn as nn

dummy = nn.Linear(128, 10)
torch.save(dummy.state_dict(), dst)
print(f"Dummy base model written to {dst}  (Linear 128→10)")
