"""Checkpoint filename construction must survive deep retrain lineages.

Every retrain appends its trigger + timestamp onto the parent's version string
(``service.py``: ``f"{parent_ckpt.version}_r{timestamp}"``), so the version grows
~12-18 chars per generation. Since the version is embedded in the checkpoint
filename, generation 17 of dataset 4 produced a 256-byte basename and
``torch.save`` died with ``[enforce fail at inline_container.cc:747] . open file
failed with strerror: File name too long`` (NAME_MAX is 255).
"""

import os

import pytest

from app.config import settings
from app.services.pam_al import _checkpoint_helpers as ckpt_h

# The lineage that actually broke: checkpoint 38 on dataset 4, one retrain away
# from the limit.
_CKPT_38_VERSION = (
    "v0_parent_r1784109270_r1784110497_manual_1784110938_r1784815800_r1784880726"
    "_manual_1784888859_manual_1784899117_manual_1784899195_r1785167637_r1785238129"
    "_r1785238195_r1785238583_r1785239581_r1785241288_r1785242515"
)
_CKPT_39_VERSION = f"{_CKPT_38_VERSION}_r1785250349"

NAME_MAX = 255


@pytest.fixture
def checkpoints_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PAM_CHECKPOINTS_DIR", str(tmp_path), raising=False)
    return tmp_path


def test_short_version_keeps_the_readable_naming(checkpoints_dir):
    """Existing checkpoints must keep resolving to their current filenames."""
    cp = ckpt_h.make_checkpoint_path(4, "pam_linear_anuraset", "v0", 21)
    lcp = ckpt_h.make_label_config_path(4, "pam_linear_anuraset", "v0", 21)

    assert os.path.basename(cp) == "pam_linear_anuraset_v0_ckpt_21.pt"
    assert os.path.basename(lcp) == "pam_linear_anuraset_v0_labels_21.json"


def test_deep_lineage_stays_writable(checkpoints_dir):
    """The generation that failed in production must now produce a usable path."""
    cp = ckpt_h.make_checkpoint_path(4, "pam_linear_anuraset", _CKPT_39_VERSION, 39)

    basename = os.path.basename(cp)
    assert len(basename.encode()) <= NAME_MAX, len(basename.encode())

    # The real assertion: the filesystem accepts it (this is what torch.save does).
    with open(cp, "wb") as f:
        f.write(b"weights")
    assert os.path.isfile(cp)


def test_deep_lineage_label_config_stays_writable(checkpoints_dir):
    lcp = ckpt_h.make_label_config_path(4, "pam_linear_anuraset", _CKPT_39_VERSION, 39)

    assert len(os.path.basename(lcp).encode()) <= NAME_MAX
    with open(lcp, "w", encoding="utf-8") as f:
        f.write("{}")


def test_sibling_lineages_do_not_collide(checkpoints_dir):
    """Two long versions differing only in their tail must not share a filename."""
    a = ckpt_h.make_checkpoint_path(4, "pam_linear_anuraset", f"{_CKPT_38_VERSION}_r1785250349", 39)
    b = ckpt_h.make_checkpoint_path(4, "pam_linear_anuraset", f"{_CKPT_38_VERSION}_r1785250350", 39)

    assert a != b


def test_absurd_family_name_still_yields_a_writable_path(checkpoints_dir):
    """Even when the non-version part alone overflows, we must return something usable."""
    cp = ckpt_h.make_checkpoint_path(4, "x" * 400, _CKPT_39_VERSION, 39)

    assert len(os.path.basename(cp).encode()) <= NAME_MAX
    with open(cp, "wb") as f:
        f.write(b"weights")
