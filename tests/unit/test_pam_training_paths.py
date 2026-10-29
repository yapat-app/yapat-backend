from app.utils.pam_training_paths import (
    resolve_pam_metadata_path,
    resolve_pam_training_paths,
)


def test_metadata_path_does_not_require_label_config(tmp_path):
    metadata = tmp_path / "reference_pools" / "pool.csv"
    metadata.parent.mkdir()
    metadata.write_text("fname,species\na.wav,DENMIN\n", encoding="utf-8")

    resolved = resolve_pam_metadata_path(
        str(tmp_path),
        "/external/audio",
        "reference_pools/pool.csv",
    )

    assert resolved == "reference_pools/pool.csv"


def test_training_paths_still_require_both_files(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "pam_metadata.csv").write_text("fname,species\n", encoding="utf-8")
    (source / "pam_label_config.json").write_text("[]", encoding="utf-8")

    assert resolve_pam_training_paths(str(tmp_path), "dataset") == (
        "dataset/pam_metadata.csv",
        "dataset/pam_label_config.json",
    )
