import joblib
import pytest

pytest.importorskip("sklearn")

from app.services.barley_rf_classifier import MODEL_PATH  # noqa: E402
from scripts.sync_orge_model import _sha256, _validate  # noqa: E402


def test_bundled_model_passes_validation():
    if not MODEL_PATH.is_file():
        pytest.skip("modèle orge absent")

    # source_dir = MODEL_DIR : orge_model.py n'est pas là, la liste de variables
    # n'est donc pas recoupée (contrôlé par le script, hors tests unitaires).
    assert _validate(MODEL_PATH, MODEL_PATH.parent) == []


def test_non_random_forest_bundle_is_rejected(tmp_path):
    from sklearn.linear_model import LogisticRegression

    broken = tmp_path / "broken.joblib"
    joblib.dump({"model": LogisticRegression(), "features": ["B2"], "threshold": 0.5}, broken)

    errors = _validate(broken, tmp_path)

    assert any("RandomForestClassifier" in error for error in errors)
    assert any("au lieu de 48" in error for error in errors)


def test_missing_threshold_is_rejected(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    broken = tmp_path / "no-threshold.joblib"
    joblib.dump(
        {"model": RandomForestClassifier(n_estimators=2), "features": [f"f{i}" for i in range(48)]},
        broken,
    )

    errors = _validate(broken, tmp_path)

    assert any("seuil" in error for error in errors)


def test_sha256_is_stable(tmp_path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"orge")

    assert _sha256(sample) == _sha256(sample)
    assert len(_sha256(sample)) == 64
