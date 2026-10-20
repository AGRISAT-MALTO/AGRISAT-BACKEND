"""
Synchronise le modele Random Forest orge du projet ASA/Orge vers le backend.

La copie etait manuelle : apres un reentrainement dans D:/ASA/Orge, le backend
continuait de servir l'ancien orge_rf.joblib. Ce script copie les quatre
artefacts attendus par app/services/barley_rf_classifier.py et
app/routes/modeles.py, apres avoir verifie qu'ils sont coherents :

    orge_rf.joblib              modele (dict : model, features, threshold...)
    metrics.json                metriques affichees par /api/modeles
    threshold_selection.json    seuil retenu (relu au chargement)
    feature_importances.csv     importances affichees par /api/modeles

    python scripts/sync_orge_model.py                     # copie depuis D:/ASA/Orge/models
    python scripts/sync_orge_model.py --source D:/ASA/Orge/models
    python scripts/sync_orge_model.py --check             # verifie sans rien ecrire

Controles avant copie (code retour 1 si l'un echoue, rien n'est ecrit) :
  - RandomForestClassifier avec 48 variables ;
  - liste des variables identique a orge_model.FEATURES quand le projet source
    est importable (meme ordre) ;
  - seuil present et dans [0, 1].

--check compare les empreintes sha256 des fichiers et sort en 1 si le backend
diverge de la source (utile en integration continue).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.barley_rf_classifier import MODEL_DIR, MODEL_PATH  # noqa: E402

DEFAULT_SOURCE = Path(os.environ.get("ORGE_MODEL_DIR", r"D:\ASA\Orge\models"))
ARTEFACTS = {
    "orge_rf.joblib": True,
    "metrics.json": True,
    "threshold_selection.json": True,
    "feature_importances.csv": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_features(source_dir: Path) -> list[str] | None:
    """Variable FEATURES de orge_model.py, si le projet source est importable."""
    project_root = source_dir.parent
    if not (project_root / "orge_model.py").is_file():
        return None
    sys.path.insert(0, str(project_root))
    try:
        import orge_model  # noqa: PLC0415 — import optionnel, depend du projet source
    except Exception as error:  # noqa: BLE001
        print(f"  ! liste de variables non verifiee ({error})")
        return None
    finally:
        sys.path.pop(0)
    return [str(name) for name in orge_model.FEATURES]


def _validate(joblib_path: Path, source_dir: Path) -> list[str]:
    import joblib  # noqa: PLC0415

    errors: list[str] = []
    bundle = joblib.load(joblib_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        return [f"format inattendu dans {joblib_path.name} (dict avec 'model' attendu)"]

    model = bundle["model"]
    features = [str(name) for name in bundle.get("features") or []]
    if type(model).__name__ != "RandomForestClassifier":
        errors.append(f"estimateur {type(model).__name__} au lieu de RandomForestClassifier")
    if len(features) != 48:
        errors.append(f"{len(features)} variables au lieu de 48")

    expected = _source_features(source_dir)
    if expected is not None and expected != features:
        errors.append("liste de variables differente de orge_model.FEATURES (ordre inclus)")

    threshold = bundle.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        errors.append(f"seuil absent ou hors [0, 1] : {threshold!r}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronise le modele RF orge du projet ASA/Orge.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help=f"dossier source (defaut : {DEFAULT_SOURCE})")
    parser.add_argument("--check", action="store_true", help="verifie sans copier ; code 1 si divergence")
    args = parser.parse_args()

    source: Path = args.source
    print(f"Source  : {source}")
    print(f"Cible   : {MODEL_DIR}")

    if not source.is_dir():
        print("ERREUR : dossier source introuvable.")
        return 1
    missing = [name for name in ARTEFACTS if not (source / name).is_file()]
    if "orge_rf.joblib" in missing:
        print(f"ERREUR : {source / 'orge_rf.joblib'} introuvable.")
        return 1
    if missing:
        print(f"  ! artefacts facultatifs absents : {', '.join(missing)}")

    print("\nVerification du modele...")
    errors = _validate(source / "orge_rf.joblib", source)
    if errors:
        print("ERREUR : modele incoherent, rien n'a ete copie.")
        for error in errors:
            print(f"  - {error}")
        return 1
    selection = source / "threshold_selection.json"
    seuil = json.loads(selection.read_text(encoding="utf-8")).get("seuil_retenu", "?") if selection.is_file() else "?"
    print(f"  RandomForestClassifier, 48 variables, seuil {seuil} -> OK")

    if args.check:
        divergent = False
        for name in ARTEFACTS:
            source_file = source / name
            target_file = MODEL_DIR / name
            if not source_file.is_file():
                continue
            if not target_file.is_file():
                print(f"  - {name} : absent du backend")
                divergent = True
            elif _sha256(source_file) != _sha256(target_file):
                print(f"  - {name} : different du projet source")
                divergent = True
            else:
                print(f"  - {name} : a jour")
        return 1 if divergent else 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in ARTEFACTS:
        source_file = source / name
        if not source_file.is_file():
            continue
        target_file = MODEL_DIR / name
        shutil.copy2(source_file, target_file)
        print(f"  copie {name} ({target_file.stat().st_size} octets)")

    print(f"\nSynchronise. Modele servi : {MODEL_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
