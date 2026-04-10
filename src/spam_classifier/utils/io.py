import os

import joblib


def save_artifact(obj, artifact_dir: str, filename: str) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, filename)
    joblib.dump(obj, path)
    return path


def load_artifact(artifact_dir: str, filename: str):
    path = os.path.join(artifact_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)
