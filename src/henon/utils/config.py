"""
Configuration du projet.

Charge les fichiers YAML depuis configs/ et les retourne
sous forme de dictionnaires.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"


def load_config(name: str) -> dict:
    """
    Charge un fichier de configuration YAML.

    Parameters
    ----------
    name : str
        Nom du fichier (sans extension), ex: "takens".

    Returns
    -------
    dict
        Contenu du fichier YAML parsé.
    """
    filepath = CONFIGS_DIR / f"{name}.yaml"
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
