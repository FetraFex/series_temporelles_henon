"""
Calcul de l'erreur moyenne d'approximation (méthode ACP).

Ce module implémente la formule d'erreur d'embedding selon la
méthode du cours (Karhunen-Loève / ACP appliquée aux vecteurs retardés).
"""

from __future__ import annotations

import numpy as np


def compute_embedding_error(
    eigenvalues: np.ndarray,
    m: int,
) -> float:
    """
    Calcule l'erreur moyenne d'approximation E(m) pour une dimension m.

    Formule (méthode du cours — ACP / Karhunen-Loève) :
        E(m) = √( Σ(j=m à d-1) λⱼ )

    où :
        d   = nombre total de valeurs propres (dimension max)
        m   = dimension candidate testée
        λⱼ  = j-ème valeur propre (ordonnée décroissante)

    L'erreur E(m) représente la racine de la somme des valeurs propres
    « écartées » lorsqu'on retient uniquement les m premières composantes
    principales. Elle quantifie la perte d'information due à la troncature.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Valeurs propres triées par ordre décroissant.
    m : int
        Dimension candidate (1 <= m <= len(eigenvalues)).

    Returns
    -------
    float
        Erreur E(m) >= 0.

    Notes
    -----
    Cette fonction est isolée par design : elle encapsule la formule
    du cours et peut être remplacée sans toucher au reste du projet.
    La formule ci-dessus est la valeur par défaut du projet.
    """
    # On somme les valeurs propres d'indice m à la fin
    # (ce sont les composantes écartées par la troncature)
    remaining = eigenvalues[m:]
    return float(np.sqrt(np.sum(remaining)))
