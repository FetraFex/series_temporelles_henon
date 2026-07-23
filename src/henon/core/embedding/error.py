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
    Calcule l'erreur d'approximation E(m) pour une dimension m.

    Formule :
        E(m) = √( λ_{m+1} )

    où λ_{m+1} est la (m+1)-ème valeur propre (ordonnée décroissante).
    Cette erreur représente la contribution de la première composante
    écartée par la troncature à la dimension m.

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
    """
    if m >= len(eigenvalues):
        return 0.0
    return float(np.sqrt(eigenvalues[m]))
