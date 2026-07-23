"""
Construction des vecteurs retardés (Takens).

Ce module construit les vecteurs de plongement x̄(i) à partir
d'une série temporelle scalaire.
"""

from __future__ import annotations

import numpy as np


def build_delay_vectors(
    series: np.ndarray | list[float],
    delay: int,
    dimension: int,
) -> np.ndarray:
    """
    Construit les vecteurs retardés pour l'algorithme de Takens.

    Chaque vecteur est défini par :
        x̄(i) = [u(i), u(i+r), u(i+2r), ..., u(i+(m-1)r)]

    pour i allant de 0 à M-1, où :
        r  = delay (délai de reconstruction)
        m  = dimension (nombre de composantes)
        N  = longueur de la série
        M  = N - (m-1)*r (nombre de vecteurs construisibles)

    Parameters
    ----------
    series : np.ndarray | list[float]
        Série temporelle scalaire u(0), u(1), ..., u(N-1).
    delay : int
        Délai de reconstruction r (>= 1).
    dimension : int
        Dimension candidate m (nombre de composantes, >= 1).

    Returns
    -------
    np.ndarray
        Matrice de shape (M, dimension) contenant les vecteurs retardés.

    Raises
    ------
    ValueError
        Si M <= 0 (dimension trop grande pour la longueur de la série),
        si delay < 1, ou si dimension < 1.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)

    if delay < 1:
        raise ValueError(f"Le délai doit être >= 1, reçu : {delay}.")
    if dimension < 1:
        raise ValueError(f"La dimension doit être >= 1, reçu : {dimension}.")

    # Nombre de vecteurs constructibles
    m_vectors = n - (dimension - 1) * delay

    if m_vectors <= 0:
        raise ValueError(
            f"Impossible de construire des vecteurs : M = {m_vectors} <= 0. "
            f"La dimension {dimension} est trop grande pour une série de "
            f"longueur {n} avec un délai de {delay}."
        )

    # Construction de la matrice (M, dimension)
    indices = np.arange(dimension) * delay
    offsets = np.arange(m_vectors)[:, np.newaxis]
    vector_indices = offsets + indices

    delay_vectors = series[vector_indices]

    return delay_vectors
