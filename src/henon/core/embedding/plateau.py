"""
Détection automatique du premier plateau.

Ce module identifie la première dimension m pour laquelle
l'erreur d'embedding se stabilise sous un seuil relatif.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def find_first_plateau(
    dimensions: np.ndarray,
    errors: np.ndarray,
    threshold: float = 0.001,
    min_consecutive: int = 2,
) -> int:
    """
    Détecte la première dimension optimale (premier plateau).

    Critère : la première valeur de m telle que la variation relative
        |E(m-1) - E(m)| / E(m_min) < threshold
    se maintient pendant au moins `min_consecutive` pas consécutifs.

    Parameters
    ----------
    dimensions : np.ndarray
        Tableau des dimensions testées (ex: [2, 3, ..., 20]).
    errors : np.ndarray
        Tableau des erreurs correspondantes E(m).
    threshold : float
        Seuil de variation relative pour considérer un plateau.
    min_consecutive : int
        Nombre de pas consécutifs sous le seuil requis.

    Returns
    -------
    int
        Dimension optimale détectée.

    Notes
    -----
    Si aucun plateau n'est détecté avant la dernière dimension,
    un avertissement est émis et la dernière dimension est retournée.
    """
    if len(dimensions) < 2:
        return int(dimensions[0])

    # Variation relative par rapport à la première erreur
    base_error = errors[0]
    if base_error == 0:
        # Toutes les erreurs sont nulles → toute dimension est optimale
        return int(dimensions[0])

    # Calcul des variations relatives
    relative_variations = np.abs(np.diff(errors)) / base_error

    # Recherche du premier plateau : min_consecutive pas consécutifs sous le seuil
    consecutive_count = 0
    for i, variation in enumerate(relative_variations):
        if variation < threshold:
            consecutive_count += 1
            if consecutive_count >= min_consecutive:
                # Le plateau commence à l'indice i - min_consecutive + 2
                # (car on cherche la dimension correspondante)
                plateau_start = i - min_consecutive + 2
                optimal = int(dimensions[plateau_start])
                logger.info(
                    f"Plateau détecté à dimension {optimal} "
                    f"({min_consecutive} pas consécutifs sous le seuil {threshold})."
                )
                return optimal
        else:
            consecutive_count = 0

    # Aucun plateau trouvé
    optimal = int(dimensions[-1])
    logger.warning(
        f"Aucun plateau détecté avant la dimension {optimal}. "
        f"La dimension maximale est retournée par défaut."
    )
    return optimal
