"""
Implémentation du système dynamique de Hénon.

Ce module fournit les fonctions nécessaires à la génération
de la série temporelle utilisée dans les expérimentations.
"""

from typing import List, Tuple


def henon_step(
    x: float,
    y: float,
    a: float = 1.4,
    b: float = 0.3,
) -> Tuple[float, float]:
    """
    Calcule une itération du système de Hénon.

    Parameters
    ----------
    x : float
        Valeur actuelle de x.
    y : float
        Valeur actuelle de y.
    a : float
        Paramètre du système.
    b : float
        Paramètre du système.

    Returns
    -------
    tuple
        (x_next, y_next)
    """

    x_next = 1 - a * x**2 + y
    y_next = b * x

    return x_next, y_next


def generate_henon_series(
    iterations: int = 500,
    x0: float = 0.0,
    y0: float = 0.0,
    a: float = 1.4,
    b: float = 0.3,
) -> Tuple[List[float], List[float]]:
    """
    Génère la série temporelle de Hénon.

    Parameters
    ----------
    iterations : int
        Nombre d'itérations.

    Returns
    -------
    tuple
        Deux listes contenant X et Y.
    """

    x = x0
    y = y0

    x_values = [x]
    y_values = [y]

    for _ in range(iterations):

        x, y = henon_step(x, y, a, b)

        x_values.append(x)
        y_values.append(y)

    return x_values, y_values
