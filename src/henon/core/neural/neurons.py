"""
Implémentation d'un neurone artificiel.

Ce module définit la classe Neuron, un neurone artificiel unique
avec poids, biais et fonction d'activation.
"""

from __future__ import annotations

import numpy as np

from henon.core.neural.activations import Activation, Linear


class Neuron:
    """
    Neurone artificiel avec poids et biais.

    Parameters
    ----------
    n_inputs : int
        Nombre d'entrées.
    activation : Activation
        Fonction d'activation.
    seed : int | None
        Graine pour la reproductibilité.
    """

    def __init__(
        self,
        n_inputs: int,
        activation: Activation | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_inputs = n_inputs
        self.activation = activation or Linear()
        self.rng = np.random.default_rng(seed)

        limit = np.sqrt(6.0 / (n_inputs + 1))
        self.weights = self.rng.uniform(-limit, limit, size=n_inputs)
        self.bias = 0.0

        # Cache pour rétropropagation
        self._last_input: np.ndarray | None = None
        self._last_z: float | None = None
