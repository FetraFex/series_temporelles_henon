"""
Implémentation des couches de neurones.

Ce module définit les classes Layer, InputLayer, HiddenLayer et OutputLayer.
La classe DenseLayer est la plus efficace car elle utilise des opérations
matricielles vectorisées sur des batches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from henon.core.neural.activations import Activation, Linear, get_activation
from henon.core.neural.neurons import Neuron


class Layer(ABC):
    """Interface abstraite pour une couche de neurones."""

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagation avant. inputs: (batch, n_in) -> (batch, n_out)."""

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Rétropropagation. grad_output: (batch, n_out) -> (batch, n_in)."""

    @abstractmethod
    def update_weights(self, learning_rate: float) -> None:
        """Met à jour les poids."""

    @property
    @abstractmethod
    def n_neurons(self) -> int:
        """Nombre de neurones."""

    @property
    @abstractmethod
    def n_inputs(self) -> int:
        """Nombre d'entrées."""


class InputLayer(Layer):
    """Couche d'entrée : pas de poids, forward = identité."""

    def __init__(self, n_neurons: int) -> None:
        self._n_neurons = n_neurons

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output

    def update_weights(self, learning_rate: float) -> None:
        pass

    @property
    def n_neurons(self) -> int:
        return self._n_neurons

    @property
    def n_inputs(self) -> int:
        return self._n_neurons


class DenseLayer(Layer):
    """
    Couche dense avec calcul matriciel vectorisé.

    Effectue : output = activation(X @ W + b)
    où X est de forme (batch, n_in), W de (n_in, n_out), b de (n_out,).

    Parameters
    ----------
    n_neurons : int
        Nombre de neurones (sortie).
    n_inputs : int
        Nombre d'entrées.
    activation : Activation
        Fonction d'activation.
    seed : int | None
        Graine pour l'initialisation.
    """

    def __init__(
        self,
        n_neurons: int,
        n_inputs: int,
        activation: Activation | None = None,
        seed: int | None = None,
    ) -> None:
        self._n_neurons = n_neurons
        self._n_inputs = n_inputs
        self.activation = activation or Linear()

        # Initialisation Xavier
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (n_inputs + n_neurons))
        self.weights = rng.uniform(-limit, limit, size=(n_inputs, n_neurons))
        self.bias = np.zeros(n_neurons)

        # Cache pour rétropropagation
        self._last_input: np.ndarray | None = None
        self._last_z: np.ndarray | None = None

        # Gradients
        self.grad_weights: np.ndarray | None = None
        self.grad_bias: np.ndarray | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Propagation avant vectorisée.

        Parameters
        ----------
        inputs : np.ndarray
            Entrées de forme (batch, n_in).

        Returns
        -------
        np.ndarray
            Sorties de forme (batch, n_out).
        """
        self._last_input = inputs
        self._last_z = inputs @ self.weights + self.bias
        return self.activation(self._last_z)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Rétropropagation vectorisée.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient de forme (batch, n_out).

        Returns
        -------
        np.ndarray
            Gradient par rapport aux entrées : (batch, n_in).
        """
        batch_size = grad_output.shape[0]

        # Dérivée de l'activation
        act_deriv = self.activation.derivative(self._last_z)  # (batch, n_out)

        # Gradient local
        delta = grad_output * act_deriv  # (batch, n_out)

        # Gradients des poids et biais
        # Le facteur 1/n est déjà inclus dans le gradient de la perte (loss.derivative)
        # X.T @ delta donne la somme sur le batch, ce qui est correct
        self.grad_weights = self._last_input.T @ delta  # (n_in, n_out)
        self.grad_bias = delta.sum(axis=0)  # (n_out,)

        # Gradient pour la couche précédente
        return delta @ self.weights.T  # (batch, n_in)

    def update_weights(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

    @property
    def n_neurons(self) -> int:
        return self._n_neurons

    @property
    def n_inputs(self) -> int:
        return self._n_inputs


# Alias pour compatibilité
HiddenLayer = DenseLayer
OutputLayer = DenseLayer
