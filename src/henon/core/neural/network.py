"""
Implémentation du réseau de neurones (MLP).

Ce module définit la classe Network qui assemble les couches
et orchestre la propagation avant et arrière avec des opérations
matricielles vectorisées.
"""

from __future__ import annotations

import numpy as np

from henon.core.neural.activations import Activation, Linear, get_activation
from henon.core.neural.layers import DenseLayer, InputLayer, Layer


class Network:
    """
    Réseau de neurones multicouche (MLP).

    Parameters
    ----------
    layer_sizes : list[int]
        Liste des tailles : [entrée, cachée1, ..., sortie].
    activations : list[str | Activation] | None
        Activations pour chaque couche (cachée + sortie).
    seed : int | None
        Graine pour la reproductibilité.

    Examples
    --------
    >>> net = Network([6, 8, 1], ["tanh", "linear"])
    >>> output = net.predict(X)  # X: (n, 6) -> output: (n, 1)
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str | Activation] | None = None,
        seed: int | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Un MLP doit avoir au moins 2 couches.")

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        if activations is None:
            activations = ["tanh"] * (self.n_layers - 2) + ["linear"]

        if len(activations) != self.n_layers - 1:
            raise ValueError(
                f"Nombre d'activations ({len(activations)}) != "
                f"nombre de transitions ({self.n_layers - 1})."
            )

        self.layers: list[Layer] = []

        # Couche d'entrée
        self.layers.append(InputLayer(layer_sizes[0]))

        # Couches cachées + sortie
        for i in range(1, self.n_layers):
            act = activations[i - 1]
            if isinstance(act, str):
                act = get_activation(act)
            self.layers.append(DenseLayer(
                n_neurons=layer_sizes[i],
                n_inputs=layer_sizes[i - 1],
                activation=act,
                seed=seed,
            ))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagation avant vectorisée.

        Parameters
        ----------
        X : np.ndarray
            Entrées de forme (batch, n_inputs) ou (n_inputs,).

        Returns
        -------
        np.ndarray
            Sorties de forme (batch, n_outputs) ou (n_outputs,).
        """
        single = X.ndim == 1
        if single:
            X = X[np.newaxis, :]  # (1, n_inputs)

        output = X
        for layer in self.layers:
            output = layer.forward(output)

        if single:
            output = output[0]  # (n_outputs,)
        return output

    def backward(self, grad: np.ndarray) -> None:
        """
        Rétropropagation du gradient.

        Parameters
        ----------
        grad : np.ndarray
            Gradient de forme (batch, n_outputs).
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit la sortie pour un ou plusieurs échantillons.

        Parameters
        ----------
        X : np.ndarray
            Entrées de forme (batch, n_inputs) ou (n_inputs,).

        Returns
        -------
        np.ndarray
            Prédictions de forme (batch, n_outputs) ou (n_outputs,).
        """
        return self.forward(X)

    def summary(self) -> dict:
        total_params = 0
        details = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                n_params = layer.n_neurons * layer.n_inputs + layer.n_neurons
                total_params += n_params
                details.append({
                    "layer": i,
                    "type": "DenseLayer",
                    "neurons": layer.n_neurons,
                    "inputs": layer.n_inputs,
                    "parameters": n_params,
                })
            else:
                details.append({
                    "layer": i,
                    "type": "InputLayer",
                    "neurons": layer.n_neurons,
                    "inputs": layer.n_inputs,
                    "parameters": 0,
                })

        return {
            "architecture": self.layer_sizes,
            "total_layers": len(self.layers),
            "total_parameters": total_params,
            "details": details,
        }
