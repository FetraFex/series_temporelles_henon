"""
Script pédagogique de propagation avant (Forward Propagation).

Ce script exécute une propagation avant complète sur UN SEUL prototype
du dataset et affiche chaque étape du calcul de manière détaillée,
adaptée à l'insertion dans un mémoire universitaire.

Architecture : 6-8-1
Activations  : Sigmoid (cachée), Linear (sortie)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from henon.data.supervised import Dataset
from henon.core.neural.network import Network

# --- Configuration ---
XLSX_PATH = Path("data/raw/henon_500.xlsx")
EMBEDDING_DIM = 6
DELAY = 1
SEED = 42
PROTOTYPE_INDEX = 0


def sigmoid(x: float) -> float:
    """
    Fonction sigmoïde : σ(x) = 1 / (1 + exp(-x)).

    Parameters
    ----------
    x : float
        Entrée.

    Returns
    -------
    float
        Sortie dans l'intervalle (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))


def linear(x: float) -> float:
    """
    Fonction linéaire (identité) : f(x) = x.

    Parameters
    ----------
    x : float
        Entrée.

    Returns
    -------
    float
        Sortie identique à l'entrée.
    """
    return x


def main() -> None:
    """Exécute la propagation avant pédagogique."""

    # ================================================================
    # Chargement des données
    # ================================================================
    dataset = Dataset.from_excel(
        path=XLSX_PATH,
        column="Xn",
        embedding_dim=EMBEDDING_DIM,
        delay=DELAY,
    )
    dataset.build_supervised()
    dataset.normalize(method="minmax")
    dataset.split()

    # ================================================================
    # Création du réseau avec des poids connus (seed=42)
    # ================================================================
    network = Network(
        layer_sizes=[6, 8, 1],
        activations=["sigmoid", "linear"],
        seed=SEED,
    )

    # Extraction des couches denses
    dense_layers = [l for l in network.layers if hasattr(l, "weights")]
    layer_hidden = dense_layers[0]  # Couche cachée (Input -> Hidden)
    layer_output = dense_layers[1]  # Couche de sortie (Hidden -> Output)

    # ================================================================
    # Sélection du prototype
    # ================================================================
    prototype = dataset.X_test[PROTOTYPE_INDEX]
    target = dataset.y_test[PROTOTYPE_INDEX][0]
    target_real = dataset.denormalize_y(np.array([[target]]))[0][0]

    # ================================================================
    # En-tête
    # ================================================================
    print("")
    print("=" * 56)
    print("  FORWARD PASS - RESEAU 6-8-1")
    print("  Propagation avant : demonstration detaillee")
    print("=" * 56)
    print("")
    print(f"  Prototype utilise : {PROTOTYPE_INDEX}")
    print(f"  Source : jeu de test (normalise)")
    print("")

    # ================================================================
    # ETAPE 1 : Donnees d'entree
    # ================================================================
    print("=" * 56)
    print("  ETAPE 1 : DONNEES D'ENTREE (Input Layer)")
    print("=" * 56)
    print("")
    print("  Les 6 valeurs d'entree correspondent aux 6 composantes")
    print("  du vecteur de Takens (reconstruction de dimension 6) :")
    print("")

    x_values = prototype.tolist()
    for i, val in enumerate(x_values):
        print(f"    x{i + 1} = {val:.8f}")
    print("")

    # ================================================================
    # ETAPE 2 : Couche cachee (Hidden Layer)
    # ================================================================
    print("=" * 56)
    print("  ETAPE 2 : COUCHE CACHEE (Hidden Layer - 8 neurones)")
    print("=" * 56)
    print("")
    print("  Formule pour chaque neurone j :")
    print("    z_j = SUM(w_i,j * x_i) + b_j")
    print("    h_j = sigmoid(z_j)")
    print("")

    # Récupération des poids et biais de la couche cachée
    W_hidden = layer_hidden.weights  # Shape: (6, 8)
    b_hidden = layer_hidden.bias    # Shape: (8,)

    h_values = []  # Pour stocker les sorties de la couche cachée

    for j in range(8):
        # Calcul de la somme pondérée z_j
        z_j = 0.0
        terms = []
        for i in range(6):
            product = W_hidden[i, j] * x_values[i]
            terms.append(f"({W_hidden[i, j]:.8f} * {x_values[i]:.8f})")
            z_j += product
        z_j += b_hidden[j]

        # Application de la fonction sigmoid
        h_j = sigmoid(z_j)
        h_values.append(h_j)

        print(f"  Neurone {j + 1} :")
        print(f"    z{j + 1} = SUM(w_i * x_i) + b")
        print(f"        = {' + '.join(terms)}")
        print(f"        = {z_j:.8f}")
        print(f"    sigmoid(z{j + 1}) = {h_j:.8f}")
        print("")

    # ================================================================
    # ETAPE 3 : Couche de sortie (Output Layer)
    # ================================================================
    print("=" * 56)
    print("  ETAPE 3 : COUCHE DE SORTIE (Output Layer - 1 neurone)")
    print("=" * 56)
    print("")
    print("  Formule :")
    print("    z = SUM(w_j * h_j) + b")
    print("    y = Linear(z) = z")
    print("")

    # Récupération des poids et biais de la couche de sortie
    W_output = layer_output.weights  # Shape: (8, 1)
    b_output = layer_output.bias    # Shape: (1,)

    # Calcul de la somme pondérée pour le neurone de sortie
    z_output = 0.0
    terms = []
    for j in range(8):
        product = W_output[j, 0] * h_values[j]
        terms.append(f"({W_output[j, 0]:.8f} * {h_values[j]:.8f})")
        z_output += product
    z_output += b_output[0]

    # Activation linéaire
    y_pred = linear(z_output)

    print("  Neurone de sortie :")
    print(f"    z = SUM(w_j * h_j) + b")
    print(f"      = {' + '.join(terms)}")
    print(f"      = {z_output:.8f}")
    print(f"    Linear(z) = {y_pred:.8f}")
    print("")

    # ================================================================
    # ETAPE 4 : Comparaison avec la valeur reelle
    # ================================================================
    print("=" * 56)
    print("  ETAPE 4 : COMPARAISON AVEC LA VALEUR REELLE")
    print("=" * 56)
    print("")

    # Dénormaliser la prédiction
    y_pred_denorm = dataset.denormalize_y(np.array([[y_pred]]))[0][0]
    error = target_real - y_pred_denorm

    print(f"  Valeur reelle     : {target_real:.8f}")
    print(f"  Valeur predite    : {y_pred_denorm:.8f}")
    print(f"  Erreur (e = y - y): {error:.8f}")
    print("")

    # ================================================================
    # Verification avec le reseau
    # ================================================================
    print("=" * 56)
    print("  VERIFICATION AVEC LE RESEAU")
    print("=" * 56)
    print("")

    y_network = network.predict(prototype.reshape(1, -1))[0][0]
    y_network_denorm = dataset.denormalize_y(np.array([[y_network]]))[0][0]

    print(f"  Sortie du reseau  : {y_network_denorm:.8f}")
    print(f"  Notre calcul      : {y_pred_denorm:.8f}")
    print(f"  Difference        : {abs(y_network_denorm - y_pred_denorm):.2e}")
    print("")

    # ================================================================
    # Resume mathematique pour le memoire
    # ================================================================
    print("=" * 56)
    print("  RESUME MATHEMATIQUE")
    print("=" * 56)
    print("")
    print("  Couche cachee (j = 1, ..., 8) :")
    print("    z_j = SUM_{i=1}^{6} (w_{i,j} * x_i) + b_j")
    print("    h_j = sigma(z_j) = 1 / (1 + exp(-z_j))")
    print("")
    print("  Couche de sortie :")
    print("    z = SUM_{j=1}^{8} (w_j * h_j) + b")
    print("    y_hat = z    (activation lineaire)")
    print("")
    print("  ou :")
    print("    x_i = composantes du vecteur de Takens")
    print("    w_{i,j} = poids de l'entree i vers le neurone j")
    print("    h_j = sortie du neurone cache j")
    print("    sigma = fonction sigmoide")
    print("=" * 56)
    print("")


if __name__ == "__main__":
    main()
