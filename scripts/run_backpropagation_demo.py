"""
Script pedagogique de retropropagation de l'erreur.

Ce script execute une seule etape complete d'apprentissage sur UN SEUL
prototype du dataset et affiche chaque etape du calcul de maniere
detaillee, adaptee a l'insertion dans un memoire universitaire.

Architecture : 6-8-1
Activations  : Sigmoid (cachee), Linear (sortie)
Fonction de perte : MSE (Mean Squared Error)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from henon.data.supervised import Dataset
from henon.core.neural.network import Network
from henon.core.neural.losses import MSELoss

# --- Configuration ---
XLSX_PATH = Path("data/raw/henon_500.xlsx")
EMBEDDING_DIM = 6
DELAY = 1
SEED = 42
LEARNING_RATE = 0.1
PROTOTYPE_INDEX = 0


def sigmoid(x: float) -> float:
    """Fonction sigmoide : sigma(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative_from_output(h: float) -> float:
    """
    Dérivée de la sigmoide en fonction de la sortie h.

    sigma'(z) = sigma(z) * (1 - sigma(z)) = h * (1 - h)
    """
    return h * (1.0 - h)


def main() -> None:
    """Execute la demonstration de retropropagation."""

    # ================================================================
    # Chargement des donnees et du reseau
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

    network = Network(
        layer_sizes=[6, 8, 1],
        activations=["sigmoid", "linear"],
        seed=SEED,
    )

    # Extraction des couches denses
    dense_layers = [l for l in network.layers if hasattr(l, "weights")]
    layer_hidden = dense_layers[0]
    layer_output = dense_layers[1]

    loss_fn = MSELoss()

    # ================================================================
    # Selection du prototype
    # ================================================================
    prototype = dataset.X_test[PROTOTYPE_INDEX].reshape(1, -1)
    target = dataset.y_test[PROTOTYPE_INDEX].reshape(1, -1)

    x_values = prototype[0].tolist()
    target_value = float(target[0, 0])

    # ================================================================
    # En-tete
    # ================================================================
    print("")
    print("=" * 56)
    print("  DEMONSTRATION DE RETROPROPAGATION")
    print("  Architecture : 6-8-1")
    print("=" * 56)
    print("")

    # ================================================================
    # 1. DONNEES UTILISEES
    # ================================================================
    print("=" * 56)
    print("  1. DONNEES UTILISEES")
    print("=" * 56)
    print("")
    print("  Prototype selectionne :")
    print(f"  Source : jeu de test (normalise), index {PROTOTYPE_INDEX}")
    print("")
    print("  Entrees (vecteur de Takens, 6 dimensions) :")
    print("")
    for i, val in enumerate(x_values):
        print(f"    x{i + 1} = {val:.8f}")
    print("")
    print(f"  Valeur reelle attendue : y = {target_value:.8f}")
    print("")

    # ================================================================
    # 2. SORTIE DU RESEAU AVANT MISE A JOUR
    # ================================================================
    print("=" * 56)
    print("  2. SORTIE DU RESEAU AVANT MISE A JOUR")
    print("=" * 56)
    print("")

    # Propagation avant
    y_pred_before = network.predict(prototype)[0, 0]
    y_pred_before_denorm = dataset.denormalize_y(
        np.array([[y_pred_before]])
    )[0, 0]

    # Extraction des valeurs intermediaires
    h_values = layer_hidden._last_input @ layer_hidden.weights + layer_hidden.bias
    h_activated = sigmoid(h_values[0, 0])  # Sigmoid pour chaque neurone
    h_all = []
    for j in range(8):
        h_all.append(sigmoid(float(h_values[0, j])))

    z_output = layer_output._last_input @ layer_output.weights + layer_output.bias

    print("  Couche cachee (sorties avant activation) :")
    print("")
    for j in range(8):
        print(f"    z_hidden_{j + 1} = {float(h_values[0, j]):.8f}")
    print("")
    print("  Couche cachee (sorties apres activation sigmoid) :")
    print("")
    for j in range(8):
        print(f"    h_{j + 1} = sigmoid(z_{j + 1}) = {h_all[j]:.8f}")
    print("")
    print(f"  Sortie du reseau (avant mise a jour) :")
    print(f"    y_pred = {y_pred_before:.8f}")
    print(f"    y_pred (denormalise) = {y_pred_before_denorm:.8f}")
    print("")

    # ================================================================
    # 3. CALCUL DE L'ERREUR
    # ================================================================
    print("=" * 56)
    print("  3. CALCUL DE L'ERREUR")
    print("=" * 56)
    print("")

    # Calcul de l'erreur
    error = target_value - y_pred_before_denorm

    # MSE sur la valeur denormalisee
    mse_value = float(loss_fn(
        np.array([[y_pred_before_denorm]]),
        np.array([[target_value]])
    ))

    print("  Formule de l'erreur :")
    print("    e = y - y_hat")
    print("")
    print(f"  Erreur de prediction :")
    print(f"    e = {target_value:.8f} - {y_pred_before_denorm:.8f}")
    print(f"    e = {error:.8f}")
    print("")
    print("  Fonction de perte MSE :")
    print("    MSE = (1/n) * SUM((y_pred - y_true)^2)")
    print("")
    print(f"  Pour cet unique prototype :")
    print(f"    MSE = ({y_pred_before_denorm:.8f} - {target_value:.8f})^2")
    print(f"    MSE = {mse_value:.8f}")
    print("")
    print("  NOTE : Cette erreur est utilisee pour guider la correction")
    print("  des poids synaptiques lors de la retropropagation.")
    print("")

    # ================================================================
    # 4. DELTA DU NEURONE DE SORTIE
    # ================================================================
    print("=" * 56)
    print("  4. DELTA DU NEURONE DE SORTIE")
    print("=" * 56)
    print("")

    # Pour un seul prototype, le gradient de la perte est :
    # grad = 2 * (y_pred - y_true) / n
    # Avec n=1 (batch de taille 1) :
    # grad = 2 * (y_pred - y_true)
    grad_output = 2.0 * (y_pred_before_denorm - target_value)

    # Delta du neurone de sortie :
    # delta = grad * derivative(activation)
    # Pour Linear, la derivee est 1
    delta_output = grad_output * 1.0  # derivee de Linear = 1

    print("  Le delta represente l'erreur locale du neurone.")
    print("  Il est calcule comme :")
    print("    delta = gradient_perte * derivee(activation)")
    print("")
    print("  Gradient de la perte (pour n=1) :")
    print("    grad = 2 * (y_pred - y_true)")
    print(f"    grad = 2 * ({y_pred_before_denorm:.8f} - {target_value:.8f})")
    print(f"    grad = {grad_output:.8f}")
    print("")
    print("  Derivee de la fonction d'activation Linear :")
    print("    Linear'(z) = 1")
    print("")
    print("  Delta du neurone de sortie :")
    print(f"    delta_output = grad * Linear'(z)")
    print(f"    delta_output = {grad_output:.8f} * 1")
    print(f"    delta_output = {delta_output:.8f}")
    print("")

    # ================================================================
    # 5. DELTAS DES NEURONES CACHES
    # ================================================================
    print("=" * 56)
    print("  5. DELTAS DES NEURONES CACHES")
    print("=" * 56)
    print("")

    # Recuperation des poids sorties -> cachee
    W_output = layer_output.weights  # (8, 1)

    # Delta cache = delta_sortie * poids * derivee_sigmoid(h)
    delta_hidden = []
    for j in range(8):
        w_j = float(W_output[j, 0])
        h_j = h_all[j]
        deriv = sigmoid_derivative_from_output(h_j)
        delta_j = delta_output * w_j * deriv
        delta_hidden.append(delta_j)

        print(f"  Neurone {j + 1} :")
        print(f"    delta_{j + 1} = delta_output * w_{j + 1} * sigma'(z_{j + 1})")
        print(f"    delta_{j + 1} = {delta_output:.8f} * {w_j:.8f} * {deriv:.8f}")
        print(f"    delta_{j + 1} = {delta_j:.8f}")
        print("")

    # ================================================================
    # 6. CALCUL DES GRADIENTS
    # ================================================================
    print("=" * 56)
    print("  6. CALCUL DES GRADIENTS")
    print("=" * 56)
    print("")

    print("  Les gradients indiquent la direction et l'amplitude")
    print("  de la correction a appliquer aux poids.")
    print("")

    # Gradient pour les poids cachee -> sortie
    print("  --- Poids Cachee -> Sortie ---")
    print("")
    print("  Formule :")
    print("    dE/dw_j = h_j * delta_output")
    print("")

    W_hidden = layer_hidden.weights  # (6, 8)

    # Exemple pour le poids w_{1,1} (entree 1 -> neurone cache 1 -> sortie)
    j_ex = 0
    i_ex = 0
    w_ex = float(W_output[j_ex, 0])
    h_ex = h_all[j_ex]
    grad_ex = h_ex * delta_output
    print(f"  Exemple : poids w_{{1,1}} (h_1 -> sortie)")
    print(f"    dE/dw_{{1,1}} = h_1 * delta_output")
    print(f"    dE/dw_{{1,1}} = {h_ex:.8f} * {delta_output:.8f}")
    print(f"    dE/dw_{{1,1}} = {grad_ex:.8f}")
    print("")

    # Gradient pour les poids entree -> cachee
    print("  --- Poids Entree -> Cachee ---")
    print("")
    print("  Formule :")
    print("    dE/dw_{i,j} = x_i * delta_j")
    print("")

    # Exemple pour le poids w_{1,1} (entree 1 -> neurone cache 1)
    x_ex = x_values[i_ex]
    delta_ex = delta_hidden[j_ex]
    grad_hidden_ex = x_ex * delta_ex
    print(f"  Exemple : poids w_{{1,1}} (x_1 -> neurone cache 1)")
    print(f"    dE/dw_{{1,1}} = x_1 * delta_1")
    print(f"    dE/dw_{{1,1}} = {x_ex:.8f} * {delta_ex:.8f}")
    print(f"    dE/dw_{{1,1}} = {grad_hidden_ex:.8f}")
    print("")

    # ================================================================
    # 7. MISE A JOUR DES POIDS
    # ================================================================
    print("=" * 56)
    print("  7. MISE A JOUR DES POIDS")
    print("=" * 56)
    print("")

    print("  Formule de mise a jour :")
    print("    w_new = w_old - eta * gradient")
    print("")
    print(f"  Learning rate (eta) = {LEARNING_RATE}")
    print("")

    # Sauvegarde des poids avant mise a jour
    w_before_hidden = float(W_hidden[i_ex, j_ex])
    w_before_output = float(W_output[j_ex, 0])

    # Calcul du gradient pour le poids cachee -> sortie
    grad_output_weight = float(h_all[j_ex] * delta_output)
    w_new_output = w_before_output - LEARNING_RATE * grad_output_weight

    # Calcul du gradient pour le poids entree -> cachee
    grad_hidden_weight = float(x_values[i_ex] * delta_hidden[j_ex])
    w_new_hidden = w_before_hidden - LEARNING_RATE * grad_hidden_weight

    print("  --- Mise a jour d'un poids Cachee -> Sortie ---")
    print("")
    print(f"  Poids w_{{h_{j_ex + 1},output}} :")
    print(f"    Avant : w = {w_before_output:.8f}")
    print(f"    Gradient : dw = {grad_output_weight:.8f}")
    print(f"    w_new = w - eta * dw")
    print(f"    w_new = {w_before_output:.8f} - {LEARNING_RATE} * {grad_output_weight:.8f}")
    print(f"    w_new = {w_new_output:.8f}")
    print("")

    print("  --- Mise a jour d'un poids Entree -> Cachee ---")
    print("")
    print(f"  Poids w_{{x_{i_ex + 1},h_{j_ex + 1}}} :")
    print(f"    Avant : w = {w_before_hidden:.8f}")
    print(f"    Gradient : dw = {grad_hidden_weight:.8f}")
    print(f"    w_new = w - eta * dw")
    print(f"    w_new = {w_before_hidden:.8f} - {LEARNING_RATE} * {grad_hidden_weight:.8f}")
    print(f"    w_new = {w_new_hidden:.8f}")
    print("")

    # ================================================================
    # Application reelle de la mise a jour sur le reseau
    # ================================================================
    print("=" * 56)
    print("  APPLICATION REELLE DE LA MISE A JOUR")
    print("=" * 56)
    print("")

    # Propagation avant + retropropagation + mise a jour
    y_pred_before_network = network.predict(prototype)[0, 0]
    mse_before = float(loss_fn(
        dataset.denormalize_y(np.array([[y_pred_before_network]])),
        dataset.denormalize_y(target)
    ))

    # Une etape d'apprentissage
    y_pred = network.forward(prototype)
    grad = loss_fn.derivative(y_pred, target)
    network.backward(grad)
    network.update_weights(LEARNING_RATE)

    y_pred_after_network = network.predict(prototype)[0, 0]
    mse_after = float(loss_fn(
        dataset.denormalize_y(np.array([[y_pred_after_network]])),
        dataset.denormalize_y(target)
    ))

    y_pred_before_denorm = float(dataset.denormalize_y(
        np.array([[y_pred_before_network]])
    )[0, 0])
    y_pred_after_denorm = float(dataset.denormalize_y(
        np.array([[y_pred_after_network]])
    )[0, 0])

    print("  Avant la mise a jour :")
    print(f"    y_pred = {y_pred_before_network:.8f} (normalise)")
    print(f"    y_pred = {y_pred_before_denorm:.8f} (denormalise)")
    print(f"    MSE    = {mse_before:.8f}")
    print("")
    print("  Apres la mise a jour (1 epoch) :")
    print(f"    y_pred = {y_pred_after_network:.8f} (normalise)")
    print(f"    y_pred = {y_pred_after_denorm:.8f} (denormalise)")
    print(f"    MSE    = {mse_after:.8f}")
    print("")

    # ================================================================
    # 8. VERIFICATION
    # ================================================================
    print("=" * 56)
    print("  8. VERIFICATION : L'ERREUR DIMINUE ?")
    print("=" * 56)
    print("")

    reduction = mse_before - mse_after
    pourcentage = (reduction / mse_before * 100) if mse_before > 0 else 0

    print(f"  MSE avant    : {mse_before:.8f}")
    print(f"  MSE apres    : {mse_after:.8f}")
    print(f"  Reduction    : {reduction:.8f}")
    print(f"  Pourcentage  : {pourcentage:.2f}%")
    print("")

    if reduction > 0:
        print("  RESULTAT : L'erreur a DIMINUE apres la mise a jour.")
        print("  La retropropagation a correctement ajuste les poids.")
    else:
        print("  RESULTAT : L'erreur n'a pas diminue (cas rare).")
        print("  Cela peut arriver si le gradient est tres petit.")
    print("")

    # ================================================================
    # Resume mathematique
    # ================================================================
    print("=" * 56)
    print("  RESUME MATHEMATIQUE")
    print("=" * 56)
    print("")
    print("  Etape 1 : Propagation avant")
    print("    h_j = sigma(SUM(w_{i,j} * x_i) + b_j)")
    print("    y_hat = SUM(w_j * h_j) + b")
    print("")
    print("  Etape 2 : Calcul de l'erreur")
    print("    E = (1/2) * (y - y_hat)^2")
    print("")
    print("  Etape 3 : Delta de sortie (Linear)")
    print("    delta_output = (y_hat - y) * 1")
    print("")
    print("  Etape 4 : Delta cache (Sigmoid)")
    print("    delta_j = delta_output * w_j * sigma'(z_j)")
    print("    ou sigma'(z_j) = h_j * (1 - h_j)")
    print("")
    print("  Etape 5 : Gradients")
    print("    dE/dw_{i,j} = x_i * delta_j    (entree -> cachee)")
    print("    dE/dw_j = h_j * delta_output     (cachee -> sortie)")
    print("")
    print("  Etape 6 : Mise a jour")
    print("    w_new = w_old - eta * gradient")
    print("=" * 56)
    print("")


if __name__ == "__main__":
    main()
