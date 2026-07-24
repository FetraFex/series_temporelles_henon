"""
Script d'initialisation et de visualisation du réseau de neurones.

Ce script instancie le MLP 6-8-1 avec activation Sigmoid/Linear,
affiche un résumé complet de l'initialisation, exporte les résultats
et génère un diagramme de l'architecture.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from henon.core.neural.network import Network

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
OUTPUT_DIR = Path("results/training")
ARCHITECTURE = [6, 8, 1]
ACTIVATIONS = ["sigmoid", "linear"]
SEED = 42
LEARNING_RATE = 0.1
MAX_EPOCHS = 1000
TOLERANCE = 1e-5


def print_summary(
    network: Network,
    weights_ih: np.ndarray,
    weights_ho: np.ndarray,
    bias_h: np.ndarray,
    bias_o: np.ndarray,
) -> str:
    """
    Affiche et retourne le résumé de l'initialisation.

    Parameters
    ----------
    network : Network
        Réseau initialisé.
    weights_ih : np.ndarray
        Poids Input→Hidden (6, 8).
    weights_ho : np.ndarray
        Poids Hidden→Output (8, 1).
    bias_h : np.ndarray
        Biais Hidden (8,).
    bias_o : np.ndarray
        Biais Output (1,).

    Returns
    -------
    str
        Texte du résumé complet.
    """
    n_weights_ih = weights_ih.size
    n_weights_ho = weights_ho.size
    n_bias_h = bias_h.size
    n_bias_o = bias_o.size
    total_weights = n_weights_ih + n_weights_ho
    total_biases = n_bias_h + n_bias_o
    total_params = total_weights + total_biases

    lines = [
        "",
        "=" * 42,
        "  NETWORK INITIALIZATION",
        "=" * 42,
        "",
        f"  Architecture       : {'-'.join(map(str, ARCHITECTURE))}",
        "",
        f"  Input neurons      : {ARCHITECTURE[0]}",
        f"  Hidden neurons     : {ARCHITECTURE[1]}",
        f"  Output neurons     : {ARCHITECTURE[2]}",
        "",
        f"  Hidden activation  : Sigmoid",
        f"  Output activation  : Linear",
        "",
        f"  Learning rate      : {LEARNING_RATE}",
        f"  Max epochs         : {MAX_EPOCHS}",
        f"  Tolerance          : {TOLERANCE}",
        "",
        f"  Weights Input->Hidden  : {n_weights_ih}",
        f"  Weights Hidden->Output : {n_weights_ho}",
        "",
        f"  Total weights      : {total_weights}",
        "",
        f"  Bias Hidden        : {n_bias_h}",
        f"  Bias Output        : {n_bias_o}",
        "",
        f"  Total biases       : {total_biases}",
        "",
        f"  Total parameters   : {total_params}",
        "",
        "-" * 42,
        "  PREMIERS POIDS (Input -> Hidden)",
        "-" * 42,
    ]

    for i in range(min(5, weights_ih.size)):
        idx = np.unravel_index(i, weights_ih.shape)
        lines.append(f"  w{i + 1} = {weights_ih[idx]:.8f}")

    lines.extend([
        "",
        "-" * 42,
        "  PREMIERS POIDS (Hidden -> Output)",
        "-" * 42,
    ])

    for i in range(min(5, weights_ho.size)):
        idx = np.unravel_index(i, weights_ho.shape)
        lines.append(f"  w{i + 1} = {weights_ho[idx]:.8f}")

    lines.extend([
        "",
        "-" * 42,
        "  PREMIERS BIAIS (Hidden)",
        "-" * 42,
    ])

    for i in range(min(5, bias_h.size)):
        lines.append(f"  b{i + 1} = {bias_h[i]:.8f}")

    lines.extend([
        "",
        "-" * 42,
        "  BIAIS (Output)",
        "-" * 42,
    ])

    for i in range(bias_o.size):
        lines.append(f"  b{i + 1} = {bias_o[i]:.8f}")

    lines.extend(["", "=" * 42, ""])

    text = "\n".join(lines)
    print(text)
    return text


def save_json(
    network: Network,
    weights_ih: np.ndarray,
    weights_ho: np.ndarray,
    bias_h: np.ndarray,
    bias_o: np.ndarray,
    filepath: Path,
) -> None:
    """
    Exporte le résumé en JSON.

    Parameters
    ----------
    network : Network
        Réseau initialisé.
    weights_ih : np.ndarray
        Poids Input→Hidden.
    weights_ho : np.ndarray
        Poids Hidden→Output.
    bias_h : np.ndarray
        Biais Hidden.
    bias_o : np.ndarray
        Biais Output.
    filepath : Path
        Chemin du fichier JSON.
    """
    summary = {
        "architecture": ARCHITECTURE,
        "activations": {
            "hidden": "sigmoid",
            "output": "linear",
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "tolerance": TOLERANCE,
        },
        "weights": {
            "input_hidden": weights_ih.tolist(),
            "hidden_output": weights_ho.tolist(),
        },
        "biases": {
            "hidden": bias_h.tolist(),
            "output": bias_o.tolist(),
        },
        "counts": {
            "weights_input_hidden": int(weights_ih.size),
            "weights_hidden_output": int(weights_ho.size),
            "total_weights": int(weights_ih.size + weights_ho.size),
            "bias_hidden": int(bias_h.size),
            "bias_output": int(bias_o.size),
            "total_biases": int(bias_h.size + bias_o.size),
            "total_parameters": int(
                weights_ih.size + weights_ho.size + bias_h.size + bias_o.size
            ),
        },
        "seed": SEED,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Résumé JSON exporté : {filepath}")


def create_diagram(output_dir: Path) -> None:
    """
    Génère le diagramme de l'architecture du réseau.

    Parameters
    ----------
    output_dir : Path
        Répertoire de sortie.
    """
    layers = ARCHITECTURE
    layer_names = ["Input\n(6)", "Hidden\n(8)", "Output\n(1)"]
    layer_colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    fig = go.Figure()

    # Positions des neurones
    x_positions = [0, 1, 2]
    y_positions = []

    for i, n in enumerate(layers):
        # Centrer verticalement
        start_y = -(n - 1) / 2
        y_pos = [start_y + j for j in range(n)]
        y_positions.append(y_pos)

    # Dessiner les connexions (liens)
    for l in range(len(layers) - 1):
        for i in range(layers[l]):
            for j in range(layers[l + 1]):
                x0, y0 = x_positions[l], y_positions[l][i]
                x1, y1 = x_positions[l + 1], y_positions[l + 1][j]
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color="rgba(150, 150, 150, 0.3)", width=0.5),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # Dessiner les neurones
    for l in range(len(layers)):
        fig.add_trace(go.Scatter(
            x=[x_positions[l]] * layers[l],
            y=y_positions[l],
            mode="markers",
            marker=dict(
                size=20,
                color=layer_colors[l],
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
            hovertext=[
                f"Layer {l}, Neuron {i}<br>"
                f"Position: ({x_positions[l]}, {y_positions[l][i]:.1f})"
                for i in range(layers[l])
            ],
            hoverinfo="text",
        ))

    # Labels des couches
    for l in range(len(layers)):
        fig.add_annotation(
            x=x_positions[l],
            y=max(y_positions[l]) + 1.2,
            text=layer_names[l],
            showarrow=False,
            font=dict(size=13, color=layer_colors[l]),
        )

    # Titre et layout
    fig.update_layout(
        title=dict(
            text=(
                "Architecture du Réseau de Neurones<br>"
                "<sup>MLP 6-8-1 — Sigmoid (cachée) / Linear (sortie)</sup>"
            ),
            font=dict(family="Times New Roman, serif", size=16),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, 2.5],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-5.5, 5.5],
            scaleanchor="x",
            scaleratio=1,
        ),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=900,
        height=600,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    # Export
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_dir / "network_diagram.png"), scale=3)
    fig.write_image(str(output_dir / "network_diagram.svg"))
    fig.write_html(str(output_dir / "network_diagram.html"))

    logger.info(f"Diagramme sauvegardé dans {output_dir}")


def main() -> None:
    """Exécute l'initialisation et la visualisation du réseau."""
    logger.info("Initialisation du réseau 6-8-1...")

    # --- Étape 1 : Instanciation du réseau ---
    network = Network(
        layer_sizes=ARCHITECTURE,
        activations=ACTIVATIONS,
        seed=SEED,
    )

    # --- Étape 2 : Extraction des poids et biais ---
    # DenseLayer[1] = couche cachée (Input→Hidden)
    # DenseLayer[2] = couche de sortie (Hidden→Output)
    dense_layers = [l for l in network.layers if hasattr(l, "weights")]

    weights_ih = dense_layers[0].weights  # (6, 8)
    bias_h = dense_layers[0].bias         # (8,)
    weights_ho = dense_layers[1].weights  # (8, 1)
    bias_o = dense_layers[1].bias         # (1,)

    # --- Étape 3 : Affichage du résumé ---
    summary_text = print_summary(
        network, weights_ih, weights_ho, bias_h, bias_o,
    )

    # --- Étape 4 : Export ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Texte
    txt_path = OUTPUT_DIR / "network_initialization.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"Fichier texte exporté : {txt_path}")

    # JSON
    json_path = OUTPUT_DIR / "network_summary.json"
    save_json(network, weights_ih, weights_ho, bias_h, bias_o, json_path)

    # --- Étape 5 : Diagramme ---
    create_diagram(OUTPUT_DIR)

    logger.info("Terminé.")


if __name__ == "__main__":
    main()
