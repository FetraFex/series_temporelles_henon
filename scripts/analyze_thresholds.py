"""
Analyse de l'influence du seuil sur la détection du plateau.

Ce script teste plusieurs valeurs de seuil (threshold) pour la
détection du plateau dans la courbe d'erreur d'embedding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ajout du répertoire src au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from henon.core.embedding.plateau import find_first_plateau

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
MAX_DIM = 20  # On utilise les données de l'expérience n_max=20
OUTPUT_DIR = Path("results/threshold_analysis")


def main() -> None:
    """Teste différents seuils et analyse la détection du plateau."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Chargement des données d'erreur (expérience n_max=20)
    errors_file = Path("results/embedding_experiments/maxdim_20/embedding_errors.csv")
    df = pd.read_csv(errors_file)
    dimensions = df["dimension"].values
    errors = df["error"].values

    print("=" * 60)
    print(" ANALYSE DE L'INFLUENCE DU SEUIL SUR LA DÉTECTION DU PLATEAU")
    print("=" * 60)
    print(f"  Dimension maximale : {MAX_DIM}")
    print(f"  Seuils testés : {THRESHOLDS}")
    print("=" * 60)

    results = []

    for threshold in THRESHOLDS:
        # Détection du plateau
        optimal_dim = find_first_plateau(
            dimensions, errors,
            threshold=threshold,
            min_consecutive=2,
        )
        plateau_found = optimal_dim < MAX_DIM

        # Calcul des variations relatives pour analyse
        base_error = errors[0]
        relative_variations = np.abs(np.diff(errors)) / base_error

        # Nombre de variations sous le seuil
        under_threshold = np.sum(relative_variations < threshold)

        results.append({
            "threshold": threshold,
            "optimal_dimension": optimal_dim,
            "plateau_found": plateau_found,
            "under_threshold_count": int(under_threshold),
            "total_variations": len(relative_variations),
        })

        plateau_str = "Oui" if plateau_found else "Non"
        print(f"\n  Seuil = {threshold:.3f}")
        print(f"    Dimension optimale : {optimal_dim}")
        print(f"    Plateau trouvé : {plateau_str}")
        print(f"    Variations sous seuil : {under_threshold}/{len(relative_variations)}")

    # Tableau récapitulatif
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)

    # Graphique : dimension détectée en fonction du seuil
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[r["threshold"] for r in results],
        y=[r["optimal_dimension"] for r in results],
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=8),
        name="Dimension optimale",
    ))

    # Ligne horizontale pour la dimension maximale
    fig.add_hline(
        y=MAX_DIM,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"n_max = {MAX_DIM}",
        annotation_position="top right",
    )

    fig.update_layout(
        title=dict(
            text="Influence du seuil sur la dimension optimale détectée",
            font=dict(family="Times New Roman, serif", size=16),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Seuil (threshold)",
            type="log",
            showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)",
            showline=True, linewidth=1, linecolor="black",
        ),
        yaxis=dict(
            title="Dimension optimale détectée",
            range=[0, MAX_DIM + 2],
            showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)",
            showline=True, linewidth=1, linecolor="black",
        ),
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        width=800, height=500,
    )

    fig.write_image(str(OUTPUT_DIR / "threshold_effect.png"), scale=3)
    fig.write_html(str(OUTPUT_DIR / "threshold_effect.html"))

    # Graphique : variations relatives avec différents seuils
    fig2 = go.Figure()

    base_error = errors[0]
    relative_variations = np.abs(np.diff(errors)) / base_error
    dims_var = dimensions[1:]  # Les dimensions correspondantes aux variations

    fig2.add_trace(go.Scatter(
        x=dims_var.tolist(),
        y=relative_variations.tolist(),
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=5),
        name="Variation relative",
    ))

    # Lignes horizontales pour chaque seuil
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    for i, threshold in enumerate(THRESHOLDS):
        fig2.add_hline(
            y=threshold,
            line_dash="dot",
            line_color=colors[i % len(colors)],
            annotation_text=f"θ={threshold}",
            annotation_position="right",
            annotation_font=dict(size=10, color=colors[i % len(colors)]),
        )

    fig2.update_layout(
        title=dict(
            text="Variations relatives |E(m-1) - E(m)| / E(2) avec seuils",
            font=dict(family="Times New Roman, serif", size=16),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Dimension m",
            showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)",
            showline=True, linewidth=1, linecolor="black",
        ),
        yaxis=dict(
            title="Variation relative",
            type="log",
            showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)",
            showline=True, linewidth=1, linecolor="black",
        ),
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        width=900, height=600,
    )

    fig2.write_image(str(OUTPUT_DIR / "relative_variations.png"), scale=3)
    fig2.write_html(str(OUTPUT_DIR / "relative_variations.html"))

    # Rapport d'analyse
    lines = [
        "=" * 60,
        "ANALYSE DE L'INFLUENCE DU SEUIL SUR LA DÉTECTION DU PLATEAU",
        "=" * 60,
        "",
        "1. Données utilisées :",
        f"   Expérience n_max = {MAX_DIM}",
        f"   Nombre de points : {len(errors)}",
        f"   E(2) = {errors[0]:.6f}",
        f"   E({MAX_DIM}) = {errors[-1]:.6f}",
        "",
        "2. Résultats par seuil :",
        "",
        f"   {'Seuil':>8} | {'Dim optimale':>12} | {'Plateau':>8} | {'Sous seuil':>12}",
        f"   {'-'*8}-+-{'-'*12}-+-{'-'*8}-+-{'-'*12}",
    ]

    for r in results:
        plateau = "Oui" if r["plateau_found"] else "Non"
        lines.append(
            f"   {r['threshold']:>8.3f} | {r['optimal_dimension']:>12} | "
            f"{plateau:>8} | {r['under_threshold_count']:>5}/{r['total_variations']}"
        )

    # Trouver le premier seuil qui détecte un plateau
    first_plateau = next((r for r in results if r["plateau_found"]), None)

    lines.extend([
        "",
        "3. Analyse :",
        "",
    ])

    if first_plateau:
        lines.append(
            f"   Le premier plateau est détecté pour le seuil θ = {first_plateau['threshold']:.3f}"
        )
        lines.append(
            f"   → Dimension optimale = {first_plateau['optimal_dimension']}"
        )
        lines.append("")
        lines.append(
            "   Cela suggère que le seuil original (0.001) était trop strict"
        )
        lines.append(
            "   pour cette série de Hénon. Un seuil plus élevé est nécessaire"
        )
        lines.append(
            "   pour capturer la zone de ralentissement relatif."
        )
    else:
        lines.append(
            "   Aucun plateau n'a été détecté même avec des seuils élevés."
        )
        lines.append(
            "   Cela confirme que la courbe E(m) décroît continûment"
        )
        lines.append(
            "   sans stagnation, caractéristique d'un spectre continu."
        )

    lines.extend([
        "",
        "4. Recommandation :",
        "",
        "   Pour la suite du projet (RNN), il est recommandé de :",
        "   - Utiliser une dimension d'embedding de 4-5 (valeur typique pour Hénon)",
        "   - Ou utiliser le seuil optimal trouvé dans cette analyse",
        "",
    ])

    with open(OUTPUT_DIR / "analysis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'=' * 60}")
    print(" RÉSULTATS")
    print(f"{'=' * 60}")
    print(f"  Fichiers générés dans {OUTPUT_DIR}/")
    print(f"  - threshold_analysis.csv")
    print(f"  - threshold_effect.png / .html")
    print(f"  - relative_variations.png / .html")
    print(f"  - analysis.txt")


if __name__ == "__main__":
    main()
