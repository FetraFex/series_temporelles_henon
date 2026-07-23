"""
Étude expérimentale de l'influence de n_max sur la courbe de Takens.

Ce script exécute l'algorithme de Takens pour plusieurs valeurs de
max_dimension et compare les courbes obtenues. L'algorithme lui-même
n'est PAS modifié — seule la valeur de max_dimension change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ajout du répertoire src au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from henon.core.embedding.covariance import compute_covariance_matrix
from henon.core.embedding.dimension import build_delay_vectors
from henon.core.embedding.eigen import compute_eigendecomposition
from henon.core.embedding.error import compute_embedding_error
from henon.core.embedding.plateau import find_first_plateau
from henon.core.dynamical_systems.henon import generate_henon_series
from henon.utils.config import load_config

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MAX_DIMENSIONS = [20, 30, 40, 50, 60, 70, 80, 100]
OUTPUT_ROOT = Path("results/embedding_experiments")

# Couleurs pour la comparaison graphique
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# ---------------------------------------------------------------------------
# Fonction d'expérience pour une seule valeur de n_max
# ---------------------------------------------------------------------------


def run_single_experiment(
    series: np.ndarray,
    max_dim: int,
    config: dict,
) -> dict:
    """
    Exécute l'algorithme de Takens pour une valeur donnée de n_max.

    Parameters
    ----------
    series : np.ndarray
        Série temporelle de Hénon.
    max_dim : int
        Dimension maximale candidate (n_max).
    config : dict
        Configuration Takens chargée depuis takens.yaml.

    Returns
    -------
    dict
        Résultats : dimensions, errors, eigenvalues, cov_matrix,
        optimal_dimension, plateau_found.
    """
    delay = config["delay"]
    min_dim = config["min_dimension"]
    center = config["center_covariance"]

    # Étape 1 : Construction des vecteurs retardés (UNE SEULE FOIS)
    delay_vectors = build_delay_vectors(series, delay, max_dim)

    # Étape 2 : Matrice de covariance (UNE SEULE FOIS)
    cov_matrix = compute_covariance_matrix(delay_vectors, center=center)

    # Étape 3 : Décomposition spectrale (UNE SEULE FOIS)
    eigenvalues, _ = compute_eigendecomposition(cov_matrix)

    # Étape 4 : Calcul vectorisé de l'erreur pour chaque m
    # E(m) = √(λ_{m+1}) — erreur basée sur la première composante écartée
    dimensions = np.arange(min_dim, max_dim + 1)
    errors = np.zeros(len(dimensions), dtype=float)

    for idx, m in enumerate(dimensions):
        if m < len(eigenvalues):
            errors[idx] = np.sqrt(eigenvalues[m])
        else:
            errors[idx] = 0.0

    # Étape 5 : Détection du plateau
    optimal_dim = find_first_plateau(
        dimensions, errors,
        threshold=config["plateau_threshold"],
        min_consecutive=config["plateau_min_consecutive"],
    )

    # Vérification : le plateau est-il atteint (ou est-ce max_dim par défaut ?)
    plateau_found = optimal_dim < max_dim

    return {
        "dimensions": dimensions,
        "errors": errors,
        "eigenvalues": eigenvalues,
        "cov_matrix": cov_matrix,
        "optimal_dimension": optimal_dim,
        "plateau_found": plateau_found,
        "error_min": float(errors[-1]),
    }


# ---------------------------------------------------------------------------
# Sauvegarde des résultats d'une expérience
# ---------------------------------------------------------------------------


def save_experiment_results(
    max_dim: int,
    results: dict,
    output_dir: Path,
) -> None:
    """
    Sauvegarde les résultats d'une expérience dans le dossier dédié.

    Parameters
    ----------
    max_dim : int
        Valeur de n_max testée.
    results : dict
        Résultats retournés par run_single_experiment().
    output_dir : Path
        Dossier de sortie (ex: results/embedding_experiments/maxdim_20/).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Matrice de covariance
    pd.DataFrame(results["cov_matrix"]).to_csv(
        output_dir / "covariance_matrix.csv", index=False
    )

    # Valeurs propres
    pd.DataFrame({
        "eigenvalue": results["eigenvalues"],
        "index": range(len(results["eigenvalues"])),
    }).to_csv(output_dir / "eigenvalues.csv", index=False)

    # Erreurs d'embedding
    pd.DataFrame({
        "dimension": results["dimensions"],
        "error": results["errors"],
    }).to_csv(output_dir / "embedding_errors.csv", index=False)

    # Graphique E(m)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results["dimensions"].tolist(),
        y=results["errors"].tolist(),
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=5),
        showlegend=False,
    ))
    fig.add_vline(
        x=results["optimal_dimension"],
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Dim optimale = {results['optimal_dimension']}",
        annotation_position="top right",
        annotation_font=dict(size=12, color="#d62728"),
    )
    fig.update_layout(
        title=dict(
            text=f"Erreur d'embedding — n_max = {max_dim}",
            font=dict(family="Times New Roman, serif", size=16),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Dimension m",
            showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)",
            dtick=5 if max_dim > 40 else 2,
            showline=True, linewidth=1, linecolor="black",
        ),
        yaxis=dict(
            title="E(m)",
            showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)",
            showline=True, linewidth=1, linecolor="black",
        ),
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        width=800, height=500,
    )
    fig.write_image(str(output_dir / "embedding_error.png"), scale=3)
    fig.write_html(str(output_dir / "embedding_error.html"))

    # Fichier résumé
    plateau_str = "Oui" if results["plateau_found"] else "Non"
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"n_max = {max_dim}\n")
        f.write(f"Dimension detectee : {results['optimal_dimension']}\n")
        f.write(f"Plateau trouve : {plateau_str}\n")
        f.write(f"Erreur minimale : {results['error_min']:.10f}\n")
        f.write(f"Nombre de valeurs propres : {len(results['eigenvalues'])}\n")


# ---------------------------------------------------------------------------
# Comparaison graphique de toutes les courbes
# ---------------------------------------------------------------------------


def plot_comparison(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """
    Trace toutes les courbes E(m) sur un même graphique.

    Parameters
    ----------
    all_results : dict[int, dict]
        Dictionnaire {max_dim: résultats} pour chaque expérience.
    output_dir : Path
        Dossier de sortie.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()

    for idx, (max_dim, res) in enumerate(sorted(all_results.items())):
        fig.add_trace(go.Scatter(
            x=res["dimensions"].tolist(),
            y=res["errors"].tolist(),
            mode="lines+markers",
            name=f"n_max = {max_dim}",
            line=dict(color=COLORS[idx % len(COLORS)], width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title=dict(
            text="Influence de la dimension maximale testee sur la courbe de Takens",
            font=dict(family="Times New Roman, serif", size=16),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Dimension m",
            showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)",
            showline=True, linewidth=1, linecolor="black",
        ),
        yaxis=dict(
            title="E(m)",
            showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)",
            showline=True, linewidth=1, linecolor="black",
        ),
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(
            font=dict(family="Times New Roman, serif", size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        width=900, height=600,
        margin=dict(l=60, r=30, t=60, b=60),
    )

    fig.write_image(str(output_dir / "comparison_curves.png"), scale=3)
    fig.write_html(str(output_dir / "comparison_curves.html"))


# ---------------------------------------------------------------------------
# Tableau récapitulatif
# ---------------------------------------------------------------------------


def save_summary_table(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """
    Crée le fichier summary.csv avec un récapitulatif de toutes les expériences.

    Parameters
    ----------
    all_results : dict[int, dict]
        Dictionnaire {max_dim: résultats}.
    output_dir : Path
        Dossier de sortie.
    """
    rows = []
    for max_dim in sorted(all_results.keys()):
        res = all_results[max_dim]
        rows.append({
            "max_dimension": max_dim,
            "dimension_detectee": res["optimal_dimension"],
            "erreur_min": f"{res['error_min']:.10f}",
            "plateau_trouve": "Oui" if res["plateau_found"] else "Non",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "summary.csv", index=False)


# ---------------------------------------------------------------------------
# Rapport d'analyse
# ---------------------------------------------------------------------------


def save_analysis(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """
    Génère le rapport d'analyse analysis.txt.

    Parameters
    ----------
    all_results : dict[int, dict]
        Dictionnaire {max_dim: résultats}.
    output_dir : Path
        Dossier de sortie.
    """
    sorted_dims = sorted(all_results.keys())

    lines = [
        "=" * 60,
        "ANALYSE DE L'INFLUENCE DE n_max SUR LA COURBE DE TAKENS",
        "=" * 60,
        "",
        "1. Valeurs de n_max testees :",
        f"   {', '.join(str(d) for d in sorted_dims)}",
        "",
        "2. Resultats par experience :",
        "",
        f"   {'n_max':>8} | {'Dim detectee':>14} | {'Erreur min':>14} | {'Plateau':>8}",
        f"   {'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}",
    ]

    for max_dim in sorted_dims:
        res = all_results[max_dim]
        plateau = "Oui" if res["plateau_found"] else "Non"
        lines.append(
            f"   {max_dim:>8} | {res['optimal_dimension']:>14} | "
            f"{res['error_min']:>14.10f} | {plateau:>8}"
        )

    # Stabilité des résultats
    detected_dims = [all_results[d]["optimal_dimension"] for d in sorted_dims]
    plateau_dims = [d for d in sorted_dims if all_results[d]["plateau_found"]]

    lines.extend([
        "",
        "3. Stabilite des resultats :",
        "",
        f"   Dimensions detectees : {detected_dims}",
        f"   Experiences avec plateau : {plateau_dims if plateau_dims else 'Aucune'}",
        "",
    ])

    if plateau_dims:
        first_plateau = min(plateau_dims)
        stable_dims = detected_dims[sorted_dims.index(first_plateau):]
        all_same = len(set(stable_dims)) == 1
        lines.append(
            f"   Le premier plateau apparait pour n_max = {first_plateau}."
        )
        if all_same:
            lines.append(
                f"   A partir de n_max = {first_plateau}, la dimension detectee "
                f"reste stable a {stable_dims[0]}."
            )
        else:
            lines.append(
                f"   La dimension detectee varie encore apres n_max = {first_plateau} : "
                f"{stable_dims}."
            )
    else:
        lines.append(
            "   Aucun plateau n'a ete detecte pour les valeurs testees."
        )
        lines.append(
            "   Cela peut indiquer que la serie de Henon n'a pas de plateau"
        )
        lines.append(
            "   clair dans l'intervalle [2, n_max] avec le seuil configure."
        )

    lines.extend([
        "",
        "4. Conclusion :",
        "",
        "   L'etude experimentale permet de determiner objectivement si",
        "   l'absence de plateau observee pour n_max=20 est due a une limite",
        "   de l'intervalle explore ou a une propriete de la serie elle-meme.",
        "",
    ])

    with open(output_dir / "analysis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------


def main() -> None:
    """Exécute la campagne expérimentale complète."""
    # Chargement de la configuration
    config = load_config("takens")

    # Chargement de la série de Hénon
    xlsx_path = Path("data/raw/henon_500.xlsx")
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Fichier non trouve : {xlsx_path}. "
            f"Executez d'abord scripts/run_generation.py."
        )
    df = pd.read_excel(xlsx_path)
    series = df["Xn"].values.astype(float)

    print("=" * 55)
    print(" ETUDE EXPERIMENTALE : INFLUENCE DE n_max SUR TAKENS")
    print("=" * 55)
    print(f"  Serie : {len(series)} points")
    print(f"  Delay : r = {config['delay']}")
    print(f"  Dimensions candidates : {config['min_dimension']} a {max(MAX_DIMENSIONS)}")
    print(f"  Seuil plateau : {config['plateau_threshold']}")
    print("=" * 55)

    all_results: dict[int, dict] = {}
    total = len(MAX_DIMENSIONS)

    for i, max_dim in enumerate(MAX_DIMENSIONS, 1):
        print(f"\n{'=' * 55}")
        print(f" EXPERIENCE {i}/{total}")
        print(f" n_max = {max_dim}")
        print(f"{'=' * 55}")

        try:
            # Exécution de l'algorithme
            results = run_single_experiment(series, max_dim, config)
            print(f"  OK Vecteurs retardés construits ({results['dimensions'][-1]} dimensions)")
            print(f"  OK Covariance calculée ({results['cov_matrix'].shape})")
            print(f"  OK Valeurs propres calculées ({len(results['eigenvalues'])} valeurs)")
            print(f"  OK Courbe générée ({len(results['dimensions'])} points)")
            plateau_str = "Oui" if results["plateau_found"] else "Non"
            print(f"  OK Plateau détecté : {plateau_str}")
            if results["plateau_found"]:
                print(f"     Dimension optimale : {results['optimal_dimension']}")

            # Sauvegarde
            output_dir = OUTPUT_ROOT / f"maxdim_{max_dim}"
            save_experiment_results(max_dim, results, output_dir)
            print(f"  OK Résultats sauvegardés dans {output_dir}/")

            all_results[max_dim] = results

        except ValueError as e:
            print(f"  ERREUR : {e}")
            print(f"  Experience {max_dim} sautée (dimension trop grande pour la série)")

    # Comparaison graphique
    if all_results:
        print(f"\n{'=' * 55}")
        print(" GENERATION DES GRAPHIQUES COMPARATIFS")
        print(f"{'=' * 55}")
        plot_comparison(all_results, OUTPUT_ROOT)
        print("  OK comparison_curves.png / .html")

        # Tableau récapitulatif
        save_summary_table(all_results, OUTPUT_ROOT)
        print("  OK summary.csv")

        # Rapport d'analyse
        save_analysis(all_results, OUTPUT_ROOT)
        print("  OK analysis.txt")

    print(f"\n{'=' * 55}")
    print(" TERMINE")
    print(f"{'=' * 55}")
    print(f"  Résultats dans {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
