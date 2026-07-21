"""
Visualisation de l'attracteur de Hénon.

Ce module fournit une fonction pour tracer et exporter
la représentation graphique yₙ = f(xₙ) du système de Hénon
à l'aide de Plotly.
"""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go


def plot_henon_attractor(
    x_values: list[float],
    y_values: list[float],
    output_dir: str | Path = "results/figures",
    color: str = "#1f77b4",
    marker_size: float = 1.5,
    opacity: float = 0.6,
) -> go.Figure:
    """
    Trace et exporte l'attracteur de Hénon : yₙ = f(xₙ).

    Parameters
    ----------
    x_values : list[float]
        Séquence des valeurs xₙ du système de Hénon.
    y_values : list[float]
        Séquence des valeurs yₙ du système de Hénon.
    output_dir : str | Path
        Répertoire de sortie pour les figures exportées.
    color : str
        Code couleur des points (hex ou nom CSS).
    marker_size : float
        Taille des marqueurs sur le graphique.
    opacity : float
        Transparence des points (0.0 à 1.0).

    Returns
    -------
    go.Figure
        Objet figure Plotly configuré et affiché.

    Raises
    ------
    ValueError
        Si les listes x_values et y_values ont des tailles différentes
        ou si l'une d'entre elles est vide.
    """

    # --- Validation des entrées ---
    if len(x_values) != len(y_values):
        raise ValueError(
            f"Les listes doivent avoir la même taille : "
            f"x_values ({len(x_values)}) ≠ y_values ({len(y_values)})."
        )

    if len(x_values) == 0:
        raise ValueError(
            "Les listes x_values et y_values ne doivent pas être vides."
        )

    # --- Création du dossier de sortie ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Construction du scatter plot ---
    fig = px.scatter(
        x=x_values,
        y=y_values,
        template="plotly_white",
        labels={"x": "xₙ", "y": "yₙ"},
    )

    # Mise à jour du trace pour un rendu publication scientifique
    fig.update_traces(
        marker=dict(
            color=color,
            size=marker_size,
            opacity=opacity,
        ),
        showlegend=False,
    )

    # --- Mise en page académique ---
    fig.update_layout(
        title=dict(
            text="Attracteur de Hénon",
            font=dict(family="Times New Roman, serif", size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="xₙ", font=dict(family="Times New Roman, serif", size=14)),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
        ),
        yaxis=dict(
            title=dict(text="yₙ", font=dict(family="Times New Roman, serif", size=14)),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=800,
        height=600,
        margin=dict(l=60, r=30, t=60, b=60),
    )

    # --- Export en PNG haute résolution ---
    png_path = output_path / "henon_attractor.png"
    fig.write_image(str(png_path), scale=3)

    # --- Export en SVG vectoriel ---
    svg_path = output_path / "henon_attractor.svg"
    fig.write_image(str(svg_path))

    # --- Affichage à l'écran ---
    fig.show()

    return fig
