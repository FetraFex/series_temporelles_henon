"""
Visualisation de l'erreur d'embedding.

Ce module trace la courbe E(m) en fonction de la dimension m
et affiche la dimension optimale détectée.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def plot_embedding_error(
    dimensions: list[int] | list[float],
    errors: list[float],
    optimal_dimension: int,
    output_dir: str | Path = "results/embedding",
    show: bool = True,
) -> go.Figure:
    """
    Trace la courbe de l'erreur d'embedding E(m) en fonction de m.

    Parameters
    ----------
    dimensions : list[int] | list[float]
        Dimensions testées.
    errors : list[float]
        Erreurs correspondantes E(m).
    optimal_dimension : int
        Dimension optimale détectée (marquée sur le graphique).
    output_dir : str | Path
        Dossier de sortie pour les exports.
    show : bool
        Si True, affiche le graphique à l'écran.

    Returns
    -------
    go.Figure
        Objet figure Plotly configuré.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Courbe principale E(m)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dimensions),
        y=list(errors),
        mode="lines+markers",
        name="E(m)",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=6, color="#1f77b4"),
    ))

    # Ligne verticale à la dimension optimale
    fig.add_vline(
        x=optimal_dimension,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Dimension optimale = {optimal_dimension}",
        annotation_position="top right",
        annotation_font=dict(size=13, color="#d62728"),
    )

    # Mise en page
    fig.update_layout(
        title=dict(
            text="Erreur d'embedding — Méthode ACP (Takens)",
            font=dict(family="Times New Roman, serif", size=18),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="Dimension m", font=dict(family="Times New Roman, serif", size=14)),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="rgba(0,0,0,0.1)",
            dtick=1,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
        ),
        yaxis=dict(
            title=dict(text="E(m)", font=dict(family="Times New Roman, serif", size=14)),
            showgrid=True,
            gridwidth=0.5,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            ticks="outside",
        ),
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=800,
        height=500,
        margin=dict(l=60, r=30, t=60, b=60),
        showlegend=False,
    )

    # Export PNG haute résolution
    fig.write_image(str(output_path / "embedding_error.png"), scale=3)

    # Export SVG vectoriel
    fig.write_image(str(output_path / "embedding_error.svg"))

    # Export HTML interactif
    fig.write_html(str(output_path / "embedding_error.html"))

    if show:
        fig.show()

    return fig
