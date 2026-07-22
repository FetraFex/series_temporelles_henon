"""Script de démonstration des 8 styles de l'attracteur de Hénon."""

from henon import generate_henon_series
from henon.visualization import compare_styles, plot_henon_attractor


def main() -> None:
    """Génère les données et trace l'attracteur avec chaque style."""

    X, Y = generate_henon_series(iterations=500, x0=0.0, y0=0.0, a=1.4, b=0.3)

    # Style par défaut (mémoire)
    print("Style par défaut : scientific")
    plot_henon_attractor(X, Y, output_dir="results/figures")

    # Comparaison des 8 styles
    print("\nGénération des 8 styles :")
    compare_styles(X, Y, output_dir="results/figures/styles")


if __name__ == "__main__":
    main()
