"""Script d'exécution du pipeline d'embedding de Takens."""

import logging

from henon.pipelines.embed_pipeline import run_embedding_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    """Exécute le pipeline complet de Takens et affiche les résultats."""
    results = run_embedding_pipeline()

    print("\n" + "=" * 50)
    print("  RÉSULTATS DE L'EMBEDDING DE TAKENS")
    print("=" * 50)
    print(f"  Dimension optimale : {results['optimal_dimension']}")
    print(f"  Dimensions testees : {results['dimensions'][0]} -> {results['dimensions'][-1]}")
    print(f"  Erreur minimale    : {results['errors'][-1]:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
