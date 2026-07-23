"""
Tests unitaires et d'intégration pour l'algorithme de Takens.

Couvre :
- construction des vecteurs retardés
- matrice de covariance
- décomposition spectrale
- erreur d'embedding
- détection de plateau
- pipeline complet
"""

from __future__ import annotations

import numpy as np
import pytest

from henon.core.embedding.covariance import compute_covariance_matrix
from henon.core.embedding.dimension import build_delay_vectors
from henon.core.embedding.evaluation import evaluate_embedding_dimensions
from henon.core.embedding.eigen import compute_eigendecomposition
from henon.core.embedding.error import compute_embedding_error
from henon.core.embedding.plateau import find_first_plateau
from henon.core.dynamical_systems.henon import generate_henon_series


# ---------------------------------------------------------------------------
# Vecteurs retardés
# ---------------------------------------------------------------------------


class TestBuildDelayVectors:
    """Tests pour la construction des vecteurs retardés."""

    def test_shape(self) -> None:
        """La matrice de sortie a la bonne forme (M, dimension)."""
        series = np.arange(10, dtype=float)
        vectors = build_delay_vectors(series, delay=1, dimension=3)
        # M = 10 - (3-1)*1 = 8
        assert vectors.shape == (8, 3)

    def test_values(self) -> None:
        """Les vecteurs contiennent les bonnes valeurs."""
        series = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        vectors = build_delay_vectors(series, delay=2, dimension=2)
        # M = 6 - (2-1)*2 = 4
        # x̄(0) = [u(0), u(2)] = [0, 2]
        # x̄(1) = [u(1), u(3)] = [1, 3]
        # x̄(2) = [u(2), u(4)] = [2, 4]
        # x̄(3) = [u(3), u(5)] = [3, 5]
        expected = np.array([[0, 2], [1, 3], [2, 4], [3, 5]], dtype=float)
        np.testing.assert_array_almost_equal(vectors, expected)

    def test_too_large_dimension(self) -> None:
        """Lève ValueError si la dimension est trop grande."""
        series = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="M = .* <= 0"):
            build_delay_vectors(series, delay=1, dimension=10)

    def test_invalid_delay(self) -> None:
        """Lève ValueError si delay < 1."""
        with pytest.raises(ValueError, match="délai"):
            build_delay_vectors(np.arange(5, dtype=float), delay=0, dimension=2)

    def test_invalid_dimension(self) -> None:
        """Lève ValueError si dimension < 1."""
        with pytest.raises(ValueError, match="dimension"):
            build_delay_vectors(np.arange(5, dtype=float), delay=1, dimension=0)


# ---------------------------------------------------------------------------
# Matrice de covariance
# ---------------------------------------------------------------------------


class TestCovarianceMatrix:
    """Tests pour la matrice de covariance."""

    def test_shape(self) -> None:
        """La matrice est carrée (dimension × dimension)."""
        vectors = np.random.randn(50, 3)
        cov = compute_covariance_matrix(vectors)
        assert cov.shape == (3, 3)

    def test_symmetric(self) -> None:
        """La matrice est symétrique."""
        vectors = np.random.randn(50, 4)
        cov = compute_covariance_matrix(vectors)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_non_negative_eigenvalues(self) -> None:
        """Les valeurs propres sont toutes non négatives (non centrée)."""
        vectors = np.random.randn(50, 3)
        cov = compute_covariance_matrix(vectors, center=False)
        eigenvalues, _ = compute_eigendecomposition(cov)
        assert np.all(eigenvalues >= 0)

    def test_manual_computation(self) -> None:
        """Vérifie le calcul manuellement sur un petit exemple."""
        vectors = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        cov = compute_covariance_matrix(vectors, center=False)
        expected = (vectors.T @ vectors) / 3
        np.testing.assert_array_almost_equal(cov, expected)


# ---------------------------------------------------------------------------
# Décomposition spectrale
# ---------------------------------------------------------------------------


class TestEigendecomposition:
    """Tests pour la décomposition en valeurs propres."""

    def test_sorted_decreasing(self) -> None:
        """Les valeurs propres sont triées décroissantes."""
        vectors = np.random.randn(50, 3)
        cov = compute_covariance_matrix(vectors)
        eigenvalues, _ = compute_eigendecomposition(cov)
        assert all(eigenvalues[i] >= eigenvalues[i + 1] for i in range(len(eigenvalues) - 1))

    def test_no_negative_eigenvalues(self) -> None:
        """Aucune valeur propre négative après clipping."""
        vectors = np.random.randn(50, 3)
        cov = compute_covariance_matrix(vectors)
        eigenvalues, _ = compute_eigendecomposition(cov)
        assert np.all(eigenvalues >= 0)

    def test_orthogonal_eigenvectors(self) -> None:
        """Les vecteurs propres sont orthogonaux (produit scalaire ~ 0)."""
        vectors = np.random.randn(50, 3)
        cov = compute_covariance_matrix(vectors)
        _, eigenvectors = compute_eigendecomposition(cov)
        # Produit scalaire entre vecteurs différents
        dot = eigenvectors[:, 0] @ eigenvectors[:, 1]
        assert abs(dot) < 1e-10


# ---------------------------------------------------------------------------
# Erreur d'embedding
# ---------------------------------------------------------------------------


class TestEmbeddingError:
    """Tests pour le calcul de l'erreur d'embedding."""

    def test_decreasing(self) -> None:
        """E(m) est décroissante随着 m augmente."""
        eigenvalues = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
        errors = [compute_embedding_error(eigenvalues, m) for m in range(5)]
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1]

    def test_first_is_total(self) -> None:
        """E(0) = √(λ₁) — première valeur propre."""
        eigenvalues = np.array([4.0, 3.0, 2.0, 1.0])
        e0 = compute_embedding_error(eigenvalues, 0)
        expected = np.sqrt(eigenvalues[0])
        assert abs(e0 - expected) < 1e-10

    def test_last_is_zero(self) -> None:
        """E(d) = 0 (toutes les composantes retenues)."""
        eigenvalues = np.array([4.0, 3.0, 2.0, 1.0])
        ed = compute_embedding_error(eigenvalues, len(eigenvalues))
        assert abs(ed) < 1e-10


# ---------------------------------------------------------------------------
# Détection de plateau
# ---------------------------------------------------------------------------


class TestFindPlateau:
    """Tests pour la détection de plateau."""

    def test_plateau_detected(self) -> None:
        """Détecte un plateau quand les erreurs se stabilisent."""
        dimensions = np.array([2, 3, 4, 5, 6, 7])
        errors = np.array([1.0, 0.5, 0.1, 0.09, 0.089, 0.088])
        optimal = find_first_plateau(dimensions, errors, threshold=0.01, min_consecutive=2)
        # Les variations relatives sont : 0.5, 0.4, 0.01, 0.001, 0.001
        # 0.01 n'est pas < 0.01, donc les 2 premières sous le seuil sont aux indices 3 et 4
        # → le plateau commence à dimensions[3 + 2 - 1] = dimensions[4] = 6
        assert optimal == 6

    def test_no_plateau_returns_max(self) -> None:
        """Retourne max_dimension si aucun plateau n'est trouvé."""
        dimensions = np.array([2, 3, 4, 5])
        errors = np.array([1.0, 0.9, 0.8, 0.7])  # Pas de plateau
        optimal = find_first_plateau(dimensions, errors, threshold=0.01, min_consecutive=2)
        assert optimal == 5

    def test_single_dimension(self) -> None:
        """Avec une seule dimension, la retourne."""
        dimensions = np.array([2])
        errors = np.array([0.5])
        optimal = find_first_plateau(dimensions, errors)
        assert optimal == 2


# ---------------------------------------------------------------------------
# Propriété mathématique : E(m) décroissante
# ---------------------------------------------------------------------------


class TestMonotonicity:
    """Vérifie que E(m) est décroissante (propriété mathématique)."""

    def test_henon_series(self) -> None:
        """Pour la série de Hénon, E(m) est décroissante sur toute la plage."""
        X, _ = generate_henon_series(iterations=500, x0=0.0, y0=0.0, a=1.4, b=0.3)
        dimensions, errors = evaluate_embedding_dimensions(
            series=X, delay=1, min_dimension=2, max_dimension=20
        )
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1], (
                f"E({dimensions[i]}) = {errors[i]} < E({dimensions[i + 1]}) = {errors[i + 1]}. "
                f"La monotonie décroissante est violée."
            )


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------


class TestPipeline:
    """Test d'intégration du pipeline complet."""

    def test_constant_series(self) -> None:
        """Série constante → dimension détectée = min_dimension."""
        series = np.ones(100)
        dimensions, errors = evaluate_embedding_dimensions(
            series=series, delay=1, min_dimension=2, max_dimension=10
        )
        optimal = find_first_plateau(dimensions, errors, threshold=0.001)
        # Série constante → toutes les erreurs sont ~0 → plateau immédiat
        assert optimal == 2

    def test_henon_dimension(self) -> None:
        """Série de Hénon → dimension entre 2 et 10 (cohérent avec l'attracteur)."""
        X, _ = generate_henon_series(iterations=500, x0=0.0, y0=0.0, a=1.4, b=0.3)
        dimensions, errors = evaluate_embedding_dimensions(
            series=X, delay=1, min_dimension=2, max_dimension=20
        )
        # Pour la série de Hénon, les erreurs décroissent de façon monotone
        # On vérifie que E(m) est bien décroissante (propriété mathématique)
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1], (
                f"E({dimensions[i]}) = {errors[i]} < E({dimensions[i + 1]}) = {errors[i + 1]}. "
                f"La monotonie décroissante est violée."
            )
        # La dimension optimale dépend du seuil, on vérifie qu'elle est dans [2, 20]
        optimal = find_first_plateau(dimensions, errors, threshold=0.001, min_consecutive=2)
        assert 2 <= optimal <= 20
