import pytest
import numpy as np
from ressort import matrice, solution, _handle_unique_solution, _handle_multiple_solutions, _handle_vector_space_solution

@pytest.mark.parametrize("x,expected_inj", [
    (np.array([1, 2, 3]), True),
    (np.array([0, 0, 0]), False),
    (np.array([1, 1, 1]), True),
    (np.array([-1, 0, 1]), True)
])
def test_matrice_injectivite(x, expected_inj):
    """Test de l'injectivité de la matrice avec différents cas"""
    result = matrice(x)
    assert result['inj'] == expected_inj
    assert result['A'] is not None
    if expected_inj:
        assert result['l'] is not None
    else:
        assert result['l'] is None

@pytest.mark.parametrize("data,y,expected_type,expected_shape", [
    (np.array([[1, 2], [2, 4], [3, 6]]), np.array([1, 2, 3]), np.ndarray, (2,)),
    (np.array([[0, 0], [0, 0], [0, 0]]), np.array([0, 0, 0]), callable, None),
    (np.array([[1, 2], [2, 4], [0, 0]]), np.array([1, 2, 0]), callable, None)
])
def test_solution_types(data, y, expected_type, expected_shape):
    """Test des différents types de solutions"""
    sol = solution(data, y)
    assert isinstance(sol, expected_type)
    if expected_shape is not None:
        assert sol.shape == expected_shape

def test_solution_multiple_values():
    """Test des valeurs retournées pour le cas de solutions multiples"""
    data = np.array([[0, 0], [0, 0], [0, 0]])
    y = np.array([0, 0, 0])
    sol = solution(data, y)
    assert callable(sol)
    result = sol(1, 2)
    assert np.array_equal(result, np.array([1, 2]))

def test_solution_span_values():
    """Test des valeurs retournées pour le cas d'un espace vectoriel"""
    data = np.array([[1, 2], [2, 4], [0, 0]])
    y = np.array([1, 2, 0])
    sol = solution(data, y)
    assert callable(sol)
    t = 2
    result = sol(t)
    assert result.shape == (2,)

@pytest.mark.parametrize("data,y,expected_error", [
    (np.array([[1, 2], [2, 4], [3, 6]]), np.array([1, 2]), ValueError),  # Dimensions incompatibles
    (np.array([]), np.array([]), ValueError),  # Tableau vide
    (np.array([[1, 2], [2, 4], [3, 6]]), None, TypeError)  # y est None
])
def test_solution_edge_cases(data, y, expected_error):
    """Test des cas limites pour la fonction solution"""
    with pytest.raises(expected_error):
        solution(data, y)

# Tests pour les nouvelles fonctions handle
def test_handle_unique_solution():
    """Test de la fonction _handle_unique_solution"""
    A = np.array([[1, 2], [2, 4], [3, 6]])
    y = np.array([1, 2, 3])
    result = _handle_unique_solution(A, y, contr=True, fcont=lambda u: u[0]>0 and u[1]>=0, info=True)
    assert isinstance(result, dict)
    assert 'u0' in result
    assert 'gen' in result
    assert result['unique'] is True
    assert result['case'] == 0

def test_handle_multiple_solutions():
    """Test de la fonction _handle_multiple_solutions"""
    result = _handle_multiple_solutions(contr=True, fcont=lambda u: u[0]>0 and u[1]>=0, info=True)
    assert isinstance(result, dict)
    assert 'u0' in result
    assert 'gen' in result
    assert result['unique'] is False
    assert result['case'] == 2

def test_handle_vector_space_solution():
    """Test de la fonction _handle_vector_space_solution"""
    X = np.array([1, 2, 3])
    f = np.array([1, 2, 3])
    l = 1.0
    result = _handle_vector_space_solution(X, f, l, contr=True, fcont=lambda u: u[0]>0 and u[1]>=0, info=True)
    assert isinstance(result, dict)
    assert 'u0' in result
    assert 'gen' in result
    assert result['unique'] is False
    assert result['case'] == 1

@pytest.mark.parametrize("contr,fcont,should_raise", [
    (True, lambda u: u[0]>0 and u[1]>=0, False),
    (True, lambda u: u[0]<0, True),  # Contrainte impossible à satisfaire
    (False, lambda u: False, False)  # Contraintes désactivées
])
def test_solution_constraints(data, y, contr, fcont, should_raise):
    """Test des contraintes sur les solutions"""
    if should_raise:
        with pytest.raises(ValueError):
            solution(data, y, contr=contr, fcont=fcont)
    else:
        sol = solution(data, y, contr=contr, fcont=fcont)
        assert sol is not None