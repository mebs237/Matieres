import pytest
import numpy as np
from ressort import matrice, solution

@pytest.mark.parametrize("x,expected_shape", [
    (np.array([1, 2, 3]), (3, 2)),  # Matrice A avec 3 lignes et 2 colonnes
    (np.array([0, 0, 0]), (3, 2)),  # Même si x est nul, A doit avoir la bonne forme
    (np.array([1]), (1, 2))         # Cas avec un seul élément
])
def test_matrice_shape(x, expected_shape):
    """Test de la forme de la matrice A générée"""
    result = matrice(x)
    assert result['A'].shape == expected_shape

@pytest.mark.parametrize("x,expected_l", [
    (np.array([1, 0, 0]), 1),  # l doit être égal au premier x non nul
    (np.array([0, 2, 0]), 4),  # l doit être égal au premier x non nul
    (np.array([0, 0, 0]), None)  # Aucun x non nul, l doit être None
])
def test_matrice_l_value(x, expected_l):
    """Test de la valeur de l dans le dictionnaire retourné par matrice"""
    result = matrice(x)
    assert result['l'] == expected_l

def test_solution_unique_entry_shape():
    """Test du cas unique (ker(A) = {0})"""
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    result = solution(x, y, info=True)
    assert result["case"] == "unique"
    assert result["entry_shape"] is None
    assert np.allclose(result["u0"], np.array([2, 0]))  # Vérifie la solution particulière

def test_solution_multiple_entry_shape():
    """Test du cas multiple (ker(A) = R²)"""
    x = np.array([0, 0, 0])
    y = np.array([0, 0, 0])
    result = solution(x, y, info=True)
    assert result["case"] == "all_space"
    assert result["entry_shape"] == (2,)
    t = np.array([1, 2])
    generated = result["gen"](t)
    assert np.array_equal(generated, np.array([1, 2]))

def test_solution_vector_space_entry_shape():
    """Test du cas espace vectoriel (ker(A) = span((-λ, 1)))"""
    x = np.array([1, 0, -1, 0, 1])
    y = np.array([2, -2, 0, 8, 11])
    result = solution(x, y, info=True)
    assert result["case"] == "vector_space"
    assert result["entry_shape"] == ()
    t = 2
    generated = result["gen"](t)
    assert generated.shape == (2,)
    assert np.allclose(generated, result["u0"] + t * np.array([-1, 1]))

@pytest.mark.parametrize("x,y,fcont,expected_error", [
    (np.array([1, 2, 3]), np.array([2, 4, 6]), lambda u: u[0] < 0, ValueError),  # Contrainte non respectée
    (np.array([1, 2, 3]), np.array([2, 4, 6]), lambda u: u[1] < 0, ValueError)   # Contrainte non respectée
])
def test_solution_constraints_exceptions(x, y, fcont, expected_error):
    """Test des exceptions levées pour des contraintes non respectées """
    with pytest.raises(expected_error):
        solution(x, y, contr=True, fcont=fcont, info=True)

def test_solution_no_constraints():
    """Test sans contraintes (contr=False)"""
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    result = solution(x, y, contr=False, info=True)
    assert result["case"] == "unique"
    assert result["entry_shape"] is None
    assert np.allclose(result["u0"], np.array([2, 0]))  # Vérifie la solution particulière

@pytest.mark.parametrize("x,y,expected_error", [
    (np.array([1, 2, 3]), np.array([1, 2]), ValueError),  # Dimensions incompatibles
    (np.array([5]), np.array([]), ValueError),  # Tableaux vides
    (np.array([1, 2, 3]), None, AttributeError)  # y est None
])
def test_solution_invalid_inputs(x, y, expected_error):
    """Test des cas limites pour les entrées invalides"""
    with pytest.raises(expected_error):
        solution(x, y)

def test_solution_generator_invalid_t():
    """Test des exceptions levées par le générateur pour des t invalides"""
    x = np.array([0, 0, 0])
    y = np.array([0, 0, 0])
    result = solution(x, y, info=True)
    with pytest.raises(ValueError , match= "t doit être une liste de deux éléments"):
        result["gen"](2)  # t qui viole les contraintes

if __name__ == '__main__':
    pytest.main([__file__])