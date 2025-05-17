import pytest
import numpy as np
from models.Equilibres import (
    equation, Pg, delta_e, dubble_u, equilibres,
    Jac, class_equilibrium, ulim, cvg, phi,
    intercept_g, separatrix_points, plot_separatrix
)

# Paramètres de test
E_TEST_VALUES = [0.0, 0.1, 0.3, 0.4 , 0.5 , 0.6 , 0.7 , 0.8]
V_TEST_VALUES = [
    np.array([0.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([1.0, 0.0]),
    np.array([0.2, 0.8])
]
TOL = 1e-6

@pytest.mark.parametrize("v,E", [
    (v, E) for v in V_TEST_VALUES for E in E_TEST_VALUES
])
def test_equation(v, E):
    """Test de la fonction equation."""
    result = equation(0, v, E)
    assert len(result) == 2
    assert isinstance(result, np.ndarray)

@pytest.mark.parametrize("x,E,expected", [
    (0, 0, 0.4),  # Cas spécial
    (0.5, 0.1, None),  # Cas général
    (1.0, 0.3, None),  # Cas limite
])
def test_Pg(x, E, expected):
    """Test de la fonction Pg."""
    result = Pg(x, E)
    if expected is not None:
        assert result == expected
    else:
        assert isinstance(result, float)
        assert 0 <= result <= 1

@pytest.mark.parametrize("E", E_TEST_VALUES)
def test_delta_e(E):
    """Test de la fonction delta_e."""
    result = delta_e(E)
    assert isinstance(result, float)

@pytest.mark.parametrize("g,E,expected", [
    (0, 0, 1),  # Cas spécial
    (0.5, 0.1, None),  # Cas général
    (1.0, 0.3, None),  # Cas limite
])
def test_dubble_u(g, E, expected):
    """Test de la fonction dubble_u."""
    result = dubble_u(g, E)
    if expected is not None:
        assert result == expected
    else:
        assert isinstance(result, float)
        assert 0 <= result <= 1

@pytest.mark.parametrize("E", E_TEST_VALUES)
def test_equilibres(E):
    """Test de la fonction equilibres."""
    result = equilibres(E)
    assert isinstance(result, list)
    for eq in result:
        assert isinstance(eq,tuple)
        assert len(eq) == 2
        assert all(0 <= x <= 1 for x in eq)

@pytest.mark.parametrize("v,E", [
    (v, E) for v in V_TEST_VALUES for E in E_TEST_VALUES
])
def test_Jac(v, E):
    """Test de la fonction Jac."""
    result = Jac(v, E)
    assert result.shape == (2, 2)
    assert isinstance(result, np.ndarray)

@pytest.mark.parametrize("E", E_TEST_VALUES)
def test_class_equilibrium(E):
    """Test de la fonction class_equilibrium."""
    result = class_equilibrium(E)
    assert isinstance(result, dict)
    for eq, cls in result.items():
        assert cls in ['stable', 'instable', 'selle']

@pytest.mark.parametrize("u0,E", [
    (v, E) for v in V_TEST_VALUES for E in E_TEST_VALUES
])
def test_ulim(u0, E):
    """Test de la fonction ulim."""
    result = ulim(u0, equation, 100, E)
    assert len(result) == 2
    assert isinstance(result, np.ndarray)
    assert all(0 <= x <= 1 for x in result)

@pytest.mark.parametrize("u0,u_star,E", [
    (np.array([0.5, 0.5]), np.array([0.7, 0.3]), E) for E in E_TEST_VALUES
])
def test_cvg(u0, u_star, E):
    """Test de la fonction cvg."""
    converge = cvg(u0, u_star, equation,t_max= 100, tol=TOL)
    assert isinstance(converge, bool)

@pytest.mark.parametrize("u0,equi1,equi2,E", [
    (np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.7, 0.3]), E) for E in E_TEST_VALUES
])
def test_phi(u0, equi1, equi2, E):
    """Test de la fonction phi."""
    result = phi(u0, equi1, equi2, equation, 100, E)
    assert isinstance(result, float)

@pytest.mark.parametrize("w,E", [
    (w, E) for w in [0.2, 0.5, 0.8] for E in E_TEST_VALUES
])
def test_intercept_g(w, E):
    """Test de la fonction intercept_g."""
    result = intercept_g(w, E)
    if result is not None:
        assert isinstance(result, float)
        assert 0 <= result <= 1

@pytest.mark.parametrize("E", E_TEST_VALUES)
def test_separatrix_points(E):
    """Test de la fonction separatrix_points."""
    w, g = separatrix_points(E)
    assert isinstance(w, np.ndarray)
    assert isinstance(g, np.ndarray)
    assert len(w) == len(g)
    if len(w) > 0:
        assert all(0 <= x <= 1 for x in w)
        assert all(0 <= x <= 1 for x in g)


@pytest.mark.parametrize("E", E_TEST_VALUES)
def test_plot_separatrix(E):
    #Test de la fonction plot_separatrix
    # Cette fonction ne retourne rien, donc on teste simplement qu'elle s'exécute sans erreur
    try:
        plot_separatrix(E)
        # Si on arrive ici, c'est que la fonction s'est exécutée sans erreur
        assert True
    except Exception as e:
        pytest.fail(f"plot_separatrix a levé une exception: {e}")


if __name__=='__main__':

    pytest.main([__file__])