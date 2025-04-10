import unittest
import numpy as np
from ressort import matrice, solution, _handle_unique_solution, _handle_multiple_solutions, _handle_vector_space_solution

class TestRessort(unittest.TestCase):

    def test_matrice_injective(self):
        """
        Cas où la matrice est injective
        """
        x = np.array([1, 2, 3])
        result = matrice(x)
        self.assertTrue(result['inj'])
        self.assertIsNotNone(result['A'])
        self.assertIsNotNone(result['l'])

    def test_matrice_non_injective(self):
        """Cas où la matrice n'est pas injective"""
        x = np.array([0, 0, 0])
        result = matrice(x)
        self.assertFalse(result['inj'])
        self.assertIsNotNone(result['A'])
        self.assertIsNone(result['l'])

    def test_solution_unique(self):
        """ Cas où la solution est unique """
        data = np.array([[1, 2], [2, 4], [3, 6]])
        y = np.array([1, 2, 3])
        sol = solution(data, y)
        self.assertIsInstance(sol, np.ndarray)
        self.assertEqual(sol.shape, (2,))

    def test_solution_multiple(self):
        """ Cas où il y a plusieurs solutions """
        data = np.array([[0, 0], [0, 0], [0, 0]])
        y = np.array([0, 0, 0])
        sol = solution(data, y)
        self.assertTrue(callable(sol))
        result = sol(1, 2)
        self.assertEqual(result.tolist(), [1, 2])

    def test_solution_span(self):
        """ Cas où les solutions sont dans un espace vectoriel """
        data = np.array([[1, 2], [2, 4], [0, 0]])
        y = np.array([1, 2, 0])
        sol = solution(data, y)
        self.assertTrue(callable(sol))
        t = 2
        result = sol(t)
        self.assertEqual(result.shape, (2,))

    def test_handle_unique_solution(self):
        """Test de la fonction _handle_unique_solution"""
        A = np.array([[1, 2], [2, 4], [3, 6]])
        y = np.array([1, 2, 3])
        result = _handle_unique_solution(A, y, contr=True, fcont=lambda u: u[0]>0 and u[1]>=0, info=True)
        self.assertIsInstance(result, dict)
        self.assertIn('u0', result)
        self.assertIn('gen', result)
        self.assertTrue(result['unique'])
        self.assertEqual(result['case'], 0)

    def test_handle_multiple_solutions(self):
        """Test de la fonction _handle_multiple_solutions"""
        result = _handle_multiple_solutions(contr=True, fcont=lambda u: u[0]>0 and u[1]>=0, info=True)
        self.assertIsInstance(result, dict)
        self.assertIn('u0', result)
        self.assertIn('gen', result)
        self.assertFalse(result['unique'])
        self.assertEqual(result['case'], 2)

    def test_handle_vector_space_solution(self):
        """Test de la fonction _handle_vector_space_solution"""
        X = np.array([1, 2, 3])
        f = np.array([1, 2, 3])
        l = 1.0
        result = _handle_vector_space_solution(X, f, l, contr=True, fcont=lambda u: u[0]>0 and u[1]>=0, info=True)
        self.assertIsInstance(result, dict)
        self.assertIn('u0', result)
        self.assertIn('gen', result)
        self.assertFalse(result['unique'])
        self.assertEqual(result['case'], 1)

    def test_solution_constraints(self):
        """Test des contraintes sur les solutions"""
        data = np.array([[1, 2], [2, 4], [3, 6]])
        y = np.array([1, 2, 3])

        # Test avec contraintes valides
        sol = solution(data, y, contr=True, fcont=lambda u: u[0]>0 and u[1]>=0)
        self.assertIsNotNone(sol)

        # Test avec contraintes impossibles
        with self.assertRaises(ValueError):
            solution(data, y, contr=True, fcont=lambda u: u[0]<0)

        # Test sans contraintes
        sol = solution(data, y, contr=False, fcont=lambda u: False)
        self.assertIsNotNone(sol)

if __name__ == '__main__':
    unittest.main()