import unittest
import numpy as np
from ressort import matrice, solution

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
        sol = solution(data)
        self.assertIsInstance(sol, np.ndarray)

    def test_solution_multiple(self):
        """ Cas où il y a plusieurs solutions """
        data = np.array([[0, 0], [0, 0], [0, 0]])
        sol = solution(data)
        self.assertTrue(callable(sol))
        self.assertEqual(sol(1, 2).tolist(), [1, 2])

    def test_solution_span(self):
        """ Cas où les solutions sont dans un espace vectoriel """
        data = np.array([[1, 2], [2, 4], [0, 0]])
        sol = solution(data)
        self.assertTrue(callable(sol))
        t = 2
        result = sol(t)
        self.assertEqual(result.shape, (2,))

if __name__ == '__main__':
    unittest.main()