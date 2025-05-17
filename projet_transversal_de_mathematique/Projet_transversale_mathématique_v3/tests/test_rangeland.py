import unittest
from src.rangeland import DefaultSystem, DefaultParams, EquilibriumType

class TestRangelandSystem(unittest.TestCase):

    def setUp(self):
        self.system = DefaultSystem()

    def test_equilibria_at_low_E(self):
        E = 0.05
        eqs = self.system.find_equilibria(E)
        self.assertTrue(len(eqs) >= 3)  # Doit contenir au moins les trivials

    def test_equilibria_at_high_E(self):
        E = 0.9
        eqs = self.system.find_equilibria(E)
        self.assertTrue(len(eqs) >= 3)  # Toujours les trivials

    def test_classified_equilibria_type(self):
        E = 0.1
        classified = self.system.get_classified_equilibria(E)
        self.assertTrue(all(hasattr(eq, 'nature') for eq in classified))

    def test_saddle_detection(self):
        E = 0.01
        classified = self.system.get_classified_equilibria(E)
        has_saddle = any(eq.nature == EquilibriumType.SADDLE for eq in classified)
        self.assertTrue(has_saddle)

    def test_extinction_of_g(self):
        E_star = self.system.find_extinction_of_g((0.01, 0.5))
        self.assertIsInstance(E_star, float)
        self.assertGreater(E_star, 0)

    def test_extinction_of_w(self):
        E_star = self.system.find_extinction_of_w((0.01, 0.9))
        self.assertIsInstance(E_star, float)
        self.assertGreater(E_star, 0)

    if __name__ == '__main__':
        unittest.main()
