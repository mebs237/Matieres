from abc import ABC , abstractmethod
from hashlib import sha256
from numpy.typing import NDArray
from numpy import array
from numpy.random import random


class Generators(ABC):
    """
    Classe abstraite définissant ce qu'est un générateur.
    """

    @abstractmethod
    def next(self) -> float:
        """
        Génére le prochain nombre pseudo-aléatoire.
        """

    def generate(self,n:int)->NDArray:
        """
        Génère n nombres pseudo aléatoires

        Args
        ---
            n (int) : quantité (nombre) de nombres à générer
        """
        return array([self.next() for _ in range(n)])

class PythonGenerator(Generators):
    """
    Classe qui implement le générateur natif de python
    """
    def __str__(self):
        return "PythoGenerator"
    def next(self):
        return random()

class OurGenerator(Generators):
    """
    Classe qui implémente notre Générateur de loi uniforme entre [0,1] utilisant la séquence de nombre 'décimals'
    """
    def __str__(self)->str:
        return "OurGenerator"
    def __init__(self, decimals: NDArray, a: int = 1664525, c: int = 1013904223, m: int = 2**32 -1 ):
        """
        Initialisation du générateur.

        Args:
            decimals (NDArray): Séquence de chiffres pour l'initialisation.
            a (int): Multiplicateur du LCG.
            c (int): Incrément du LCG.
            m (int): Modulo du LCG.
        """
        self.a = a
        self.c = c
        self.m = m

        # Initialisation robuste avec hachage des décimales
        seed = sha256(decimals.tobytes()).digest()
        self.seed = int.from_bytes(seed[:8], 'big') % self.m  # Graine initiale sur 64 bits
        self.random_constant = int.from_bytes(seed[8:16], 'big') % self.m

    def next(self) -> float:
        """
        Génère le prochain nombre pseudo-aléatoire entre [0, 1[.
        """
        # Congruence linéaire (LCG)
        self.seed = (self.a * self.seed + self.c) % self.m

        # Opérations supplémentaires sur le seed
        self.seed ^= (self.seed >> 17)  # XOR avec une partie décalée du seed
        self.seed = (self.seed >> 16) | (self.seed << 16)  # Décalage circulaire de 16 bits
        self.seed = (self.seed + self.random_constant) % self.m  # Ajout d'une constante aléatoire

        # Post-traitement XORShift pour améliorer la qualité statistique
        x = self.seed
        x ^= (x << 21) & 0xFFFFFFFF  # Décalage à gauche de 21 bits
        x ^= (x >> 35)                # Décalage à droite de 35 bits
        x ^= (x << 14) & 0xFFFFFFFF   # Décalage à gauche de 14 bits

        # Normalisation entre [0, 1[
        return x / 0xFFFFFFFF
