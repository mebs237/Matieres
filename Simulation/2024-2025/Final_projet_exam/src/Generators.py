from abc import ABC , abstractmethod
from hashlib import sha256
from time import time_ns
from os import urandom
from numpy.typing import NDArray
from numpy import array
from numpy.random import random




class Generator(ABC):
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
            n (int) : quantité  de nombres à générer
        """
        return array([self.next() for _ in range(n)])

class PythonGenerator(Generator):
    """
    Classe qui implement le générateur natif de python
    """
    def __str__(self):
        return "PythoGenerator"

    def next(self):
        return random()

class OurGenerator(Generator):
    """
    Classe qui implémente notre Générateur de loi uniforme entre [0,1]
    """
    def __str__(self)->str:
        return "OurGenerator"

    def __init__(self, decimals: NDArray, a: int = 1664525, c: int = 1013904223, m: int = 2**32 -1):
        """
        Initialisation du générateur avec une graine unique à chaque instanciation.

        Args:
            decimals (NDArray): Séquence de chiffres pour l'initialisation.
            a (int): Multiplicateur du LCG.
            c (int): Incrément du LCG.
            m (int): Modulo du LCG (nombre premier de Mersenne).
        """
        self.a = a
        self.c = c
        self.m = m

        # Création d'une graine unique combinant :
        # 1. Les decimals fournis
        # 2. Le timestamp actuel (nanosecondes)
        # 3. Une valeur aléatoire du système

        # Combine les trois sources d'aléatoire
        unique_seed = decimals.tobytes() + time_ns().to_bytes(8, 'big') + urandom(8)

        # Hachage final pour obtenir une graine uniforme
        seed_hash = sha256(unique_seed).digest()
        self.seed = int.from_bytes(seed_hash[:8], 'big') % self.m
        self.mask = int.from_bytes(seed_hash[8:16], 'big') % self.m

    def next(self) -> float:
        """
        Génère le prochain nombre pseudo-aléatoire entre [0, 1[.
        Combine LCG avec des opérations de bits pour améliorer la qualité.
        """
        # LCG de base
        self.seed = (self.a * self.seed + self.c) % self.m

        # Opérations de bits essentielles
        x = self.seed
        x ^= (x >> 16)  # XOR avec décalage à droite
        x = (x + self.mask) % self.m  # Ajout d'une constante masquée

        # Normalisation
        return x / self.m
