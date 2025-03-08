from abc import ABC, abstractmethod
import numpy as np
import hashlib


class Generators(ABC):
    """
    Classe abstraite définissant ce qu'est un générateur.
    """

    @abstractmethod
    def next(self) -> float:
        """
        Génére le prochain nombre pseudo-aléatoire.
        """

    def generate_n(self,n:int)->np.ndarray:
        """
        Génère n nombres pseudo aléatoires

        Args
        ---
            n (int) : quantité (nombre) de nombres à générer
        """
        return np.array([self.next() for _ in range(n)])

class Im2Generator(Generators):
    """
    Générateur robuste combinant deux LCG et un post-traitement XORShift.
    Améliorations incluses :
    - Initialisation forte via hachage des décimales
    - Paramètres personnalisables
    - Optimisation des opérations binaires
    """

    # Paramètres par défaut (peuvent être personnalisés)
    DEFAULT_A1, DEFAULT_C1, DEFAULT_M1 = np.int64(1664525), np.int64(1013904223), np.int64(2**32)
    DEFAULT_A2, DEFAULT_C2, DEFAULT_M2 = np.int64(16807), np.int64(0), np.int64(2**64 - 1)

    def __init__(self, decimals: np.ndarray, lg: int = 10,
                 a1: int = DEFAULT_A1, c1: int = DEFAULT_C1, m1: int = DEFAULT_M1,
                 a2: int = DEFAULT_A2, c2: int = DEFAULT_C2, m2: int = DEFAULT_M2):
        """
        Args:
            decimals (np.ndarray): Séquence de chiffres pour l'initialisation
            lg (int): Taille du bloc de décimales utilisé pour le hachage (défaut=10)
            a1, c1, m1: Paramètres du premier LCG
            a2, c2, m2: Paramètres du deuxième LCG
        """
        self.decimals = decimals
        self.lg = lg
        self.len_decimals = len(decimals)

        # 1. Initialisation robuste via hachage des décimales
        self.index = self._secure_index_initialization(decimals, lg)
        self.a1, self.c1, self.m1 = a1, c1, m1
        self.a2, self.c2, self.m2 = a2, c2, m2

        # 2. États initiaux des LCG calculés via hachage SHA-256
        self.x1 = self._compute_initial_state(decimals, self.index, lg, "x1")
        self.x2 = self._compute_initial_state(decimals, self.index, lg, "x2")

    def _secure_index_initialization(self, decimals, lg):
        """ Génère un index de départ via hachage des premières décimales """
        seed = "".join(map(str, decimals[:lg])).encode()
        return int(hashlib.sha256(seed).hexdigest(), 16) % (len(decimals) - lg)

    def _compute_initial_state(self, decimals, index, lg, salt):
        """ Calcule un état initial unique via combinaison de décimales et hachage """
        chunk = decimals[index:index + lg]
        seed = "".join(map(str, chunk + [salt])).encode()
        return int(hashlib.sha256(seed).hexdigest(), 16) % self.m1

    def next(self) -> float:
        # 1. Génération avec deux LCG
        self.x1 = (self.a1 * self.x1 + self.c1) % self.m1
        self.x2 = (self.a2 * self.x2 + self.c2) % self.m2

        # 2. Combinaison XOR et post-traitement optimisé
        combined = (self.x1 ^ self.x2) & 0xFFFFFFFF
        combined ^= (combined << 17) & 0xFFFFFFFF  # Optimisation du décalage
        combined ^= (combined >> 15)

        # 3. Mise à jour périodique de l'index (tous les 1000 appels)
        if combined % 1000 == 0:
            self.index = self._secure_index_initialization(self.decimals, self.lg)

        return combined / 0xFFFFFFFF  # Normalisation entre [0, 1[
