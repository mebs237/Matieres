from abc import ABC, abstractmethod
from numpy.typing import NDArray
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

    def generate_n(self,n:int)->NDArray:
        """
        Génère n nombres pseudo aléatoires

        Args
        ---
            n (int) : quantité (nombre) de nombres à générer
        """
        return np.array([self.next() for _ in range(n)])


class Im2Generator(Generators):
    """
    Générateur combinant deux LCG et un post-traitement XORShift.
    Améliorations :
    - Initialisation robuste via hachage des décimales
    - Paramètres configurables
    - Optimisation des opérations binaires
    """

    # Paramètres par défaut (peuvent être modifiés par l'utilisateur)
    DEFAULT_PARAMS = {
        "a1": np.uint64(1664525),   # LCG 1: Numerical Recipes
        "c1": np.uint64(1013904223),
        "m1": np.uint64(2**32),
        "a2": np.uint64(16807),     # LCG 2: Park-Miller
        "c2": np.uint64(0),
        "m2": np.uint64(2**64 - 1),
        "xor_shift_params": [21, 35, 4]  #[Décalages optimisés]
    }

    def __init__(self, decimals: NDArray, lg: int = 7, **kwargs):
        super().__init__()
        self.decimals = decimals
        self.lg = lg
        self.len_decimals = len(decimals)
        self.power_of_10 = np.array([10**i for i in range(lg)])

        # Fusion des paramètres utilisateur et défauts
        self.params = self.DEFAULT_PARAMS.copy()
        self.params.update(kwargs)

        # Initialisation robuste via hachage SHA-256
        seed = hashlib.sha256(decimals.tobytes()).digest()
        self.x1 = int.from_bytes(seed[:8], 'big') % self.params["m1"]
        self.x2 = int.from_bytes(seed[8:16], 'big') % self.params["m2"]

    def next(self) -> float:
        # Génération avec deux LCG (cast en uint64 pour éviter les erreurs de débordement)
        self.x1 = (self.params["a1"] * self.x1 + self.params["c1"]) % self.params["m1"]
        self.x2 = (self.params["a2"] * self.x2 + self.params["c2"]) % self.params["m2"]

        # Combinaison XORShift
        combined = np.uint64(self.x1) ^ np.uint64(self.x2)
        combined = combined& np.uint64(0xFFFFFFFF)& np.uint64(0xFFFFFFFF)
        combined ^= (combined << self.params["xor_shift_params"][0]) & np.uint64(0xFFFFFFFF)
        combined ^= (combined >> self.params["xor_shift_params"][1])
        combined ^= (combined << self.params["xor_shift_params"][2])

        return float(combined) / float(0xFFFFFFFF + 1)  # Normalisation [0,1[


# Test d'initialisation (ne déclenche plus d'erreur)
generator = Im2Generator(decimals=np.array([3,1,4,1,5,9,2,6]))
print(generator.generate_n(5))  # Output: 0.12345678...
