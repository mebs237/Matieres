import time
import sys

def barre_de_progression_simple(iteration, total, prefixe='', suffixe='', longueur=50, remplissage='█', vide='-'):
    """
    Affiche une barre de progression.

    Args:
        iteration (int): L'itération actuelle.
        total (int): Le nombre total d'itérations.
        prefixe (str): Préfixe à afficher avant la barre (ex: 'Progrès: ').
        suffixe (str): Suffixe à afficher après la barre (ex: 'Complet').
        longueur (int): Longueur de la barre en caractères.
        remplissage (str): Caractère utilisé pour le remplissage de la barre.
        vide (str): Caractère utilisé pour la partie vide de la barre.
    """
    pourcentage = 100 * (iteration / float(total))
    rempli = int(longueur * iteration // total)
    barre = remplissage * rempli + vide * (longueur - rempli)
    sys.stdout.write(f'\r{prefixe}[{barre}] {pourcentage:.1f}% {suffixe}')
    sys.stdout.flush() # Assure que la sortie est affichée immédiatement

# Exemple d'utilisation :
print("Début de la tâche simple...")
total_taches = 100
for i in range(total_taches + 1):
    barre_de_progression_simple(i, total_taches, prefixe='Progression :', suffixe='Terminé', longueur=40)
    time.sleep(0.05) # Simule un travail en cours
print("\nTâche simple terminée !")