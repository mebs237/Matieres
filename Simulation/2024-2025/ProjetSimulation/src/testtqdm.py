import time
from tqdm import tqdm

print("\nDébut de la tâche avec tqdm...")
total_taches = 100
for i in tqdm(range(total_taches + 1), desc="Progression"):
    time.sleep(0.05) # Simule un travail en cours
print("Tâche avec tqdm terminée !")

print("\nDébut d'une autre tâche avec tqdm (itérations manuelles)...")
with tqdm(total=total_taches, desc="Traitement") as pbar:
    for i in range(total_taches):
        time.sleep(0.05)
        pbar.update(1) # Mettre à jour la barre manuellement
print("Deuxième tâche avec tqdm terminée !")