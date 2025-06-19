# src/main.py

# src/solver/main_local.py

import argparse
import numpy as np
from src.utils.read_instance import read_instance_auto
from src.utils.instance import Instance
from src.algorithms.localsearch import VerboseOptimizedLocalSearchPCenter

def main():
    parser = argparse.ArgumentParser(
        description="Résolution du problème P-Center par recherche locale optimisée (debug)"
    )
    parser.add_argument(
        "instance_file",
        help="Chemin vers le fichier d’instance (format matrice ou graphe)"
    )
    args = parser.parse_args()

    # Lecture de l'instance
    n, p, dist = read_instance_auto(args.instance_file)
    print(f"[INFO] Instance chargée : n={n}, p={p}")

    # Construction des structures numpy
    distances = np.array(dist, dtype=np.float64)
    neighbors = np.argsort(distances, axis=1)

    # Initialisation en mode verbose pour debug
    solver = VerboseOptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=True)
    print("[INFO] Lancement de la recherche locale optimisée en mode verbose\n")
    solution, cost = solver.search()

    # Affichage du résultat
    print("\n=== Résultat Debug ===")
    print(f"Centres choisis : {sorted(solution)}")
    print(f"Coût maximum    : {cost:.2f}")

if __name__ == "__main__":
    main()
