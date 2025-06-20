# src/solver/main_pbs.py

import argparse
import time
from src.utils.read_instance import read_instance_auto
from src.utils.instance import Instance
#from src.algorithms.pbs import PBS

# src/solver/main_pbs.py

import argparse
import numpy as np
from src.algorithms.pbs import (
    OptimizedPBSAlgorithm,
    create_instance,
    read_graph_instance
)

def main():
    parser = argparse.ArgumentParser(
        description="Résolution du PBS (algorithme hybride PBS)"
    )
    parser.add_argument(
        "instance_file",
        help="Chemin vers le fichier d’instance Pullan"
    )
    args = parser.parse_args()

    # Lecture de l'instance
    n, p, dist_list = read_graph_instance(args.instance_file)

    # Construction des structures numpy
    distances, neighbors = create_instance(n, p, dist_list)

    # Lancement de PBS
    pbs = OptimizedPBSAlgorithm(n, p, distances, neighbors)
    sol, cost = pbs.run(target_cost=None, max_generations=20, parallel=True)

    # Affichage du résultat
    print("Solution:", sol)
    print("Coût   :", cost)

if __name__ == "__main__":
    main()
