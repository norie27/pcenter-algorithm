# src/main.py

# src/solver/main_local.py

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
        help="Chemin vers le fichier d'instance Pullan"
    )
    args = parser.parse_args()

    # Lecture de l'instance
    n, p, dist_list = read_graph_instance(args.instance_file)
    distances, neighbors = create_instance(n, p, dist_list)

    # Multi-restart avec seed variable
    NUM_RESTARTS = 10
    best_cost = float('inf')
    best_solution = None
    
    print(f"=== MULTI-RESTART PBS: {NUM_RESTARTS} exécutions ===\n")
    
    for i in range(NUM_RESTARTS):
        print(f"--- Exécution {i+1}/{NUM_RESTARTS} ---")
        
        # SEED DIFFÉRENT À CHAQUE FOIS
        import random
        random.seed(i * 42)
        np.random.seed(i * 42)
        
        # Nouvelle exécution de PBS
        pbs = OptimizedPBSAlgorithm(n, p, distances, neighbors)
        sol, cost = pbs.run(target_cost=74, max_generations=20, parallel=True)
        
        print(f"Résultat: {cost:.4f}")
        
        # Garder le meilleur
        if cost < best_cost:
            best_cost = cost
            best_solution = sol
            print(f"*** Nouvelle meilleure solution! ***")
        
        print()
    
    # Affichage du résultat final
    print(f"=== MEILLEUR RÉSULTAT SUR {NUM_RESTARTS} EXÉCUTIONS ===")
    print("Solution:", best_solution)
    print("Coût   :", best_cost)

if __name__ == "__main__":
    main()