# src/solver/main_pbs.py

import argparse
import time
import numpy as np
from src.algorithms.pbs import (
    OptimizedPBSAlgorithm,
    create_instance,
    read_graph_instance
)

def main():
    parser = argparse.ArgumentParser(
        description="Résolution du problème p-center avec l'algorithme PBS (Population-Based Search)"
    )
    parser.add_argument(
        "instance_file",
        help="Chemin vers le fichier d'instance (format Pullan)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Nombre maximum de générations (défaut: 20)"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Coût cible à atteindre (arrêt anticipé si atteint)"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Désactiver la parallélisation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode verbeux pour la recherche locale"
    )
    
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"PBS - Résolution du problème p-center")
    print(f"{'='*60}")
    print(f"Instance: {args.instance_file}")
    
    # Mesure du temps de lecture
    start_read = time.time()
    n, p, dist_list = read_graph_instance(args.instance_file)
    read_time = time.time() - start_read
    
    print(f"Nombre de sommets: {n}")
    print(f"Nombre de centres: {p}")
    print(f"Temps de lecture: {read_time:.2f}s")
    print(f"{'='*60}\n")

    # Construction des structures numpy
    start_prep = time.time()
    distances, neighbors = create_instance(n, p, dist_list)
    prep_time = time.time() - start_prep
    print(f"Temps de préparation des structures: {prep_time:.2f}s\n")

    # Configuration de l'algorithme
    print(f"Configuration:")
    print(f"- Générations max: {args.generations}")
    print(f"- Coût cible: {args.target if args.target else 'Aucun'}")
    print(f"- Parallélisation: {'Désactivée' if args.no_parallel else 'Activée'}")
    print(f"- Mode verbeux: {'Oui' if args.verbose else 'Non'}")
    print(f"{'='*60}\n")

    # Lancement de PBS
    start_algo = time.time()
    pbs = OptimizedPBSAlgorithm(
        n, p, distances, neighbors, 
        ls_verbose=args.verbose
    )
    
    sol, cost = pbs.run(
        target_cost=args.target, 
        max_generations=args.generations, 
        parallel=not args.no_parallel
    )
    
    algo_time = time.time() - start_algo

    # Affichage des résultats finaux
    print(f"\n{'='*60}")
    print(f"RÉSULTATS FINAUX")
    print(f"{'='*60}")
    print(f"Meilleur coût trouvé: {cost:.6f}")
    print(f"Solution (centres): {sorted(sol)}")
    print(f"\nTemps d'exécution:")
    print(f"- Lecture instance: {read_time:.2f}s")
    print(f"- Préparation: {prep_time:.2f}s")
    print(f"- Algorithme PBS: {algo_time:.2f}s")
    print(f"- TOTAL: {read_time + prep_time + algo_time:.2f}s")
    print(f"{'='*60}\n")

    # Vérification optionnelle de la solution
    if args.verbose:
        print("Vérification de la solution...")
        max_dist = 0
        for i in range(n):
            min_dist_to_center = min(distances[i, c] for c in sol)
            max_dist = max(max_dist, min_dist_to_center)
        
        print(f"Coût vérifié: {max_dist:.6f}")
        if abs(max_dist - cost) > 1e-6:
            print("⚠️  ATTENTION: Divergence entre le coût rapporté et vérifié!")
        else:
            print("✓ Coût vérifié correctement")

if __name__ == "__main__":
    main()