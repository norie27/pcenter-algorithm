import numpy as np
import random
from multiprocessing import Pool, cpu_count
from .genetic_operators import M1_global, M2_global, X1_global, X2_global
from .localsearch import VerboseOptimizedLocalSearchPCenter as OptimizedLocalSearchPCenter

# Variables globales
_global_distances = None
_global_neighbors = None

def init_worker(distances, neighbors, seed):
    """Initialise les variables globales dans chaque processus worker"""
    global _global_distances, _global_neighbors
    _global_distances = distances
    _global_neighbors = neighbors
    # Initialiser le générateur aléatoire avec un seed unique pour chaque processus
    random.seed(seed)
    
    
def process_pair_global(args):
    """Fonction globale pour traitement parallèle"""
    i, j, Pi, Pj, generation, n, p, ls_verbose = args
    
 
    # Utiliser les variables globales pour distances et neighbors
    global _global_distances, _global_neighbors
    distances = _global_distances
    neighbors = _global_neighbors
    
    results = []
    
    try:
        # L(M1(Pj))
        S = M1_global(Pj, n, p)
        ls = OptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=ls_verbose)
        sol, cost = ls.search(S, generation)
        results.append((sol, cost))
        
        # L(M2(X1(Pi,Pj)))
        S = X1_global(Pi, Pj, n, p)
        S = M2_global(S, distances)
        ls = OptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=ls_verbose)
        sol, cost = ls.search(S, generation)
        results.append((sol, cost))
        
        # X2 → M2 pour chaque enfant
        S1, S2 = X2_global(Pi, Pj, n, p, distances)
        for seed in (S1, S2):
            Snew = M2_global(seed, distances)
            ls = OptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=ls_verbose)
            sol, cost = ls.search(Snew, generation)
            results.append((sol, cost))
    
    except Exception as e:
        print(f"[PBS] Erreur dans process_pair_global: {type(e).__name__}: {e}")
        # Retourner des résultats vides en cas d'erreur
        return []
    
    return results