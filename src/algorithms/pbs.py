import numpy as np
import random
import time
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count
import math
import os

# src/algorithms/pbs.py
from src.algorithms.localsearch import VerboseOptimizedLocalSearchPCenter as OptimizedLocalSearchPCenter
# Variables globales pour la mémoire partagée (initialisées dans run_pbs)
_global_distances = None
_global_neighbors = None

def init_worker(distances, neighbors, seed):
    """Initialise les variables globales dans chaque processus worker"""
    global _global_distances, _global_neighbors
    _global_distances = distances
    _global_neighbors = neighbors
    # Initialiser le générateur aléatoire avec un seed unique pour chaque processus
    random.seed(seed)

def M1_global(Pj, n, p):
    """Version globale de M1 pour la parallélisation"""
    q = random.randint(p // 2, p)
    selected = random.sample(Pj, min(q, len(Pj)))
    available = [v for v in range(n) if v not in selected]
    if len(selected) < p and available:
        needed = p - len(selected)
        selected.extend(random.sample(available, min(needed, len(available))))
    return selected

def M2_global(Pi, distances):
    """Version globale de M2 pour la parallélisation"""
    if len(Pi) < 2:
        return Pi.copy()
    
    min_dist = float('inf')
    ri, rj = 0, 1
    
    for i in range(len(Pi)):
        for j in range(i + 1, len(Pi)):
            d = distances[Pi[i], Pi[j]]
            if d < min_dist:
                min_dist = d
                ri, rj = i, j
    
    return [c for k, c in enumerate(Pi) if k not in (ri, rj)]

def X1_global(Pi, Pj, n, p):
    """Version globale de X1 pour la parallélisation"""
    combined = list(set(Pi) | set(Pj))
    
    if len(combined) >= p:
        return random.sample(combined, p)
    else:
        available = [v for v in range(n) if v not in combined]
        needed = p - len(combined)
        if needed > 0 and available:
            additional = random.sample(available, min(needed, len(available)))
            combined.extend(additional)
        return combined

def X2_global(Pi, Pj, n, p, distances):
    """Version globale de X2 pour la parallélisation - VERSION CORRIGÉE"""
    u1, u2 = random.sample(range(n), 2)
    q = random.uniform(0.1, 0.9)
    
    S1, S2 = [], []
    
    # Traiter Pi séparément
    for f in Pi:
        d1 = distances[f, u1]
        d2 = distances[f, u2]
        
        if d2 <= 1e-12:
            S1.append(f)
            continue
            
        ratio = d1 / d2
        if ratio <= q:
            S1.append(f)
        else:
            S2.append(f)
    
    # Traiter Pj séparément
    for f in Pj:
        d1 = distances[f, u1]
        d2 = distances[f, u2]
        
        if d2 <= 1e-12:
            S2.append(f)
            continue
            
        ratio = d1 / d2
        if ratio > q:
            S1.append(f)
        else:
            S2.append(f)
    
    # Enlever les doublons potentiels
    S1 = list(set(S1))
    S2 = list(set(S2))
    
    if len(S1) > p:
        S1 = random.sample(S1, p)
    if len(S2) > p:
        S2 = random.sample(S2, p)
        
    return S1, S2

def process_pair_global(args):
    """Fonction globale pour traitement parallèle"""
    i, j, Pi, Pj, generation, n, p, ls_verbose = args
    
    # Import local dans la fonction pour éviter les problèmes de sérialisation
 
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

# Classe PBS
def pbs_init_worker(args):
    n, p, distances, neighbors, verbose, worker_id = args
    import random, time, os
    seed = int(time.time() * 1e6) + os.getpid() + worker_id
    seed = seed % (2**32 - 1)  # <--- On force la seed à la bonne plage
    random.seed(seed)
    np.random.seed(seed)

    S0 = [random.randint(0, n-1)]
    ls = OptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=verbose)
    return ls.search(S0, 0)

class OptimizedPBSAlgorithm:
    """PBS EXACT selon Pullan avec optimisations numpy et parallélisation"""

    def __init__(
        self,
        n: int,
        p: int,
        distances: np.ndarray,
        neighbors: np.ndarray,
        ls_verbose: bool = False,
    ):
        self.n = n
        self.p = p
        self.distances = distances
        self.neighbors = neighbors
        self.ls_verbose = ls_verbose

        self.population_size = 8
        self.P: List[Tuple[List[int], float]] = []
        self.generation = 0

    def create_initial_solution(self) -> List[int]:
        """C - Crée une solution avec un sommet aléatoire"""
        return [random.randint(0, self.n - 1)]

    def is_too_close(self, S: List[int], cost: float) -> bool:
        """Vérifie la diversité - selon Pullan: même coût ou facilités trop similaires"""
        for Pi, Pi_cost in self.P:
            # Comparaison de coût plus stricte
            if abs(cost - Pi_cost) < 1e-12:
                return True
            # Seuil de similarité ajusté à 70% (moins strict que 90%)
            if len(set(S) & set(Pi)) > 0.7 * self.p:
                return True
        return False

    def update_population(self, S: List[int], cost: float) -> bool:
        """Ajoute une solution si suffisamment différente"""
        if self.is_too_close(S, cost):
            return False
        self.P.append((S.copy(), cost))
        if len(self.P) > self.population_size:
            self.P.sort(key=lambda x: x[1])
            self.P.pop()
        self.P.sort(key=lambda x: x[1])
        return True

    def M1(self, Pj: List[int]) -> List[int]:
        """M1 - Mutation aléatoire selon Pullan"""
        return M1_global(Pj, self.n, self.p)

    def M2(self, Pi: List[int]) -> List[int]:
        """M2 - Mutation dirigée selon Pullan"""
        return M2_global(Pi, self.distances)

    def X1(self, Pi: List[int], Pj: List[int]) -> List[int]:
        """X1 - Crossover aléatoire selon Pullan"""
        return X1_global(Pi, Pj, self.n, self.p)

    def X2(self, Pi: List[int], Pj: List[int]) -> Tuple[List[int], List[int]]:
        """X2 - Crossover phénotype selon Pullan - VERSION CORRIGÉE"""
        return X2_global(Pi, Pj, self.n, self.p, self.distances)

    def perform_generation(self, parallel: bool = True):
        """Effectue une génération avec parallélisation corrigée"""
        tasks = []
        for i, (Pi, _) in enumerate(self.P):
            for j, (Pj, _) in enumerate(self.P):
                if i == j:
                    continue
                # Ne pas passer distances et neighbors dans args
                tasks.append((i, j, Pi, Pj, self.generation,
                              self.n, self.p, self.ls_verbose))
        
        print(f"[PBS] Génération {self.generation+1}: {len(tasks)} tâches")
        gen_start_time = time.time()
        
        if parallel and len(tasks) > 1:
            try:
                parallel_start = time.time()
                # Utiliser le nombre optimal de processus
                n_processes = min(cpu_count(), len(tasks))

                
                # Créer le pool avec initialisation des workers
                with Pool(
                    processes=n_processes,
                    initializer=init_worker,
                    initargs=(self.distances, self.neighbors, os.getpid() + self.generation)
                ) as pool:
                    all_results = pool.map(process_pair_global, tasks)
                
                parallel_time = time.time() - parallel_start
                print(f"[PBS] Parallélisation réussie avec {n_processes} processus (temps: {parallel_time:.2f}s)")
                
            except Exception as e:
                print(f"[PBS] Erreur parallélisation: {type(e).__name__}: {e}")
                print(f"[PBS] Fallback séquentiel...")
                
                # Initialiser les variables globales pour le mode séquentiel
                import sys
                sys.modules[__name__]._global_distances = self.distances
                sys.modules[__name__]._global_neighbors = self.neighbors
                
                seq_start = time.time()
                all_results = []
                for t in tasks:
                    try:
                        result = process_pair_global(t)
                        all_results.append(result)
                    except Exception as task_e:
                        print(f"[PBS] Erreur tâche séquentielle: {type(task_e).__name__}: {task_e}")
                        all_results.append([])
                
                seq_time = time.time() - seq_start
                print(f"[PBS] Traitement séquentiel terminé (temps: {seq_time:.2f}s)")
        else:
            # Mode séquentiel direct
            import sys
            sys.modules[__name__]._global_distances = self.distances
            sys.modules[__name__]._global_neighbors = self.neighbors
            
            seq_start = time.time()
            all_results = []
            for t in tasks:
                try:
                    result = process_pair_global(t)
                    all_results.append(result)
                except Exception as task_e:
                    print(f"[PBS] Erreur tâche: {type(task_e).__name__}: {task_e}")
                    all_results.append([])
            
            seq_time = time.time() - seq_start
            print(f"[PBS] Traitement séquentiel (temps: {seq_time:.2f}s)")
        
        # Traiter tous les résultats
        improvements = 0
        for group in all_results:
            for sol, cost in group:
                if self.update_population(sol, cost):
                    improvements += 1
        
        self.generation += 1
        gen_total_time = time.time() - gen_start_time
        print(f"[PBS] Fin gen {self.generation}, meilleur Sc = {self.P[0][1]:.4f}, {improvements} améliorations (temps total: {gen_total_time:.2f}s)\n")
    
    
    def parallel_init_population(self):
        args_list = [
        (self.n, self.p, self.distances, self.neighbors, self.ls_verbose, i)
        for i in range(self.population_size)
        ]
        with Pool(self.population_size) as pool:
             results = pool.map(pbs_init_worker, args_list)
        for sol, cost in results:
            self.P.append((sol, cost))

    def run(
        self,
        target_cost: Optional[float] = None,
        max_generations: int = 30,
        parallel: bool = True
    ) -> Tuple[List[int], float]:
        """Exécute l'algorithme PBS principal"""
        
        # Import local pour éviter les problèmes circulaires
         
        
        # Initialisation de la population P
        print("[PBS] Initialisation de la population...")
        init_start_time = time.time()
        
        self.parallel_init_population()

        # Trier la population par coût
        self.P.sort(key=lambda x: x[1])
        init_time = time.time() - init_start_time
        print(f"[PBS] Init pop terminée en {init_time:.2f}s, meilleur Sc = {self.P[0][1]:.4f}\n")

        generations_start_time = time.time()
        total_generations = 0
        
        # Boucle principale des générations
        while self.generation < max_generations:
            if target_cost is not None and self.P[0][1] <= target_cost:
                print(f"[PBS] Objectif atteint en gen {self.generation}, Sc = {self.P[0][1]:.4f}\n")
                break
            
            self.perform_generation(parallel)
            total_generations += 1
        
        generations_time = time.time() - generations_start_time if total_generations > 0 else 0
        total_time = time.time() - init_start_time
        
        best_sol, best_cost = self.P[0]
        
        print(f"[PBS] === RÉSUMÉ FINAL ===")
        print(f"[PBS] Temps initialisation: {init_time:.2f}s")
        print(f"[PBS] Temps générations: {generations_time:.2f}s ({total_generations} générations)")
        if total_generations > 0:
            print(f"[PBS] Temps moyen par génération: {generations_time/total_generations:.2f}s")
        print(f"[PBS] Temps total: {total_time:.2f}s")
        print(f"[PBS] Meilleur coût: {best_cost:.4f}")
        print(f"[PBS] =====================\n")
        
        return best_sol, best_cost


# Fonctions utilitaires
def create_instance(n: int, p: int, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crée une instance avec matrice de distances et voisins triés"""
    distances = np.array(distances, dtype=np.int32)  # int32 au lieu de float64
    neighbors = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        neighbors[i] = np.argsort(distances[i])
    return distances, neighbors


def read_graph_instance(path: str) -> Tuple[int, int, List[List[int]]]:  # Changé float → int
    """Lit une instance de graphe depuis un fichier"""
    with open(path, 'r') as f:
        line = f.readline()
        n, _, p = map(int, line.strip().split())
        
        # Initialiser la matrice de distances EN ENTIERS
        dist = [[999999] * n for _ in range(n)]  # Au lieu de float('inf')
        for i in range(n):
            dist[i][i] = 0
        
        # Lire les arêtes
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            i, j, d = map(int, parts)  # d reste en int
            dist[i-1][j-1] = d
            dist[j-1][i-1] = d
        
        # Floyd-Warshall - tout reste en entiers
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
    
    return n, p, dist

def read_instance(filepath: str) -> Tuple[int, int, np.ndarray]:
    """Lit une instance et retourne n, p, distances"""
    n, p, dist = read_graph_instance(filepath)
    return n, p, np.array(dist, dtype=np.float64)


def run_pbs(
    n: int,
    p: int,
    dist_list: List[List[float]],
    target_cost: Optional[float] = None,
    max_generations: int = 100,
    parallel: bool = True,
    ls_verbose: bool = False
) -> Tuple[List[int], float]:
    """Interface simplifiée pour exécuter PBS"""
    distances, neighbors = create_instance(n, p, dist_list)
    pbs = OptimizedPBSAlgorithm(n, p, distances, neighbors, ls_verbose=ls_verbose)
    return pbs.run(target_cost, max_generations, parallel)