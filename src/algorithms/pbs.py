import numpy as np
import random
import time
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count
import os
from .genetic_operators import M1_global, M2_global, X1_global, X2_global
from .parallel_worker import init_worker, process_pair_global,pbs_init_worker
from .localsearch import VerboseOptimizedLocalSearchPCenter as OptimizedLocalSearchPCenter
from src.utils.read_instance import create_instance

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
                n_processes = min(8, cpu_count(), len(tasks))
                
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