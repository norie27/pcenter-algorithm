import numpy as np
import random
import time
from typing import List, Tuple, Optional
from multiprocessing import Pool
import math

from .localsearch import VerboseOptimizedLocalSearchPCenter as OptimizedLocalSearchPCenter

# src/algorithms/pbs.py

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
        return [random.randint(0, self.n - 1)]  # Utiliser random.randint standard

    def is_too_close(self, S: List[int], cost: float) -> bool:
        """Vérifie la diversité - selon Pullan: même coût ou facilités trop similaires"""
        for Pi, Pi_cost in self.P:
            # Comparaison de coût plus stricte
            if abs(cost - Pi_cost) < 1e-12:
                return True
            # Seuil de similarité ajusté à 90% comme mentionné dans l'article
            if len(set(S) & set(Pi)) > 0.9 * self.p:
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
        q = random.randint(self.p // 2, self.p)
        selected = random.sample(Pj, min(q, len(Pj)))
        available = [v for v in range(self.n) if v not in selected]
        if len(selected) < self.p and available:
            needed = self.p - len(selected)
            selected.extend(random.sample(available, min(needed, len(available))))
        return selected

    def M2(self, Pi: List[int]) -> List[int]:
        """M2 - Mutation dirigée selon Pullan"""
        if len(Pi) < 2:
            return Pi.copy()
        
        min_dist = float('inf')
        ri, rj = 0, 1
        
        # Trouver la paire de facilities la plus proche
        for i in range(len(Pi)):
            for j in range(i + 1, len(Pi)):
                d = self.distances[Pi[i], Pi[j]]
                if d < min_dist:
                    min_dist = d
                    ri, rj = i, j
        
        # Retourner une solution incomplète (p-2 facilities)
        # La recherche locale devra la compléter
        return [c for k, c in enumerate(Pi) if k not in (ri, rj)]

    def X1(self, Pi: List[int], Pj: List[int]) -> List[int]:
        """X1 - Crossover aléatoire selon Pullan"""
        combined = list(set(Pi) | set(Pj))
        
        if len(combined) >= self.p:
            return random.sample(combined, self.p)
        else:
            # Compléter avec des vertices aléatoires
            available = [v for v in range(self.n) if v not in combined]
            needed = self.p - len(combined)
            if needed > 0 and available:
                additional = random.sample(available, min(needed, len(available)))
                combined.extend(additional)
            return combined

    def X2(self, Pi: List[int], Pj: List[int]) -> Tuple[List[int], List[int]]:
        """X2 - Crossover phénotype selon Pullan - CORRIGÉ"""
        # Sélectionner deux users aléatoires distincts
        u1, u2 = random.sample(range(self.n), 2)
        
        # q doit être dans [0.1, 0.9] selon l'article
        q = random.uniform(0.1, 0.9)
        
        S1, S2 = [], []
        Pi_set = set(Pi)
        Pj_set = set(Pj)
        
        for f in set(Pi) | set(Pj):
            d1 = self.distances[f, u1]
            d2 = self.distances[f, u2]
            
            # Gestion du cas d2 = 0 (éviter division par zéro)
            if d2 <= 1e-12:
                # Assigner à S1 par défaut (ou on pourrait faire aléatoirement)
                S1.append(f)
                continue
                
            ratio = d1 / d2
            
            if f in Pi_set and f not in Pj_set:
                # Facility uniquement dans Pi
                if ratio <= q:
                    S1.append(f)
                else:
                    S2.append(f)
            elif f in Pj_set and f not in Pi_set:
                # Facility uniquement dans Pj - CORRECTION CRITIQUE
                if ratio > q:
                    S1.append(f)
                else:
                    S2.append(f)
            else:
                # Facility dans les deux parents
                if ratio <= q:
                    S1.append(f)
                else:
                    S2.append(f)
        
        # Selon Pullan: si plus de p facilities, les réduire aléatoirement
        # Si moins de p, laisser la recherche locale compléter
        if len(S1) > self.p:
            S1 = random.sample(S1, self.p)
        if len(S2) > self.p:
            S2 = random.sample(S2, self.p)
            
        return S1, S2

    def _process_pair_parallel(self, args) -> List[Tuple[List[int], float]]:
        """Traitement parallèle d'une paire de solutions"""
        i, j, Pi, Pj, generation, n, p, distances, neighbors, ls_verbose = args
        results = []
        
        # L(M1(Pj))
        S = self.M1(Pj)
        ls = OptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=ls_verbose)
        sol, cost = ls.search(S, generation)
        results.append((sol, cost))
        
        # L(M2(X1(Pi,Pj)))
        S = self.X1(Pi, Pj)
        S = self.M2(S)
        ls = OptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=ls_verbose)
        sol, cost = ls.search(S, generation)
        results.append((sol, cost))
        
        # X2 → M2 pour chaque enfant
        S1, S2 = self.X2(Pi, Pj)
        for seed in (S1, S2):
            Snew = self.M2(seed)
            ls = OptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=ls_verbose)
            sol, cost = ls.search(Snew, generation)
            results.append((sol, cost))
        
        return results

    def perform_generation(self, parallel: bool = True):
        """Effectue une génération complète selon l'algorithme PBS"""
        tasks = []
        for i, (Pi, _) in enumerate(self.P):
            for j, (Pj, _) in enumerate(self.P):
                if i == j:
                    continue
                tasks.append((i, j, Pi, Pj, self.generation,
                              self.n, self.p, self.distances, self.neighbors, self.ls_verbose))
        
        print(f"[PBS] Génération {self.generation+1}: {len(tasks)} tâches")
        
        if parallel:
            with Pool(processes=4) as pool:
                all_results = pool.map(self._process_pair_parallel, tasks)
        else:
            all_results = [self._process_pair_parallel(t) for t in tasks]
        
        # Traiter tous les résultats
        for group in all_results:
            for sol, cost in group:
                self.update_population(sol, cost)
        
        self.generation += 1
        print(f"[PBS] Fin gen {self.generation}, meilleur Sc = {self.P[0][1]:.4f}\n")

    def run(
        self,
        target_cost: Optional[float] = None,
        max_generations: int = 30,
        parallel: bool = True
    ) -> Tuple[List[int], float]:
        """Exécute l'algorithme PBS principal"""
        
        # Initialisation de la population P
        print("[PBS] Initialisation de la population...")
        for _ in range(self.population_size):
            S0 = self.create_initial_solution()
            ls = OptimizedLocalSearchPCenter(self.n, self.p, self.distances, self.neighbors, verbose=self.ls_verbose)
            sol, cost = ls.search(S0, 0)
            self.P.append((sol, cost))
        
        # Trier la population par coût
        self.P.sort(key=lambda x: x[1])
        print(f"[PBS] Init pop, meilleur Sc = {self.P[0][1]:.4f}\n")

        start_time = time.time()
        
        # Boucle principale des générations
        while self.generation < max_generations:
            if target_cost is not None and self.P[0][1] <= target_cost:
                print(f"[PBS] Objectif atteint en gen {self.generation}, Sc = {self.P[0][1]:.4f}\n")
                break
            
            self.perform_generation(parallel)
        
        elapsed_time = time.time() - start_time
        best_sol, best_cost = self.P[0]
        
        print(f"[PBS] Terminé en {elapsed_time:.2f}s, gén.={self.generation}, Sc={best_cost:.4f}\n")
        return best_sol, best_cost


# Fonctions utilitaires

def create_instance(n: int, p: int, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crée une instance avec matrice de distances et voisins triés"""
    distances = np.array(distances, dtype=np.float64)
    neighbors = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        neighbors[i] = np.argsort(distances[i])
    return distances, neighbors


def read_graph_instance(path: str) -> Tuple[int, int, List[List[float]]]:
    """Lit une instance de graphe depuis un fichier"""
    with open(path, 'r') as f:
        line = f.readline()
        n, _, p = map(int, line.strip().split())
        
        # Initialiser la matrice de distances
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        
        # Lire les arêtes
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            i, j, d = map(int, parts)
            dist[i-1][j-1] = d
            dist[j-1][i-1] = d
        
        # Floyd-Warshall pour les plus courts chemins
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