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
        ls_verbose: bool = False,          # ← nouveau flag
    ):
        self.n = n
        self.p = p
        self.distances = distances
        self.neighbors = neighbors
        self.ls_verbose = ls_verbose       # ← mémoriser

        self.population_size = 10
        self.P: List[Tuple[List[int], float]] = []
        self.generation = 0

    def create_initial_solution(self) -> List[int]:
        """C - Crée une solution avec un sommet aléatoire"""
        return [np.random.randint(0, self.n)]

    def is_too_close(self, S: List[int], cost: float) -> bool:
        """Vérifie la diversité - mêmes coûts ou >90% de facilities communes"""
        for Pi, Pi_cost in self.P:
            if abs(cost - Pi_cost) < 1e-9:
                return True
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
        """M1 - Mutation aléatoire"""
        q = random.randint(self.p // 2, self.p)
        selected = random.sample(Pj, min(q, len(Pj)))
        available = [v for v in range(self.n) if v not in selected]
        if len(selected) < self.p and available:
            selected.extend(random.sample(available, self.p - len(selected)))
        return selected

    def M2(self, Pi: List[int]) -> List[int]:
        """M2 - Mutation dirigée"""
        if len(Pi) < 2:
            return Pi.copy()
        min_dist = np.inf
        ri, rj = 0, 1
        for i in range(len(Pi)):
            for j in range(i + 1, len(Pi)):
                d = self.distances[Pi[i], Pi[j]]
                if d < min_dist:
                    min_dist = d
                    ri, rj = i, j
        return [c for k, c in enumerate(Pi) if k not in (ri, rj)]

    def X1(self, Pi: List[int], Pj: List[int]) -> List[int]:
        """X1 - Crossover aléatoire"""
        comb = list(set(Pi) | set(Pj))
        if len(comb) >= self.p:
            return random.sample(comb, self.p)
        avail = [v for v in range(self.n) if v not in comb]
        return comb + random.sample(avail, self.p - len(comb))

    def X2(self, Pi: List[int], Pj: List[int]) -> Tuple[List[int], List[int]]:
        """X2 - Crossover phénotype"""
        u1, u2 = random.sample(range(self.n), 2)
        q = random.random()
        S1, S2 = [], []
        for f in set(Pi) | set(Pj):
            d1 = self.distances[f, u1]
            d2 = self.distances[f, u2]
            if d2 <= 1e-9:
                continue
            if f in Pi and f not in Pj:
                (S1 if d1 / d2 <= q else S2).append(f)
            elif f in Pj and f not in Pi:
                (S1 if d1 / d2 >  q else S2).append(f)
            else:
                (S1 if d1 / d2 <= q else S2).append(f)
        return S1[:self.p], S2[:self.p]

    def _process_pair_parallel(self, args) -> List[Tuple[List[int], float]]:
        i, j, Pi, Pj, generation, n, p, distances, neighbors = args
        results = []
        # L(M1(Pj))
        S = self.M1(Pj)
        ls = OptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=self.ls_verbose)
        sol, cost = ls.search(S, generation)
        results.append((sol, cost))
        # L(M2(X1(Pi,Pj)))
        S = self.X1(Pi, Pj)
        S = self.M2(S)
        ls = OptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=self.ls_verbose)
        sol, cost = ls.search(S, generation)
        results.append((sol, cost))
        # X2 → M2
        S1, S2 = self.X2(Pi, Pj)
        for seed in (S1, S2):
            Snew = self.M2(seed)
            ls = OptimizedLocalSearchPCenter(n, p, distances, neighbors, verbose=self.ls_verbose)
            sol, cost = ls.search(Snew, generation)
            results.append((sol, cost))
        return results

    def perform_generation(self, parallel: bool = True):
        tasks = []
        for i, (Pi, _) in enumerate(self.P):
            for j, (Pj, _) in enumerate(self.P):
                if i == j:
                    continue
                tasks.append((i, j, Pi, Pj, self.generation,
                              self.n, self.p, self.distances, self.neighbors))
        print(f"[PBS] Génération {self.generation+1}: {len(tasks)} tâches")
        if parallel:
            with Pool(processes=4) as pool:
                all_res = pool.map(self._process_pair_parallel, tasks)
        else:
            all_res = [self._process_pair_parallel(t) for t in tasks]
        for group in all_res:
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
        # initialisation de P
        for _ in range(self.population_size):
            S0 = self.create_initial_solution()
            ls = OptimizedLocalSearchPCenter(self.n, self.p, self.distances, self.neighbors, verbose=self.ls_verbose)
            sol, cost = ls.search(S0, 0)
            self.P.append((sol, cost))
        self.P.sort(key=lambda x: x[1])
        print(f"[PBS] Init pop, meilleur Sc = {self.P[0][1]:.4f}\n")

        start = time.time()
        while self.generation < max_generations:
            if target_cost is not None and self.P[0][1] <= target_cost:
                print(f"[PBS] Objectif atteint en gen {self.generation}, Sc = {self.P[0][1]:.4f}\n")
                break
            self.perform_generation(parallel)
        elapsed = time.time() - start

        best_sol, best_cost = self.P[0]
        print(f"[PBS] Terminé en {elapsed:.2f}s, gén.={self.generation}, Sc={best_cost:.4f}\n")
        return best_sol, best_cost

# … (reste des utilitaires : create_instance, read_graph_instance, etc.)



def create_instance(n: int, p: int, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.array(distances, dtype=np.float64)
    neighbors = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        neighbors[i] = np.argsort(distances[i])
    return distances, neighbors


def read_graph_instance(path: str) -> Tuple[int, int, List[List[float]]]:
    with open(path, 'r') as f:
        line = f.readline()
        n, _, p = map(int, line.strip().split())
        dist = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            i, j, d = map(int, parts)
            dist[i-1][j-1] = d
            dist[j-1][i-1] = d
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
    return n, p, dist


def read_instance(filepath: str) -> Tuple[int, int, np.ndarray]:
    n, p, dist = read_graph_instance(filepath)
    return n, p, np.array(dist, dtype=np.float64)


def run_pbs(
    n: int,
    p: int,
    dist_list: List[List[float]],
    target_cost: Optional[float] = None,
    max_generations: int = 100,
    parallel: bool = True
) -> Tuple[List[int], float]:
    distances, neighbors = create_instance(n, p, dist_list)
    pbs = OptimizedPBSAlgorithm(n, p, distances, neighbors)
    return pbs.run(target_cost, max_generations, parallel)


