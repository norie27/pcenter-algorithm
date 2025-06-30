


import numpy as np
import random
from typing import List, Optional, Tuple
import math
from numba import njit, jit
# Ajout de l'import pour récupérer les paramètres de l'instance si besoin
from src.utils.instance import Instance


# Fonctions Numba pour les calculs intensifs
@njit
def find_next_numba(v: int, S: np.ndarray, F0: np.ndarray, distances: np.ndarray, n_facilities: int) -> Tuple[float, int]:
    """Version Numba de find_next"""
    min_dist = np.inf
    second = -1
    for i in range(n_facilities):
        f = S[i]
        if f != F0[v] and distances[v, f] < min_dist:
            min_dist = distances[v, f]
            second = f
    return min_dist, second


@njit
def update_after_add_numba(n: int, f: int, distances: np.ndarray, 
                          F0: np.ndarray, D0: np.ndarray, 
                          F1: np.ndarray, D1: np.ndarray):
    """Version Numba pour mettre à jour après ajout d'une facility"""
    for v in range(n):
        dist_to_f = distances[v, f]
        if dist_to_f < D0[v]:
            D1[v] = D0[v]
            F1[v] = F0[v]
            D0[v] = dist_to_f
            F0[v] = f
        elif dist_to_f < D1[v]:
            D1[v] = dist_to_f
            F1[v] = f
    return np.max(D0)


@njit
def calculate_swap_cost_numba(n: int, i: int, f: int, S: np.ndarray, n_facilities: int,
                              F0: np.ndarray, D0: np.ndarray, F1: np.ndarray, D1: np.ndarray,
                              distances: np.ndarray) -> float:
    """Calcule le coût d'un swap (sortir f, entrer i)"""
    max_cost = 0.0
    
    for v in range(n):
        # Skip si v est une facility
        is_facility = False
        for j in range(n_facilities):
            if S[j] == v:
                is_facility = True
                break
        if is_facility:
            continue
        
        # Coût si on supprime f
        if F0[v] == f:
            # v serait servi par sa deuxième facility ou par i
            cost_without_f = min(distances[i, v], D1[v])
        else:
            # v garde sa facility actuelle
            cost_without_f = min(distances[i, v], D0[v])
        
        max_cost = max(max_cost, cost_without_f)
    
    return max_cost


class VerboseOptimizedLocalSearchPCenter:
    """Recherche locale optimisée (Pullan) avec niveau de verbosité contrôlable - AVEC NUMBA"""

    def __init__(self,
                 n: int,
                 p: int,
                 distances: np.ndarray,
                 neighbors: np.ndarray,
                 verbose: bool = False):
        self.n = n
        self.p = p
        self.distances = distances
        self.neighbors = neighbors
        self.verbose = verbose
        # Closest and second-closest facility for each vertex
        self.F0 = np.full(n, -1, dtype=np.int32)
        self.D0 = np.full(n, np.inf, dtype=np.float64)
        self.F1 = np.full(n, -1, dtype=np.int32)
        self.D1 = np.full(n, np.inf, dtype=np.float64)
        self.Sc = 0.0
        self.S: List[int] = []
        self.S_set = set()
        self.forbidden_swaps = set()
        # Buffers for fast state save/restore
        self._saved_F0 = np.empty_like(self.F0)
        self._saved_D0 = np.empty_like(self.D0)
        self._saved_F1 = np.empty_like(self.F1)
        self._saved_D1 = np.empty_like(self.D1)
        self._saved_Sc = 0.0
        self._saved_S = []
        self._saved_S_set = set()

    def add_facility(self, f: int):
        if self.verbose:
            print(f"[ADD_FACILITY] Ajout du centre {f}")
        
        # Utiliser la version Numba pour la mise à jour
        self.Sc = update_after_add_numba(self.n, f, self.distances, 
                                         self.F0, self.D0, self.F1, self.D1)
        
        if self.verbose:
            dist_to_f = self.distances[:, f]
            mask0 = dist_to_f < self.D0
            mask1 = (~mask0) & (dist_to_f < self.D1)
            for v in np.where(mask0)[0]:
                old = self.D0[v]
                prev = self.F0[v] if self.F0[v]>=0 else 'aucun'
                print(f"  noeud {v}: {f} devient F0 (était {prev}), distance {dist_to_f[v]:.2f} < {old:.2f}")
            for v in np.where(mask1)[0]:
                old = self.D1[v]
                print(f"  noeud {v}: {f} devient F1, distance {dist_to_f[v]:.2f} < {old:.2f}")
        
        self.S.append(f)
        self.S_set.add(f)
        
        if self.verbose:
            print(f"[ADD_FACILITY] Nouveau coût Sc = {self.Sc:.2f}\n")

    def remove_facility(self, f: int):
        if self.verbose:
            print(f"[REMOVE_FACILITY] Suppression du centre {f}")
        self.S.remove(f)
        self.S_set.remove(f)
        
        # Convertir S en numpy array pour Numba
        S_array = np.array(self.S, dtype=np.int32)
        
        affected0 = np.where(self.F0==f)[0]
        affected1 = np.where(self.F1==f)[0]
        for v in affected0:
            self.D0[v] = self.D1[v]
            self.F0[v] = self.F1[v]
            d, s = find_next_numba(v, S_array, self.F0, self.distances, len(self.S))
            self.D1[v], self.F1[v] = d, s
            if self.verbose:
                print(f"  noeud {v}: ancien F0 supprimé, nouveau F0={self.F0[v]}, D0={self.D0[v]:.2f}, F1={s}, D1={d:.2f}")
        for v in affected1:
            d, s = find_next_numba(v, S_array, self.F0, self.distances, len(self.S))
            self.D1[v], self.F1[v] = d, s
            if self.verbose:
                print(f"  noeud {v}: ancien F1 supprimé, nouveau F1={s}, D1={d:.2f}")
        self.Sc = np.max(self.D0)
        if self.verbose:
            print(f"[REMOVE_FACILITY] Nouveau coût Sc = {self.Sc:.2f}\n")

    def find_next(self, v: int) -> Tuple[float,int]:
        min_dist = np.inf
        second = -1
        for f in self.S:
            if f != self.F0[v] and self.distances[v,f] < min_dist:
                min_dist = self.distances[v,f]
                second = f
        return min_dist, second

    def find_pair(self, w: int) -> Tuple[Optional[int],Optional[int]]:
        """Find Pair selon Pullan - AVEC NUMBA"""
        if self.verbose:
            print(f"[FIND_PAIR] Début pour w={w}")
        C = np.max(self.distances)
        L = []
        
        # Trouver k selon Pullan : index de la facility actuelle de w
        current = self.F0[w]
        if current >= 0:
            k_idx = np.where(self.neighbors[w] == current)[0]
            k = k_idx[0] + 1 if len(k_idx) > 0 else self.n  # k+1 pour inclure la facility actuelle
        else:
            k = self.n  # Si pas de facility assignée, considérer tous
        
        # Prendre les k premiers voisins selon Pullan
        Nwk = self.neighbors[w][:k]
        if self.verbose:
            print(f"  k={k}, Nwk={Nwk[:10].tolist()}..." if k > 10 else f"  k={k}, Nwk={Nwk.tolist()}")
        
        # Candidats : voisins pas encore dans S
        candidates = [v for v in Nwk if v not in self.S_set]
        
        if self.verbose:
            print(f"  candidats={len(candidates)} total")
        if not candidates:
            if self.verbose:
                print("  Aucun candidat trouvé\n")
            return None, None

        # Sauvegarder l'état
        np.copyto(self._saved_F0, self.F0)
        np.copyto(self._saved_D0, self.D0)
        np.copyto(self._saved_F1, self.F1)
        np.copyto(self._saved_D1, self.D1)
        saved_Sc = self.Sc
        saved_S = self.S.copy()
        saved_S_set = self.S_set.copy()

        # Convertir S en numpy array pour Numba
        S_array = np.array(self.S, dtype=np.int32)

        for i in candidates:
            if self.verbose:
                print(f"  Test candidat {i}")
            
            # Ajouter temporairement la facility
            self.add_facility(i)
            
            # Recréer S_array avec la nouvelle facility
            S_array_with_i = np.array(self.S, dtype=np.int32)
            
            # Calculer le coût de suppression de chaque facility existante
            M = {}
            for f in self.S:
                if f == i:
                    continue
                
                # Utiliser Numba pour calculer le coût du swap
                M[f] = calculate_swap_cost_numba(self.n, i, f, S_array_with_i, len(self.S),
                                                self.F0, self.D0, self.F1, self.D1,
                                                self.distances)
            
            # Trouver le meilleur swap
            for f in M:
                Mf = M[f]
                if self.verbose and len(candidates) <= 5:  # Verbose seulement si peu de candidats
                    print(f"    swap: sortir {f}, entrer {i}, coût={Mf:.2f}")
                
                if Mf < C:
                    L = [(f, i)]
                    C = Mf
                elif abs(Mf - C) < 1e-12:
                    L.append((f, i))
            
            # Restaurer l'état
            np.copyto(self.F0, self._saved_F0)
            np.copyto(self.D0, self._saved_D0)
            np.copyto(self.F1, self._saved_F1)
            np.copyto(self.D1, self._saved_D1)
            self.Sc = saved_Sc
            self.S = saved_S.copy()
            self.S_set = saved_S_set.copy()

        if L and self.verbose:
            choice = random.choice(L)
            print(f"[FIND_PAIR] Choix: sortir {choice[0]}, entrer {choice[1]} (C={C:.2f})\n")
        
        return random.choice(L) if L else (None, None)

    def search(self,
               initial_solution: Optional[List[int]] = None,
               generation: int = 0) -> Tuple[List[int], float]:
        if self.verbose:
            print("=== RECHERCHE LOCALE ===")
        
        if initial_solution is None:
            S0 = [random.randint(0, self.n-1)]
        else:
            S0 = initial_solution.copy()
        
        if self.verbose:
            print(f"n={self.n}, p={self.p}, init={S0}, gen={generation}\n")
        
        # Réinitialiser l'état
        self.S = []
        self.S_set.clear()
        self.F0.fill(-1)
        self.D0.fill(np.inf)
        self.F1.fill(-1)
        self.D1.fill(np.inf)
        self.Sc = 0.0
        self.forbidden_swaps.clear()
        
        # Ajouter les facilities initiales
        for f in S0:
            self.add_facility(f)
        
        if self.verbose:
            print("=== PHASE 1: CONSTRUCTION ===")
        
        
        while len(self.S) < self.p:
            # Trouver le vertex le plus critique (distance max)
            critical_vertices = np.where(np.abs(self.D0 - self.Sc) < 1e-9)[0]
            non_facilities = [v for v in range(self.n) if v not in self.S_set]
            
            if len(critical_vertices) > 0:
                w = random.choice(critical_vertices)
            else:
                w = max(non_facilities, key=lambda v: self.D0[v]) if non_facilities else 0
            
            if self.verbose:
                print(f"[Construction] w={w}, Sc={self.Sc:.2f}")
            
            # Selon Pullan : sélectionner uniformément dans Nwk
            current = self.F0[w]
            if current >= 0:
                k_idx = np.where(self.neighbors[w] == current)[0]
                k = k_idx[0] if len(k_idx) > 0 else 1
            else:
                k = 1
            
            Nwk = self.neighbors[w][:k+1]  # Include k+1 selon l'article
            candidates = [v for v in Nwk if v not in self.S_set]
            
            if not candidates:
                candidates = non_facilities
            
            newf = random.choice(candidates) if candidates else random.choice(non_facilities)
            
            if self.verbose:
                print(f"[Construction] Ajout centre {newf}\n")
            self.add_facility(newf)
        
        if self.verbose:
            print("=== PHASE 2: AMÉLIORATION ===")
        
        # Phase d'amélioration - CRITÈRES AJUSTÉS
        max_iter = 2 * self.n  
        max_no_improve = max(int(0.2 * (generation + 1) * self.n), self.n)  # Plus flexible
        
        iteration = 0
        no_improve = 0
        best_cost = self.Sc
        best_sol = self.S.copy()
        
        while iteration < max_iter and no_improve < max_no_improve:
            # Trouver les vertices critiques
            critical = np.where(np.abs(self.D0 - self.Sc) < 1e-9)[0]
            if len(critical) == 0:
                break
            
            w = random.choice(critical)
            
            if self.verbose:
                print(f"[Amélioration] iter {iteration}, w={w}, Sc={self.Sc:.2f}")
            
            f_out, v_in = self.find_pair(w)
            
            if f_out is None or (f_out, v_in) in self.forbidden_swaps:
                no_improve += 1
                iteration += 1
                continue
            
            if self.verbose:
                print(f"[Amélioration] Swap: sortir {f_out}, entrer {v_in}\n")
            
            # Effectuer le swap
            self.remove_facility(f_out)
            self.add_facility(v_in)
            self.forbidden_swaps.add((f_out, v_in))
            
            # Nettoyer les swaps interdits si trop nombreux
            if len(self.forbidden_swaps) > self.p * self.n:
                self.forbidden_swaps.clear()
            
            if self.Sc < best_cost - 1e-12:  # Amélioration significative
                if self.verbose:
                    print(f"[Amélioration] Amélioration: {best_cost:.2f} -> {self.Sc:.2f}\n")
                best_cost = self.Sc
                best_sol = self.S.copy()
                no_improve = 0
            else:
                if self.verbose:
                    print(f"[Amélioration] Pas d'amélioration ({self.Sc:.2f})\n")
                no_improve += 1
            
            iteration += 1
        
        if self.verbose:
            print(f"=== FIN Recherche: iter={iteration}, meilleur coût = {best_cost:.2f}, solution = {best_sol} ===")
        
        return best_sol, best_cost