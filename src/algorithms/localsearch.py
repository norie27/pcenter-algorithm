# src/algorithms/localsearch_verbose.py

import numpy as np
import random
from typing import List, Optional, Tuple
import math
from src.utils.instance import Instance

class VerboseOptimizedLocalSearchPCenter:
    """Recherche locale optimisée (Pullan) avec niveau de verbosité contrôlable"""

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
        self.F0 = np.full(n, -1, dtype=np.int32)
        self.D0 = np.full(n, np.inf, dtype=np.float64)
        self.F1 = np.full(n, -1, dtype=np.int32)
        self.D1 = np.full(n, np.inf, dtype=np.float64)
        self.Sc = 0.0
        self.S: List[int] = []
        self.S_set = set()
        self.forbidden_swaps = set()
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
        dist_to_f = self.distances[:, f]
        mask0 = dist_to_f < self.D0
        mask1 = (~mask0) & (dist_to_f < self.D1)
        if self.verbose:
            for v in np.where(mask0)[0]:
                old = self.D0[v]
                prev = self.F0[v] if self.F0[v]>=0 else 'aucun'
                print(f"  noeud {v}: {f} devient F0 (était {prev}), distance {dist_to_f[v]:.2f} < {old:.2f}")
            for v in np.where(mask1)[0]:
                old = self.D1[v]
                print(f"  noeud {v}: {f} devient F1, distance {dist_to_f[v]:.2f} < {old:.2f}")
        self.D1[mask0] = self.D0[mask0]
        self.F1[mask0] = self.F0[mask0]
        self.D0[mask0] = dist_to_f[mask0]
        self.F0[mask0] = f
        self.D1[mask1] = dist_to_f[mask1]
        self.F1[mask1] = f
        self.S.append(f)
        self.S_set.add(f)
        self.Sc = np.max(self.D0)
        if self.verbose:
            print(f"[ADD_FACILITY] Nouveau coût Sc = {self.Sc:.2f}\n")

    def remove_facility(self, f: int):
        if self.verbose:
            print(f"[REMOVE_FACILITY] Suppression du centre {f}")
        self.S.remove(f)
        self.S_set.remove(f)
        affected0 = np.where(self.F0==f)[0]
        affected1 = np.where(self.F1==f)[0]
        for v in affected0:
            self.D0[v] = self.D1[v]
            self.F0[v] = self.F1[v]
            d, s = self.find_next(v)
            self.D1[v], self.F1[v] = d, s
            if self.verbose:
                print(f"  noeud {v}: ancien F0 supprimé, nouveau F0={self.F0[v]}, D0={self.D0[v]:.2f}, F1={s}, D1={d:.2f}")
        for v in affected1:
            d, s = self.find_next(v)
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
        if self.verbose:
            print(f"[FIND_PAIR] Début pour w={w}")
        C = np.max(self.distances)
        L = []
        # indice naturel de la facility la plus proche
        current = self.F0[w]
        if current>=0:
            k_idx = np.where(self.neighbors[w]==current)[0]
            k = k_idx[0] if len(k_idx)>0 else 0
        else:
            k = 0
        # Pas de plafonnement : explore tous les k premiers voisins
        # Taille minimale du voisinage
        MIN_K = 50
        k_eff = max(k, MIN_K)
        Nwk = self.neighbors[w][:k_eff]

        if self.verbose:
            print(f"  k={k}, Nwk={Nwk.tolist()}")
        candidates = [v for v in Nwk if v not in self.S_set]
        if self.verbose:
            print(f"  candidats={candidates}")
        if not candidates:
            if self.verbose:
                print("  Aucun candidat trouvé\n")
            return None, None
        # sauvegarde de l'état
        np.copyto(self._saved_F0,self.F0)
        np.copyto(self._saved_D0,self.D0)
        np.copyto(self._saved_F1,self.F1)
        np.copyto(self._saved_D1,self.D1)
        saved_Sc = self.Sc
        saved_S = self.S.copy()
        saved_S_set = self.S_set.copy()
        # exploration complète
        for i in candidates:
            if self.verbose:
                print(f"  Test swap out f<->in {i}")
            self.add_facility(i)
            M = {f:0 for f in self.S}
            for v in range(self.n):
                if v in self.S_set: continue
                mind = min(self.distances[i,v], self.D1[v])
                f0v = self.F0[v]
                if f0v in M and mind > M[f0v]:
                    M[f0v] = mind
            for f in self.S:
                if f==i: continue
                Mf = M.get(f,0)
                if self.verbose:
                    print(f"    swap tentativa: sortir {f}, entrer {i}, Mf={Mf:.2f}")
                if Mf < C:
                    L = [(f,i)]; C = Mf
                elif Mf == C:
                    L.append((f,i))
            # restauration
            np.copyto(self.F0,self._saved_F0)
            np.copyto(self.D0,self._saved_D0)
            np.copyto(self.F1,self._saved_F1)
            np.copyto(self.D1,self._saved_D1)
            self.Sc = saved_Sc
            self.S = saved_S.copy()
            self.S_set = saved_S_set.copy()
        if L and self.verbose:
            choice = random.choice(L)
            print(f"[FIND_PAIR] Choix: sortir {choice[0]}, entrer {choice[1]} (C={C:.2f})\n")
        return random.choice(L) if L else (None,None)

    def search(self,
               initial_solution: Optional[List[int]] = None,
               generation: int = 0) -> Tuple[List[int],float]:
        if self.verbose:
            print("=== RECHERCHE LOCALE OPTIMISÉE VERBOSE ===")
        if initial_solution is None:
            S0 = [random.randint(0,self.n-1)]
        else:
            S0 = initial_solution.copy()
        if self.verbose:
            print(f"n={self.n}, p={self.p}, init={S0}, gen={generation}\n")
        self.S=[]; self.S_set.clear()
        self.F0.fill(-1); self.D0.fill(np.inf)
        self.F1.fill(-1); self.D1.fill(np.inf)
        self.Sc=0.0; self.forbidden_swaps.clear()
        for f in S0:
            self.add_facility(f)
        if self.verbose:
            print("=== PHASE 1: CONSTRUCTION ===")
        while len(self.S)<self.p:
            critical = np.where(np.abs(self.D0 - self.Sc)<1e-9)[0]
            nonf = [v for v in range(self.n) if v not in self.S_set]
            w = random.choice(critical) if len(critical)>0 else max(nonf, key=lambda v: self.D0[v])
            if self.verbose:
                print(f"[Construction] w={w}, Sc={self.Sc:.2f}")
            current = self.F0[w]
            if current>=0:
                k_idx = np.where(self.neighbors[w]==current)[0]
                k = k_idx[0] if len(k_idx)>0 else 1
            else:
                k=1
            # plus de plafonnement ici
            
            MIN_K = 50
            k_eff = max(k, MIN_K)
            Nwk = self.neighbors[w][:k_eff]

            candidates = [v for v in Nwk if v not in self.S_set]
            if not candidates:
                candidates = nonf
            newf = random.choice(candidates)
            if self.verbose:
                print(f"[Construction] Ajout centre {newf}\n")
            self.add_facility(newf)
        if self.verbose:
            print("=== PHASE 2: AMÉLIORATION ===")
        max_iter = 2*self.n
        max_no = int(0.1*(generation+1)*self.n)
        it,no=0,0
        best_cost=self.Sc; best_sol=self.S.copy()
        while it<max_iter and no<max_no:
            crit = np.where(np.abs(self.D0-self.Sc)<1e-9)[0]
            if len(crit)==0: break
            w = random.choice(crit)
            if self.verbose:
                print(f"[Amélioration] iteration {it}, w={w}, Sc={self.Sc:.2f}")
            f_out, v_in = self.find_pair(w)
            if f_out is None or (f_out,v_in) in self.forbidden_swaps:
                no+=1; it+=1; continue
            if self.verbose:
                print(f"[Amélioration] Swap: sortir {f_out}, entrer {v_in}\n")
            self.remove_facility(f_out)
            self.add_facility(v_in)
            self.forbidden_swaps.add((f_out,v_in))
            if self.Sc<best_cost:
                if self.verbose:
                    print(f"[Amélioration] Amélioration: {best_cost:.2f} -> {self.Sc:.2f}\n")
                best_cost, best_sol, no = self.Sc, self.S.copy(), 0
            else:
                if self.verbose:
                    print(f"[Amélioration] Pas d'amélioration ({self.Sc:.2f})\n")
                no+=1
            it+=1
        if self.verbose:
            print(f"=== FIN Recherche: meilleur coût = {best_cost:.2f}, solution = {best_sol} ===")
        return best_sol, best_cost
