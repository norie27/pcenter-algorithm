


from typing import List, Tuple
import numpy as np
import random
import math

# -> Ajoute cet import :
from src.utils.instance import Instance



class LocalSearchPCenter:
    """Recherche locale CORRIGÉE selon l'article PBS"""

    def __init__(self, instance: Instance):
        self.instance = instance
        self.n = instance.n
        self.p = instance.p
        self.distances = instance.dist
        self.neighbors = instance.N
        self.F0 = {}  # F0v = facility which is closest to vertex v
        self.D0 = {}  # D0v = distance from vertex v to the closest facility
        self.F1 = {}  # F1v = facility which is second closest to vertex v
        self.D1 = {}  # D1v = distance from vertex v to the second closest facility
        self.Sc = 0   # max{mind(vi,vj)}, vi ∈ V, vj ∈ S
        self.forbidden_swaps = set()

    def add_facility(self, f: int):
        """Procedure Add Facility - exactement selon l'article"""
        print(f"    [ADD_FACILITY] Ajout de la facilité {f}")
        self.Sc = 0
        for v in range(self.n):
            dfv = self.distances[v][f]
            old_D0 = self.D0.get(v, float('inf'))
            
            if v not in self.D0 or dfv < self.D0[v]:
                if v in self.D0:
                    self.D1[v] = self.D0[v]
                    self.F1[v] = self.F0[v]
                    print(f"      Vertex {v}: {f} devient F0 (était {self.F0[v]}), distance {dfv:.2f} < {old_D0:.2f}")
                else:
                    print(f"      Vertex {v}: {f} devient F0 (première facilité), distance {dfv:.2f}")
                self.D0[v] = dfv
                self.F0[v] = f
            elif v not in self.D1 or dfv < self.D1[v]:
                old_D1 = self.D1.get(v, float('inf'))
                self.D1[v] = dfv
                self.F1[v] = f
                print(f"      Vertex {v}: {f} devient F1, distance {dfv:.2f} < {old_D1:.2f}")
            
            if self.D0[v] > self.Sc:
                self.Sc = self.D0[v]
        
        print(f"    [ADD_FACILITY] Nouveau coût Sc = {self.Sc:.2f}")

    def _find_next(self, v: int, S: List[int], exclude: int) -> Tuple[float, int]:
        """Find Next(v) - trouve la 2ème facilité la plus proche"""
        min_dist = float('inf')
        closest = -1
        for s in S:
            if s != exclude and self.distances[v][s] < min_dist:
                min_dist = self.distances[v][s]
                closest = s
        return min_dist, closest

    def find_pair(self, w: int, S: List[int], verbose: bool = True) -> Tuple[int, int, List]:
        """Function Find Pair - EXACTEMENT selon l'article PBS"""
        C = self.distances.max()  # Line 1: C ← max(dij)
        L = []  # Liste des meilleures paires
        swap_attempts = []
        
        # Déterminer k selon l'article: "k is the index in Nw of the facility that currently services vertex w"
        current_facility = self.F0.get(w, -1)
        if current_facility == -1:
            k = 0
            print(f"    [FIND_PAIR] w={w} n'a pas de facilité assignée, k=0")
            Nwk = []  # Pas de voisins à considérer
        else:
            # k est l'INDEX de F0[w] dans la liste des voisins Nw
            k = np.where(self.neighbors[w] == current_facility)[0][0]
            print(f"    [FIND_PAIR] w={w}, F0[w]={current_facility}, k={k} (index dans Nw)")
            # CORRECTION: Nwk = les k premiers éléments (pas k+1)
            Nwk = self.neighbors[w][:k]  # Éléments de 0 à k-1
        
        if verbose:
            print(f"    [FIND_PAIR] Nw complet: {self.neighbors[w].tolist()}")
            print(f"    [FIND_PAIR] Nwk (les {k} premiers éléments): {Nwk.tolist()}")
        
        # Candidats dans Nwk non dans S
        candidates = [v for v in Nwk if v not in S]
        
        if verbose:
            print(f"    [FIND_PAIR] Candidats (non dans S): {candidates}")
        
        if not candidates:
            print(f"    [FIND_PAIR] Aucun candidat disponible dans Nwk")
            return None, None, swap_attempts
        
        # Pour chaque candidat i dans Nwk (Line 2)
        for i in candidates:
            print(f"\n      [FIND_PAIR] Test candidat i={i}")
            
            # Sauvegarder l'état
            saved_state = {
                'F0': self.F0.copy(),
                'D0': self.D0.copy(),
                'F1': self.F1.copy(),
                'D1': self.D1.copy(),
                'Sc': self.Sc
            }
            
            # Add Facility(i) - Line 3
            self.add_facility(i)
            
            # ∀ f ∈ S, Mf ← 0 - Line 4
            M = {f: 0 for f in S}
            
            # ∀ v ∈ V - S - Line 5
            print(f"      [FIND_PAIR] Calcul des Mf pour chaque facilité f dans S")
            for v in range(self.n):
                if v in S or v == i:  # v est une facilité
                    continue
                
                # Calculer min(div, D1v) - Lines 6-7
                div = self.distances[i][v]
                d1v = self.D1.get(v, float('inf'))
                min_dist = min(div, d1v)
                
                if v in self.F0:
                    f0v = self.F0[v]
                    if f0v in M and min_dist > M[f0v]:
                        M[f0v] = min_dist
                        print(f"        M[{f0v}] = {min_dist:.2f} (depuis vertex {v})")
            
            # ∀ f ∈ S - Lines 8-12
            for f in S:
                Mf = M.get(f, 0)
                print(f"      [FIND_PAIR] Facilité f={f}, Mf={Mf:.2f}")
                
                swap_attempts.append((f, i, Mf))
                
                if Mf == C:
                    L.append((f, i))
                elif Mf < C:
                    L = [(f, i)]
                    C = Mf
                    print(f"      [FIND_PAIR] Nouveau meilleur coût C={C:.2f}")
            
            # Restaurer l'état - Line 13 (Remove Facility(i))
            self.F0 = saved_state['F0']
            self.D0 = saved_state['D0']
            self.F1 = saved_state['F1']
            self.D1 = saved_state['D1']
            self.Sc = saved_state['Sc']
        
        if verbose and L:
            print(f"    [FIND_PAIR] Meilleures paires: {L}, coût: {C:.2f}")
        
        # return (f,i) ∈ L - Line 14 (sélection uniforme aléatoire)
        if L:
            chosen = random.choice(L)
            print(f"    [FIND_PAIR] Paire choisie aléatoirement: {chosen}")
            return chosen[0], chosen[1], swap_attempts
        
        return None, None, swap_attempts

    def search(self, initial_solution: List[int] = None, generation: int = 0,
               max_iterations: int = None, verbose: bool = True) -> Tuple[List[int], float]:
        """Local Search L - selon l'article PBS avec g pour generation count"""
        
        if max_iterations is None:
            max_iterations = 2 * self.n  # Selon l'article
        
        if initial_solution is None:
            S = [random.randint(0, self.n - 1)]
        else:
            S = initial_solution.copy()
        
        print("=== RECHERCHE LOCALE P-CENTER (Article PBS) ===")
        print(f"n={self.n}, p={self.p}, generation={generation}")
        print(f"Solution initiale: {S}")
        print(f"Max iterations: {max_iterations}")
        
        # Réinitialiser
        self.F0.clear()
        self.D0.clear()
        self.F1.clear()
        self.D1.clear()
        self.Sc = 0
        
        for f in S:
            self.add_facility(f)
        
        # PHASE 1 : Construction (if |S| < p)
        print(f"\n=== PHASE 1: CONSTRUCTION ===")
        while len(S) < self.p:
            print(f"\n[Construction] |S|={len(S)}, besoin de {self.p-len(S)} facilités supplémentaires")
            
            # Trouver w tel que D0[w] = Sc (sommet le plus mal desservi)
            worst_vertices = []
            for v in range(self.n):
                if v not in S and v in self.D0 and abs(self.D0[v] - self.Sc) < 1e-6:
                    worst_vertices.append(v)
            
            if worst_vertices:
                w = random.choice(worst_vertices)
                print(f"[Construction] Sommets avec D0=Sc={self.Sc:.2f}: {worst_vertices}")
                print(f"[Construction] w choisi: {w}")
            else:
                # Cas spécial: pas de sommet critique ou première itération
                candidates = [v for v in range(self.n) if v not in S and v in self.D0]
                if candidates:
                    w = max(candidates, key=lambda v: self.D0[v])
                    print(f"[Construction] Pas de sommet avec D0=Sc, w={w} avec D0={self.D0[w]:.2f}")
                else:
                    w = random.choice([v for v in range(self.n) if v not in S])
                    print(f"[Construction] Premier centre, w choisi aléatoirement: {w}")
            
            # Déterminer k selon l'article
            if w in self.F0:
                current_facility = self.F0[w]
                Nw = self.neighbors[w]
                print(f"    Nw (voisins de w): {Nw.tolist()}")
                k = np.where(Nw == current_facility)[0][0]
                print(f"[Construction] F0[{w}]={current_facility}, index k={k}")
            else:
                Nw = self.neighbors[w]
                print(f"    Nw (voisins de w): {Nw.tolist()}")
                k = 1
                print(f"[Construction] w={w} n'a pas de facilité, k=1")
            
            # CORRECTION: Nwk = les k premiers éléments (pas k+1)
            Nwk = self.neighbors[w][:k]
            print(f"[Construction] Nwk (les {k} premiers éléments): {Nwk.tolist()}")
            
            # Candidats dans Nwk non dans S
            candidates = [v for v in Nwk if v not in S]
            print(f"[Construction] Candidats disponibles: {candidates}")
            
            if not candidates:
                candidates = [v for v in range(self.n) if v not in S]
                print(f"[Construction] Aucun candidat dans Nwk, tous les non-utilisés: {candidates[:10]}...")
            
            # Sélection uniforme aléatoire (selon l'article)
            new_facility = random.choice(candidates)
            print(f"[Construction] Facilité choisie aléatoirement: {new_facility}")
            
            S.append(new_facility)
            self.add_facility(new_facility)
            print(f"[Construction] S après ajout: {S}, coût = {self.Sc:.2f}")
        
        # PHASE 2: Amélioration (swap phase)
        print(f"\n=== PHASE 2: AMÉLIORATION ===")
        print(f"Solution complète: S = {S}, coût initial = {self.Sc:.2f}")
        
        # Paramètres selon l'article
        max_no_improve = int(0.1 * (generation + 1) * self.n)  # CORRIGÉ: avec (g+1)
        print(f"[Paramètres] max_no_improve = 0.1 × ({generation}+1) × {self.n} = {max_no_improve}")
        
        self.forbidden_swaps.clear()
        iterations = 0
        no_improve = 0
        best_cost = self.Sc
        best_solution = S.copy()
        
        while iterations < max_iterations and no_improve < max_no_improve:
            print(f"\n--- Iteration {iterations} ---")
            print(f"S actuel: {S}, coût: {self.Sc:.2f}")
            print(f"no_improve: {no_improve}/{max_no_improve}")
            
            # Trouver les sommets critiques (D0[v] = Sc)
            critical_vertices = []
            for v in range(self.n):
                if v in self.D0 and abs(self.D0[v] - self.Sc) < 1e-6:
                    critical_vertices.append(v)
            
            if not critical_vertices:
                print("[STOP] Plus de sommets critiques")
                break
            
            print(f"[Amélioration] Sommets critiques (D0=Sc={self.Sc:.2f}): {critical_vertices}")
            
            # Choisir w aléatoirement parmi les critiques
            w = random.choice(critical_vertices)
            print(f"[Amélioration] w choisi: {w}")
            
            # Trouver la meilleure paire
            f_out, v_in, swap_attempts = self.find_pair(w, S, verbose=True)
            
            print(f"\n[Amélioration] Résumé des swaps testés:")
            for f, v, cost in swap_attempts:
                print(f"  Swap({f},{v}) => coût {cost:.2f}")
            
            if f_out is None:
                print("[Amélioration] Aucun swap trouvé")
                no_improve += 1
                iterations += 1
                continue
            
            if (f_out, v_in) in self.forbidden_swaps:
                print(f"[Amélioration] Swap({f_out},{v_in}) est interdit")
                no_improve += 1
                iterations += 1
                continue
            
            # Effectuer le swap
            print(f"\n[Amélioration] EFFECTUER SWAP: retirer {f_out}, ajouter {v_in}")
            S[S.index(f_out)] = v_in
            
            # Recalculer complètement les structures
            self.F0.clear()
            self.D0.clear()
            self.F1.clear()
            self.D1.clear()
            self.Sc = 0
            
            for f in S:
                self.add_facility(f)
            
            # Interdire ce swap
            self.forbidden_swaps.add((f_out, v_in))
            print(f"[Amélioration] Swap({f_out},{v_in}) ajouté aux swaps interdits")
            
            # Vérifier l'amélioration
            if self.Sc < best_cost:
                print(f"[Amélioration] ✓ AMÉLIORATION! {best_cost:.2f} -> {self.Sc:.2f}")
                best_cost = self.Sc
                best_solution = S.copy()
                no_improve = 0
            else:
                print(f"[Amélioration] ✗ Pas d'amélioration (coût {self.Sc:.2f})")
                no_improve += 1
            
            iterations += 1
        
        print(f"\n=== FIN DE LA RECHERCHE LOCALE ===")
        print(f"Terminé après {iterations} itérations")
        print(f"Raison d'arrêt: ", end="")
        if iterations >= max_iterations:
            print(f"limite d'itérations atteinte ({max_iterations})")
        elif no_improve >= max_no_improve:
            print(f"pas d'amélioration depuis {max_no_improve} itérations")
        else:
            print("plus de sommets critiques")
        
        print(f"Meilleure solution: {sorted(best_solution)}")
        print(f"Coût final: {best_cost:.2f}")
        
        return best_solution, best_cost