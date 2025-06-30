# src/algorithms/genetic_operators.py

from typing import List, Tuple
import random
import numpy as np
from src.utils.instance import Instance


class GeneticOperators:
    """Opérateurs génétiques M1, M2, X1, X2 selon l'article PBS"""
    
    def __init__(self, instance: Instance):
        self.instance = instance
        self.n = instance.n
        self.p = instance.p
        self.distances = instance.dist
    
    def mutation_m1(self, solution: List[int]) -> List[int]:
        """
        Mutation M1 - Random mutation operator
        Selon l'article (page 420, section 2.1):
        "M1, is a random mutation operator that takes a p-center solution Pi 
        and constructs a child solution S by first randomly selecting some 
        number q in the range p/2,...,p and then using q randomly selected 
        facilities from Pi and p − q randomly selecting vertices from V − Pi 
        to construct S."
        """
        # Sélectionner q dans [p/2, p]
        q = random.randint(self.p // 2, self.p)
        
        # Garder q facilités aléatoires de la solution
        kept_facilities = random.sample(solution, q)
        
        # Ajouter p-q nouvelles facilités aléatoires
        available = [v for v in range(self.n) if v not in kept_facilities]
        new_facilities = random.sample(available, self.p - q)
        
        child = kept_facilities + new_facilities
        
        print(f"[M1] q={q}, gardé {len(kept_facilities)} facilités, ajouté {len(new_facilities)} nouvelles")
        
        return child
    
    def mutation_m2(self, solution: List[int]) -> List[int]:
        """
        Mutation M2 - Directed mutation operator
        Selon l'article (page 420, section 2.1):
        "M2 is a directed mutation operator that, based on the assumption 
        that facilities should not be "close" to each other, identifies the 
        closest pair of facilities within Pi, deletes them from Pi"
        """
        if len(solution) < 2:
            return solution.copy()
        
        # Trouver la paire de facilités la plus proche
        min_dist = float('inf')
        closest_pair = (0, 1)
        
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                dist = self.distances[solution[i]][solution[j]]
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (i, j)
        
        # Retirer ces deux facilités
        child = []
        for i, facility in enumerate(solution):
            if i != closest_pair[0] and i != closest_pair[1]:
                child.append(facility)
        
        print(f"[M2] Retiré facilités {solution[closest_pair[0]]} et {solution[closest_pair[1]]} (distance={min_dist:.2f})")
        print(f"[M2] Solution réduite de {len(solution)} à {len(child)} facilités")
        
        return child
    
    def crossover_x1(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Crossover X1 - Random crossover operator
        Selon l'article (page 420, section 2.2):
        "X1, a random crossover operator which constructs a single child 
        solution by randomly selecting p facilities from the two parents"
        """
        # Combiner toutes les facilités des deux parents
        all_facilities = list(set(parent1 + parent2))
        
        # Sélectionner aléatoirement p facilités
        if len(all_facilities) >= self.p:
            child = random.sample(all_facilities, self.p)
        else:
            # Compléter avec des facilités aléatoires si nécessaire
            child = all_facilities.copy()
            available = [v for v in range(self.n) if v not in all_facilities]
            needed = self.p - len(all_facilities)
            if needed > 0 and available:
                additional = random.sample(available, min(needed, len(available)))
                child.extend(additional)
        
        print(f"[X1] Parents: {len(parent1)} + {len(parent2)} facilités, union: {len(all_facilities)}, enfant: {len(child)}")
        
        return child
    
    def crossover_x2(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Crossover X2 - Phenotype crossover operator
        Selon l'article (page 420, section 2.2):
        "X2, which first randomly selects two users and a random number 
        q ∈ [0.1,...,0.9]. Let di1 be the distance between facility i and 
        the first selected user and di2 the distance between facility i and 
        the second selected user..."
        """
        # Sélectionner deux utilisateurs aléatoires
        user1 = random.randint(0, self.n - 1)
        user2 = random.randint(0, self.n - 1)
        while user2 == user1:  # S'assurer qu'ils sont différents
            user2 = random.randint(0, self.n - 1)
        
        # Paramètre q aléatoire
        q = random.uniform(0.1, 0.9)
        
        print(f"[X2] Users: {user1}, {user2}, q={q:.2f}")
        
        child1_facilities = []
        child2_facilities = []
        
        # Pour chaque facilité du parent 1
        for f in parent1:
            d1 = self.distances[f][user1]
            d2 = self.distances[f][user2]
            
            if d2 > 0:  # Éviter division par zéro
                ratio = d1 / d2
                if ratio <= q:
                    child1_facilities.append(f)
                else:
                    child2_facilities.append(f)
            else:
                # Si d2 = 0, mettre dans child1
                child1_facilities.append(f)
        
        # Pour chaque facilité du parent 2
        for f in parent2:
            if f not in parent1:  # Éviter les doublons
                d1 = self.distances[f][user1]
                d2 = self.distances[f][user2]
                
                if d2 > 0:
                    ratio = d1 / d2
                    if ratio > q:
                        child1_facilities.append(f)
                    else:
                        child2_facilities.append(f)
                else:
                    child2_facilities.append(f)
        
        # Ajuster à exactement p facilités
        child1 = self._adjust_to_p(child1_facilities)
        child2 = self._adjust_to_p(child2_facilities)
        
        print(f"[X2] Enfant1: {len(child1)} facilités, Enfant2: {len(child2)} facilités")
        
        return child1, child2
    
    def _adjust_to_p(self, facilities: List[int]) -> List[int]:
        """
        Ajuste une liste de facilités à exactement p éléments
        Selon l'article: "if more than p facilities are added to a child 
        solution then the required number are randomly removed to reduce 
        the total down to p"
        """
        facilities = list(set(facilities))  # Supprimer les doublons
        
        if len(facilities) > self.p:
            # Retirer aléatoirement l'excès
            return random.sample(facilities, self.p)
        elif len(facilities) < self.p:
            # Ajouter aléatoirement ce qui manque
            available = [v for v in range(self.n) if v not in facilities]
            needed = self.p - len(facilities)
            if available and needed > 0:
                additional = random.sample(available, min(needed, len(available)))
                return facilities + additional
            return facilities
        else:
            return facilities