import math
from typing import List, Tuple
import numpy as np

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
            dist[i - 1][j - 1] = d
            dist[j - 1][i - 1] = d
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
    return n, p, dist

def create_instance(n: int, p: int, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crée une instance avec matrice de distances et voisins triés"""
    distances = np.array(distances, dtype=np.float64)
    neighbors = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        neighbors[i] = np.argsort(distances[i])
    return distances, neighbors


def read_instance(filepath: str) -> Tuple[int, int, np.ndarray]:
    """Lit une instance et retourne n, p, distances"""
    n, p, dist = read_graph_instance(filepath)
    return n, p, np.array(dist, dtype=np.float64)


def read_instance_auto(path: str) -> Tuple[int, int, List[List[float]]]:
    """
    Détecte automatiquement le format de l'instance : matrice ou graphe.
    """
    with open(path, 'r') as f:
        header = f.readline().strip()
        if not header:
            raise ValueError("Fichier vide.")
        fields = header.split()
        if len(fields) == 2:
            # Probablement matrice (n, p)
            return read_matrix_instance(path)
        elif len(fields) == 3:
            # Probablement graphe (n, m, p)
            return read_graph_instance(path)
        else:
            raise ValueError("Format de fichier inconnu.")

def read_matrix_instance(path: str) -> Tuple[int, int, List[List[float]]]:
    with open(path, 'r') as f:
        n, p = map(int, f.readline().strip().split())
        dist = [list(map(float, f.readline().strip().split())) for _ in range(n)]
    return n, p, dist
