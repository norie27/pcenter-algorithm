import numpy as np
from typing import List

class Instance:
    def __init__(self, n: int, p: int, dist: List[List[float]]):
        self.n = n
        self.p = p
        self.dist = np.array(dist, dtype=np.float32)
        self.N = self._build_neighbors()

    def _build_neighbors(self) -> np.ndarray:
        neighbors = np.zeros((self.n, self.n), dtype=np.int32)
        for i in range(self.n):
            neighbors[i] = np.argsort(self.dist[i])
        
        return neighbors

