from typing import Literal
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import queue
from scipy.spatial.distance import cdist

class NBC:

    def __init__(self, l: int = 2, method: Literal['normal', 'optimized', 'heuristic'] = 'normal'):
        self.l = l
        self.method = method

    def fit(self, X: pd.DataFrame):
        self.n = len(X.index)
        if self.method == 'normal':
            self.distances = [
                [
                    (i, self.distance(row, inner_row, self.l))
                    for i, inner_row in X.iterrows()
                ]
                for _, row in X.iterrows()
            ]
        elif self.method == 'optimized':
            self.distances = cdist(X, X, 'minkowski', p=self.l)
        elif self.method == 'heuristic':
            X['distance_with_r'] = 0.0
            r = X.min()

            for index, row in X.iterrows():
                X.at[index, 'distance_with_r'] = self.distance(row, r, self.l)

            X = X.sort_values(by='distance_with_r')

    def predict(self, X: pd.DataFrame, k: int):
        
        n = len(X.index)
        kNN_counter = np.zeros(n, dtype=np.int32)
        RkNN_counter = np.zeros(n, dtype=np.int32)
        clst_no = np.full(n, -1)
        neighbours = []
        for i in range(n):
            indices = self._find_neighbors(i, k)

            counter = len(indices)
            kNN_counter[i] = np.finfo(float).eps if counter == 0 else counter
            for neighbour_index in indices:
                RkNN_counter[neighbour_index] += 1
            neighbours.append(indices)
        ndf = RkNN_counter//kNN_counter

        cluster_count = 0
        DPSet = queue.Queue()
        
        for p in range(n):
            if clst_no[p] != -1 or ndf[p] < 1: continue
            clst_no[p] = cluster_count
            DPSet.queue.clear()
            for q in neighbours[p]:
                clst_no[q] = cluster_count
                if ndf[q] >= 1: DPSet.put(q)
            
            while not DPSet.empty():
                point = DPSet.get()
                for q in neighbours[point]:
                    if clst_no[q] != -1: continue
                    clst_no[q] = cluster_count
                    if ndf[q] >= 1: DPSet.put(q)
            cluster_count += 1
        return clst_no

        
    
    def _find_neighbors(self, current_element: int, k: int):
        indices = None

        if self.method == 'normal':
            row_distances = self.distances[current_element]
            row_distances.sort(key=lambda x: x[1])
            indices = [index for index, _ in row_distances[1:k+1]]

            # Sprawdzenie, czy istnieją inne punkty równej odległości
            equal_distance_indices = [index for index, dist in row_distances[k+1:] if dist == row_distances[k][1]]
            indices.extend(equal_distance_indices)

        elif self.method == 'optimized':
            indices = np.argsort(self.distances[current_element])[1:k+1]
            if len(indices) == 0:
                return indices
            max_distance = self.distances[current_element, indices[-1]]

            # Sprawdzenie, czy istnieją inne punkty równej odległości
            equal_distance_mask = (self.distances[current_element] == max_distance) & ~np.isin(np.arange(self.n), indices)

            # Dodaj indeksy o równej odległości do listy
            equal_distance_indices = np.where(equal_distance_mask)[0]
            indices = np.concatenate((indices, equal_distance_indices))

        return indices
        
    def distance(self, p: ArrayLike, q: ArrayLike, l: int) -> float:
        return np.power(np.sum([np.abs(i-j)**l for i, j in zip(p, q)]), 1/l)