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
            indices = self._find_neighbors(i, k, X)

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

        
    
    def _find_neighbors(self, current_element: int, k: int, X: pd.DataFrame):
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

        elif self.method == 'heuristic':

            for i in range(self.n):
                left_index = max(0, i - k)
                right_index = min(self.n, i + k + 1)

                target_series = X.iloc[i]
                indexed_values = [[j, X.iloc[j], 0] for j in range(left_index, right_index) if j != i]
                sorted_indexes = sorted(indexed_values, key=lambda x: np.abs(x[1]['distance_with_r'] - target_series['distance_with_r']))
                candidates = sorted_indexes[:k]
                eps = 0
                for index, c in enumerate(candidates):
                    real_distance = self.distance(c[1].drop('distance_with_r', errors='ignore'), target_series.drop('distance_with_r', errors='ignore'), self.l)
                    candidates[index][2] = real_distance
                    if real_distance > eps:
                        eps = real_distance
                inner_left_index = min(candidates, key=lambda x: x[0])[0] - 1
                if inner_left_index == i:
                    inner_left_index -= 1
                inner_right_index = max(candidates, key=lambda x: x[0])[0] + 1
                if inner_right_index == i:
                    inner_right_index -= 1
                up_run = True
                down_run = True
                while up_run or down_run:
                    if inner_right_index < self.n and down_run:
                        record = X.iloc[inner_right_index]
                        pesimistic_distance = np.abs(record['distance_with_r'] - target_series['distance_with_r'])
                        if pesimistic_distance >= eps:
                            down_run = False
                        else:
                            real_distance = self.distance(record.drop('distance_with_r', errors='ignore'), target_series.drop('distance_with_r', errors='ignore'), self.l)
                            if real_distance < eps:
                                same_size_elements = [elem[0] for elem in candidates if elem[2] == eps]
                                if len(candidates) - len(same_size_elements) >= k - 1:
                                    candidates = [elem for elem in candidates if elem[0] not in same_size_elements]
                                    candidates.append((inner_right_index, record, real_distance))
                                    eps = max(candidates, key=lambda x: x[2])[2]
                                else:
                                    candidates.append((inner_right_index, record, real_distance))
                            elif real_distance == eps:
                                candidates.append((inner_right_index, record, real_distance))
                        inner_right_index += 1

                    if inner_left_index >= 0 and up_run:
                        record = X.iloc[inner_left_index]
                        pesimistic_distance = np.abs(record['distance_with_r'] - target_series['distance_with_r'])
                        if pesimistic_distance >= eps:
                            down_run = False
                        else:
                            real_distance = self.distance(record.drop('distance_with_r', errors='ignore'), target_series.drop('distance_with_r', errors='ignore'), self.l)
                            if real_distance < eps:
                                same_size_elements = (elem[0] for elem in candidates if elem[2] == eps)
                                if len(candidates) - len(same_size_elements) >= k - 1:
                                    candidates = [elem for elem in candidates if elem[0] not in same_size_elements]
                                    candidates.append((inner_right_index, record, real_distance))
                                    eps = max(candidates, key=lambda x: x[2])[2]
                                else:
                                    candidates.append((inner_right_index, record, real_distance))
                            elif real_distance == eps:
                                candidates.append((inner_right_index, record, real_distance))
                        inner_left_index -= 1
                print(candidates)
        return indices

    def _compare_with_epsilon(self, eps: float, current_series: pd.Series, target_series: pd.Series) -> tuple[bool, float]:

        greater_than_eps = False
        

        return (greater_than_eps, eps)


    def kneighbors(self):
        pass
        
    def distance(self, p: ArrayLike, q: ArrayLike, l: int) -> float:
        return np.power(np.sum([np.abs(i-j)**l for i, j in zip(p, q)]), 1/l)
    
    def _pesimistic_cache_check_or_save(self, obj1: pd.Series, obj2: pd.Series) -> float:
        
        if (obj1.name, obj2.name) in self.pesimistic_distance_cache:
            return self.pesimistic_distance_cache[(obj1.name, obj2.name)]
        if (obj2.name, obj1.name) in self.pesimistic_distance_cache:
            return self.pesimistic_distance_cache[(obj2.name, obj1.name)]
        
        distance = np.abs(obj1['distance_with_r'] - obj2['distance_with_r'])

        self.pesimistic_distance_cache[(obj1.name, obj2.name)] = distance

        return distance
    
    def _real_cache_check_or_save(self, obj1: pd.Series, obj2: pd.Series) -> float:
        
        if (obj1.name, obj2.name) in self.real_distance_cache:
            return self.real_distance_cache[(obj1.name, obj2.name)]
        if (obj2.name, obj1.name) in self.real_distance_cache:
            return self.real_distance_cache[(obj2.name, obj1.name)]
        
        distance = self.distance(obj1.drop('distance_with_r', errors='ignore'), obj2.drop('distance_with_r', errors='ignore'), self.l)

        self.real_distance_cache[(obj1.name, obj2.name)] = distance

        return distance