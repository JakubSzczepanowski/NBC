import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import queue
import csv

class NBC:

    def __init__(self, k: int):
        self.k = k
        self.pesimistic_distance_cache = {}
        self.real_distance_cache = {}
        self.neighbours = None
        self.ndf = None

    def fit(self, X: pd.DataFrame, r: np.ndarray = None, l: int = 2):
        
        self.l = l
        n = len(X.index)
        kNN_counter = np.zeros(n)
        RkNN_counter = np.zeros(n)
        clst_no = np.full(n, -1)
        neighbours = []
        for index, row in X.iterrows():
            distances = [(i, self._real_cache_check_or_save(row, inner_row)) for i, inner_row in X.iterrows()]
            distances.sort(key=lambda x: x[1])
            indices = [index for index, _ in distances[:self.k]]

            # Sprawdzenie, czy istnieją inne punkty równej odległości
            equal_distance_indices = [index for index, dist in distances[self.k:] if dist == distances[self.k - 1][1]]
            indices.extend(equal_distance_indices)

            kNN_counter[index] = len(indices)
            for neighbour_index in indices:
                RkNN_counter[neighbour_index] += 1
            neighbours.append(indices)
        ndf = RkNN_counter/kNN_counter

        # ndf = np.load('ndf.npy')

        # data = ''
        # with open('neighbours.csv', 'r') as f:
        #     data = f.read()
        # csv_reader = csv.reader(data.splitlines())
        # neighbours = [list(map(int, row)) for row in csv_reader]

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

        # X['distance_with_r'] = 0.0
        # self.l = l

        # for index, row in X.iterrows():
        #     X.at[index, 'distance_with_r'] = self.distance(row, r, l)

        # X = X.sort_values(by='distance_with_r')

        # n = len(X.index)
        # for i in range(n):
        #     left_index = max(0, i - self.k)
        #     right_index = min(n, i + self.k + 1)

        #     target_series = X.iloc[i]
        #     indexed_values = [[j, X.iloc[j], 0] for j in range(left_index, right_index) if j != i]
        #     sorted_indexes = sorted(indexed_values, key=lambda x: self._pesimistic_cache_check_or_save(x[1], target_series))
        #     candidates = sorted_indexes[:self.k]
        #     eps = 0
        #     for index, c in enumerate(candidates):
        #         real_distance = self._real_cache_check_or_save(c[1], target_series)
        #         candidates[index][2] = real_distance
        #         if real_distance > eps:
        #             eps = real_distance
        #     inner_left_index = min(candidates, key=lambda x: x[0])[0] - 1
        #     if inner_left_index == i:
        #         inner_left_index -= 1
        #     inner_right_index = max(candidates, key=lambda x: x[0])[0] + 1
        #     if inner_right_index == i:
        #         inner_right_index -= 1
        #     up_run = True
        #     down_run = True
        #     while up_run or down_run:
        #         if inner_right_index < n:
        #             record = X.iloc[inner_right_index]
        #             greater_than_eps, new_eps = self._compare_with_epsilon(eps, record, target_series)
        #             if greater_than_eps:
        #                 down_run = False
        #             else:
        #                 eps = new_eps
        #                 candidates = [c for c in candidates if c[2] <= new_eps]
        #                 candidates.append((inner_right_index, record, new_eps))
        #             inner_right_index += 1
        #         elif down_run:
        #             down_run = False
        #         if inner_left_index >= 0:
        #             record = X.iloc[inner_left_index]
        #             greater_than_eps, new_eps = self._compare_with_epsilon(eps, record, target_series)
        #             if greater_than_eps:
        #                 up_run = False
        #             else:
        #                 candidates = [c for c in candidates if c[2] <= new_eps]
        #                 candidates.append((inner_right_index, record, new_eps))
        #                 eps = max(candidates, key=lambda x: x[2])
        #             inner_left_index -= 1
        #         elif up_run:
        #             up_run = False
        #     print(candidates)

    def _compare_with_epsilon(self, eps: float, current_series: pd.Series, target_series: pd.Series) -> tuple[bool, float]:

        greater_than_eps = False
        pesimistic_distance = self._pesimistic_cache_check_or_save(current_series, target_series)
        if pesimistic_distance >= eps:
            greater_than_eps = True
        else:
            real_distance = self._real_cache_check_or_save(current_series, target_series)
            if real_distance < eps:
                eps = real_distance

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