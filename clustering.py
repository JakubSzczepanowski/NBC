import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

class NBC:

    def __init__(self, k: int):
        self.k = k
        self.pesimistic_distance_cache = {}
        self.real_distance_cache = {}

    def fit(self, X: pd.DataFrame, r: np.ndarray, l: int = 2):
        X['distance_with_r'] = 0.0
        self.l = l

        for index, row in X.iterrows():
            X.at[index, 'distance_with_r'] = self.distance(row, r, l)

        X = X.sort_values(by='distance_with_r')

        n = len(X.index)
        for i in range(n):
            left_index = max(0, i - self.k)
            right_index = min(n, i + self.k + 1)

            target_series = X.iloc[i]
            indexed_values = [[j, X.iloc[j], 0] for j in range(left_index, right_index) if j != i]
            sorted_indexes = sorted(indexed_values, key=lambda x: self._pesimistic_cache_check_or_save(x[1], target_series))
            candidates = sorted_indexes[:self.k]
            eps = 0
            for index, c in enumerate(candidates):
                real_distance = self._real_cache_check_or_save(c[1], target_series)
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
                if inner_right_index < n:
                    record = X.iloc[inner_right_index]
                    greater_than_eps, new_eps = self._compare_with_epsilon(eps, record, target_series)
                    if greater_than_eps:
                        down_run = False
                    else:
                        eps = new_eps
                        candidates = [c for c in candidates if c[2] <= new_eps]
                        candidates.append((inner_right_index, record, new_eps))
                    inner_right_index += 1
                elif down_run:
                    down_run = False
                if inner_left_index >= 0:
                    record = X.iloc[inner_left_index]
                    greater_than_eps, new_eps = self._compare_with_epsilon(eps, record, target_series)
                    if greater_than_eps:
                        up_run = False
                    else:
                        candidates = [c for c in candidates if c[2] <= new_eps]
                        candidates.append((inner_right_index, record, new_eps))
                        eps = max(candidates, key=lambda x: x[2])
                    inner_left_index -= 1
                elif up_run:
                    up_run = False
            print(candidates)

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
        
        distance = self.distance(obj1.drop('distance_with_r'), obj2.drop('distance_with_r'), self.l)

        self.real_distance_cache[(obj1.name, obj2.name)] = distance

        return distance