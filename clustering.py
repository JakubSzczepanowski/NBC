import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

class NBC:

    def __init__(self, k: int):
        self.k = k
        self.pesimistic_distance_cache = {}
        self.real_distance_cache = {}

    def fit(self, X: pd.DataFrame, l: int = 2):
        X['distance_with_r'] = 0
        dim = len(X.columns)
        r = np.zeros(dim)

        for index, row in X.iterrows():
            X.at[index, 'distance_with_r'] = self.distance(row, r, l)

        X = X.sort_values(by='distance_with_r')

        n = len(X.index)
        for i in range(n):
            left_index = max(0, i - self.k)
            right_index = min(n, i + self.k + 1)

            target_series = X.iloc[i]
            target_value = target_series['distance_with_r']
            indexed_values = [(X.iloc[j].name, X.iloc[j]) for j in range(left_index, right_index) if j != i]
            sorted_indexes = sorted(indexed_values, key=lambda x: self._pesimistic_cache_check_or_save(x[1]['distance_with_r'], target_value))
            candidates = sorted_indexes[:self.k]
            eps = 0
            for c in candidates:
                real_distance = self._real_cache_check_or_save(c, (target_series.name, target_series))
                if real_distance > eps:
                    eps = real_distance
            


    def kneighbors(self):
        pass
        
    def distance(self, p: ArrayLike, q: ArrayLike, l: int) -> float:
        return np.power(np.sum([np.abs(i-j)**l for i, j in zip(p, q)]), 1/l)
    
    def _pesimistic_cache_check_or_save(self, compare_value: np.float64, target_value: np.float64) -> float:
        
        if (compare_value, target_value) in self.pesimistic_distance_cache:
            return self.pesimistic_distance_cache[(compare_value, target_value)]
        if (target_value, compare_value) in self.pesimistic_distance_cache:
            return self.pesimistic_distance_cache[(target_value, compare_value)]
        
        distance = np.abs(compare_value - target_value)

        self.pesimistic_distance_cache[(compare_value, target_value)] = distance

        return distance
    
    def _real_cache_check_or_save(self, obj1: tuple[int, pd.Series], obj2: tuple[int, pd.Series]) -> float:
        
        if (obj1[0], obj2[0]) in self.real_distance_cache:
            return self.real_distance_cache[(obj1[0], obj2[0])]
        if (obj2[0], obj1[0]) in self.real_distance_cache:
            return self.real_distance_cache[(obj2[0], obj1[0])]
        
        distance = self.distance(obj1[1].drop('distance_with_r'), obj2[1].drop('distance_with_r'))

        self.real_distance_cache[(obj1[0], obj2[0])] = distance

        return distance