from sklearn.metrics.cluster import adjusted_rand_score
import clustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_best_predictions(X: pd.DataFrame, y: np.ndarray, model: clustering.NBC, max_range=100) -> tuple[int, float]:
    k_values = np.arange(5, max_range+1, step=5)
    rand_values = []
    for k in k_values:
        rand_values.append(adjusted_rand_score(model.predict(X, k), y))
    best_k = k_values[np.argmax(rand_values)]
    k_values = np.arange(best_k-5, best_k+6)
    rand_values.clear()
    for k in k_values:
        rand_values.append(adjusted_rand_score(model.predict(X, k), y))
    plt.plot(k_values, rand_values)
    return k_values[np.argmax(rand_values)], np.max(rand_values)