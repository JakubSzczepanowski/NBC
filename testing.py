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

from concurrent.futures import ProcessPoolExecutor

def compute_rand_value(args):
    X, y, model, k = args
    predictions = model.predict(X, k)
    return adjusted_rand_score(predictions, y)

def find_best_predictions_async(X: pd.DataFrame, y: np.ndarray, model: clustering.NBC, max_range=100, n_jobs=4) -> tuple[int, float]:

    k_values = np.arange(5, max_range + 1, step=5)

    # Używamy ProcessPoolExecutor do równoległego przetwarzania
    args_list = [(X, y, model, k) for k in k_values]
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        rand_values = list(executor.map(compute_rand_value, args_list))

    best_k = k_values[np.argmax(rand_values)]
    k_values = np.arange(best_k - 5, best_k + 6)

    # Ponowne obliczenia dla wąskiego zakresu
    args_list = [(X, y, model, k) for k in k_values]
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        rand_values = list(executor.map(compute_rand_value, args_list))

    plt.plot(k_values, rand_values)
    plt.xlabel('k')
    plt.ylabel('Adjusted Rand Index')
    plt.title('Adjusted Rand Index for Different k Values')
    plt.show()

    return k_values[np.argmax(rand_values)], np.max(rand_values)