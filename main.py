import json
import pandas as pd
import os
import clustering
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import davies_bouldin_score, silhouette_score, adjusted_rand_score
import numpy as np

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    if config['format'] == 'csv':
        data = pd.read_csv(config['dataset'])
    elif config['format'] == 'arff':
        from scipy.io.arff import loadarff
        data = loadarff(config['dataset'])
        data = pd.DataFrame(data[0])
    else:
        raise ValueError("Niepoprawna wartość w polu 'format'. Dostępne wartości to: 'csv' lub 'arff'")
    
    data = data.dropna().reset_index(drop=True)
    data_cat = data.select_dtypes(include=object)

    if not data_cat.empty:
        encoder = OrdinalEncoder()
        data[data_cat.columns] = encoder.fit_transform(data_cat).astype(int)


    if config['target_feature'] is not None:
        y = data[config['target_feature']]
        data = data.drop(config['target_feature'], axis=1)
    
    if type(config['k']) != int:
        raise ValueError("Parametr 'k' powienien być liczbą całkowitą")
    
    if config['method'] == 'normal':
        nbc = clustering.NBC(method='normal')
    elif config['method'] == 'optimized':
        nbc = clustering.NBC(method='optimized')
    else:
        raise ValueError("Parametr 'method' powinien zawierać wartości 'normal' lub 'optimized'")
    
    nbc.fit(data)
    predicted = nbc.predict(data, config['k'])

    np.save('output.npy', predicted)
    
    if len(pd.unique(predicted)) >= 2:
        print('Ewaluacja wewnętrzna:')
        print('Metryka Davies-Bouldin: ', round(davies_bouldin_score(data, predicted), 4))
        print('Metryka Silhouette: ', round(silhouette_score(data, predicted), 4))

    if config['target_feature'] is not None:
        print('Ewaluacja zewnętrzna')
        print('Metryka Rand: ', round(adjusted_rand_score(y, predicted), 4))

    if not type(config['plot']) == bool:
        raise ValueError("Parametr 'plot' powinien zawierać wartość logiczną określającą, czy chcesz, aby wyświetlony był wykres")
    
    if config['plot']:
        import matplotlib.pyplot as plt
        if len(data.shape) == 2:
            plt.scatter(data[data.columns[0]], data[data.columns[1]], c=predicted)
        else:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(data)
            df_tsne = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2'])
            plt.scatter(df_tsne['Dim1'], df_tsne['Dim2'], c=predicted)
        plt.show()

if __name__ == '__main__':
    assert os.path.isfile('config.json')
    try:
        main()
    except Exception as e:
        print(e)