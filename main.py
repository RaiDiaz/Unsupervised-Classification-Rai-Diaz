import pandas as pd
import numpy as np
import os
from openpyxl import Workbook
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import seaborn as sns
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from CURE import *


# Funcion para guardar resultados en archivo csv o xlsx
def write_to_csv(data, name):
    pd_data = pd.DataFrame.from_dict(data)

    pd_data.to_csv(f'Result_csv/{name}.csv')
    pd_data.to_excel(f'Result_xlx/{name}.xlsx')


# Funcion para leer los algoritmos de la carpeta, separando el numero de clusters seleccionados
def read_algorithm(numAlgorithm, numClusters):
    print(
        f'\033[32m -- Analizando ({numAlgorithm}) ALGORITMO(S) con ({numClusters}) CLUSTER(S) para encontrar el mejor K \033[039m\n')
    # Se obtiene los resultados de la carpeta de todos los algoritmos para luego poder seleccionarlos
    list_files = os.listdir('Result_csv')
    list_files = np.array(list_files)

    selected_algorithms = list()
    for x in range(int(numAlgorithm)):
        selected_algorithms.append(list_files[x].split('_')[0])  # Se agrega a la lista los algoritmos ha analizar
    print(f'\033[32m Algoritmos ah analizar \033[039m')
    print(selected_algorithms)

    selected_clusters = list()
    for x in range(int(numClusters)):
        selected_clusters.append(f'K{x + 2}')  # Se agrega a la lista los K ha analizar
    print(f'\n\033[32m Clusters ah analizar \033[039m')
    print(selected_clusters)
    print('\n\033[32m*************** Resultado de los Datasets a analizar ***************\033[039m\n')

    algorithms_analyze = dict()
    for x in range(int(numAlgorithm)):
        print(list_files[x])
        df_analyze = pd.read_csv(
            f'Result_csv/{list_files[x]}')  # Se lee el resultado del algoritmo clusterizado para su analisis
        df_analyze = df_analyze[selected_clusters]  # Se selecciona las columnas ha analizar
        df_analyze.head()
        print(df_analyze)
        algorithms_analyze[list_files[x].split('_')[0]] = df_analyze
    return algorithms_analyze, selected_clusters, selected_algorithms


# Funcion que separa los grupos de cada cluster para analizarlos
def separated_cluster_per_id():
    algorithms_analyze_cluster = dict()
    for k in algorithms_analyze:
        print(
            f'\033[32m**************************** Separando clusters {k} en grupos ****************************\033[039m\n')
        num = 1
        for i in range(int(numClusters)):
            df_analyze_cluster = algorithms_analyze[k][
                selected_clusters[i]]  # se separa cada cluster de la matriz del algoritmo
            df_analyze_cluster.head()
            numCluster = 0
            while numCluster <= num:
                print(f'\033[32m -- {k}-{selected_clusters[i]}-{numCluster}\033[039m\n')
                df_to_analyze = df_analyze_cluster[df_analyze_cluster == numCluster]
                algorithms_analyze_cluster[
                    f'{k}-{selected_clusters[i]}-{numCluster}'] = df_to_analyze  # Se separa cada grupo del cluster y se guarda en un diccionario
                print(algorithms_analyze_cluster[f'{k}-{selected_clusters[i]}-{numCluster}'].to_numpy())
                numCluster = numCluster + 1
            num = num + 1
    return algorithms_analyze_cluster

# Funcion que analiza todos los grupos de cada cluster para encontrar la interseccion
def filter_cluster_per_group(algorithms_analyze_cluster, Kn):
    print(
        f'\033[32m***************************** Lista de algoritmos de clusters K{Kn} ***************************\033[039m\n')
    df_filtered = dict()
    name_filtered = list()
    # se realiza una busqueda de cada algoritmo y de cada grupo de cluster para su analisis ej. Kmean-0,...
    for i in algorithms_analyze_cluster:
        if f'K{Kn}' in algorithms_analyze_cluster[i].name:
            df_filtered[i] = algorithms_analyze_cluster[i]
            name_filtered.append(i)
            index_matrix.append(i)
            print(i)
    count = 0
    df_length_filtered = len(df_filtered) - Kn
    # Se analiza el grupo seleccionado con los demas grupos de la lista previa
    for i in df_filtered:
        if count < df_length_filtered:
            print(f'\033[32m************************** {i} ANALIZANDO... ******************************\033[039m\n')
            df_initial_analyze = list(df_filtered.values())[count]
            print(df_initial_analyze)
            result_per_cluster_analized(count, df_filtered, df_initial_analyze, Kn, name_filtered, i)
        count = count + 1

# muesta el resultado de la interseccion de cada algoritmo analizado con cada grupo de cluster
def result_per_cluster_analized(count, df_cluster, df_initial_analyze, Kn, name_filtered, file_name):
    df_final_analyze = dict()
    df_from_analyze = pd.Index(df_initial_analyze.index)
    c = switch(count, Kn)
    sum_per_cluster = 0
    print(f'\033[32m --- {file_name} TOTAL -> {len(df_from_analyze)} \033[039m\n')
    # bucle que analiza el grupo seleccionado con los grupos de la lista de los clusters analizar
    while c < len(df_cluster):
        df_to_analyze = pd.Index(df_cluster[name_filtered[c]].index)
        df_final_analyze[name_filtered[c]] = df_from_analyze.intersection(df_to_analyze)
        print(f'\033[32m --- {file_name} analizando con: {name_filtered[c]} \033[039m\n')
        print(f'{len(df_final_analyze[name_filtered[c]])}/{len(df_from_analyze)}\n')
        if len(df_from_analyze) == 0:
            num = 0
        else:
            num = len(df_final_analyze[name_filtered[c]]) / len(df_from_analyze)
        dec_num = num * 100
        print(f'\033[32m => {round(dec_num, 2)} % \033[039m\n')

        if name_filtered[c] not in dict_values_cluster.keys():
            dict_values_cluster[name_filtered[c]] = [round(dec_num, 2)]
        else:
            dict_values_cluster[name_filtered[c]].append(round(dec_num, 2))
        list_matrix_values.append(round(dec_num, 2))
        sum_per_cluster = sum_per_cluster + len(df_final_analyze[name_filtered[c]])
        c = c + 1

    pd_data = pd.DataFrame.from_dict(df_final_analyze, orient='index')
    pd_data = pd_data.transpose()
    pd_data.to_excel(f'Result_Intersection/Result_{file_name}.xlsx')

# Funcion que completa la matriz del K analizado, para poder usarla en un futuro
def complete_matrix(df_original, df_transpose, Kn):
    num_final = Kn * int(numAlgorithm)
    num_initial = num_final - Kn

    for i in range(num_initial, num_final):
        df_original[i] = 0

    dict_values = dict()
    for i in range(Kn):
        df_transpose.insert(loc=i, column=index_matrix[i], value=0)

    for i in range(len(index_matrix)):
        if i < len(df_transpose):
            dict_values[index_matrix[i]] = df_transpose.iloc[i].to_numpy()
        else:
            dict_values[index_matrix[i]] = df_original.loc[index_matrix[i]].to_numpy()

    df_complete_matrix = add_values_to_dataframe(dict_values)

    df_transpose = df_complete_matrix.transpose()

    c = switch_matrix(int(numAlgorithm), Kn)
    for i in range(c):
        df_complete_matrix.loc[index_matrix[i]] = df_transpose.iloc[i].to_numpy()

    return df_complete_matrix

# funcion que analiza los resltados de la matriz completa
def matrix_max_average(Kn, average):
    max_average = dict()
    for i in range(len(index_matrix)):
        if i < len(df_transpose):
            max_average[index_matrix[i]] = df_transpose.iloc[i].to_numpy()
        else:
            max_average[index_matrix[i]] = df_original.loc[index_matrix[i]].to_numpy()

    dict_average = dict()
    dict_average = set_filter_by_average(max_average, average)

    df_temp_matrix = add_values_to_dataframe(dict_average)

    df_temp_transpose = df_temp_matrix.transpose()
    c = switch_matrix(int(numAlgorithm), Kn)
    for i in range(c):
        df_temp_matrix.loc[index_matrix[i]] = df_temp_transpose.iloc[i].to_numpy()

    # Se completa la matriz con los valores correspondientes del analisis (Se realiza por columna)
    pair_sum_col = 0
    num_alg_col = 0
    num_range_col = Kn
    df_temp = pd.DataFrame()
    while pair_sum_col < len(df_temp_matrix):
        col_list = list()
        for i in range(pair_sum_col, num_range_col, 1):
            col_list.append(df_temp_matrix[index_matrix[i]].name)
        df_temp[selected_algorithms[num_alg_col]] = df_temp_matrix[col_list].sum(axis=1)
        pair_sum_col = pair_sum_col + Kn
        num_range_col = num_range_col + Kn
        num_alg_col = num_alg_col + 1

    # Se completa la matriz con los valores correspondientes del analisis (Se realiza por fila)
    pair_sum_row = 0
    num_alg_row = 0
    num_range_row = Kn
    df_temp_row = pd.DataFrame(columns=selected_algorithms)
    while pair_sum_row < len(df_temp_matrix):
        row_list = list()
        for i in range(pair_sum_row, num_range_row, 1):
            row_list.append(df_temp_matrix[index_matrix[i]].name)
        df_temp_row.loc[selected_algorithms[num_alg_row]] = df_temp.loc[row_list].sum()
        pair_sum_row = pair_sum_row + Kn
        num_range_row = num_range_row + Kn
        num_alg_row = num_alg_row + 1

    return df_temp_row

# Funcion que cuenta los resultados de la matriz mayores al value (average) ingresado
def set_filter_by_average(max_average, value):
    _dic_values = dict()
    for i in max_average:
        for j in max_average[i]:
            if value < j:
                count = 1
            else:
                count = 0
            if i not in _dic_values.keys():
                _dic_values[i] = [count]
            else:
                _dic_values[i].append(count)
    return _dic_values

# Funcion que realiza la suma de cada matriz analizada de cada grupo de cluster
def average_per_column(df_a):
    df_a['Average Freq'] = df_a.sum(axis=1)
    df_a.loc['Total'] = pd.Series(df_a['Average Freq'].sum(), index=['Average Freq'])

    return df_a


def avergae_per_Kn_algorithm(df_m, Kn):
    Kn_average_algo[f'K{Kn}'] = df_m['Average Freq'] / Kn

# Funcion que controla los resultados de los graficos
def graph_tsne(best_k):
    # Se inicia el TSNE con los valores del dataset normalizado
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto')
    tsne_result = tsne.fit_transform(df_normalized)
    plt.title('T-SNE sin Clusterización')
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='purple', marker='x', s=20)
    plt.savefig(f'IMG/Graph_Original.png')
    plt.show()

    n = 0
    # Se inicia el subplot de acuerdo al numero de algoritmos analizados
    fig, ax = plt.subplots(int(numAlgorithm), 3, figsize=(20, 40), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.01)

    for a in selected_algorithms:
        cluster_result = pd.read_csv(f'Result_csv/{a}_result.csv')
        plot_tsne_2d(best_k, cluster_result, a, n, tsne_result, ax, fig)
        n = n + 1
    # plt.tight_layout()
    plt.savefig(f'IMG/Graph_Result.png')
    plt.show()

# Funcion para grafica los cluster k, K-1, K+1
def plot_tsne_2d(k: int, cluster_result, a, n, tsne_result, ax, fig):
    fig.suptitle("Graficos TSNE", fontsize=20)
    ax[0, 0].set_title(f'k={k-1}', fontsize=18)
    ax[0, 1].set_title(f'k={k}', fontsize=18)
    ax[0, 2].set_title(f'k={k+1}', fontsize=18)

    ax[n, 0].set_ylabel(f'{a}', fontsize=16)
    plot_tsne_for_model_2d(tsne_result, cluster_result[f'K{k - 1}'], ax[n, 0], a, k-1)
    plot_tsne_for_model_2d(tsne_result, cluster_result[f'K{k}'], ax[n, 1], a, k)
    plot_tsne_for_model_2d(tsne_result, cluster_result[f'K{k}'], ax[n, 2], a, k=k+1)

# Funcion para graficar los scatter (Tsne) de cada grafico
def plot_tsne_for_model_2d(tsne_result: np.array, y_pred, ax: plt.Axes, name, k: any):
    n = 0
    for i, c in enumerate(np.unique(y_pred)):
        # Condicion para quitar grupos de cluster no analizados
        if i > (k-1):
            exit
        # Condicion que analiza DBSCAN y OPTICS ya que poseen resultados con ruido (noise)
        elif name == 'DBSCAN' or name == 'OPTICS':
            cols = switch_pltcolor(y_pred[y_pred == n])
            markers = switch_pltmarker(i)
            ax.scatter(tsne_result[:, 0][y_pred == n], tsne_result[:, 1][y_pred == n], c=cols, marker=markers, s=20,
                       label=n)
            n = n + 1
        else:
            cols = switch_pltcolor(y_pred[y_pred == c])
            markers = switch_pltmarker(i)
            ax.scatter(tsne_result[:, 0][y_pred == c], tsne_result[:, 1][y_pred == c], c=cols, marker=markers, s=20,
                       label=c)

    ax.legend(loc="lower right")

# funcion que anade los valores correspondientes a la matriz
def add_values_to_dataframe(_df_dict):
    _df_matrix = pd.DataFrame(columns=index_matrix, index=index_matrix)
    for k, v in _df_dict.items():
        _df_matrix[k] = v
    return _df_matrix

# Funcion switch que retorna la marca del grupo del cluster a mostrar
def switch_pltmarker(_m):
    if _m == 0:
        return '.'
    elif _m == 1:
        return 'x'
    elif _m == 2:
        return '^'
    elif _m == 3:
        return '*'
    else:
        return '+'

# Funcion switch que devuelve una lista de colores dependiendo del grupo a mostrar
def switch_pltcolor(lst):
    cols = []
    for l in lst:
        if l == 0:
            cols.append('red')
        elif l == 1:
            cols.append('blue')
        elif l == 2:
            cols.append('yellow')
        elif l == 3:
            cols.append('green')
        else:
            cols.append('purple')
    return cols

# funcion switch para poder completa la matrix dependiendo del cluster que se encuentre
def switch_matrix(num, Kn):
    if num == 3:
        return Kn * 1
    elif num == 4:
        return Kn * 2
    elif num == 5:
        return Kn * 3
    elif num == 6:
        return Kn * 4
    elif num == 6:
        return Kn * 5
    elif num == 7:
        return Kn * 6
    elif num == 8:
        return Kn * 7
    elif num == 9:
        return Kn * 8
    else:
        return Kn

# funcion switch que realiza el analisis de los cluster por grupos, para poder analizar el grupo correspondiente y no repetir
def switch(num, Kn):
    if (Kn - 1) < num < (Kn * 2):
        return Kn * 2
    elif ((Kn * 2) - 1) < num < (Kn * 3):
        return Kn * 3
    elif ((Kn * 3) - 1) < num < (Kn * 4):
        return Kn * 4
    elif ((Kn * 4) - 1) < num < (Kn * 5):
        return Kn * 5
    elif ((Kn * 5) - 1) < num < (Kn * 6):
        return Kn * 6
    elif ((Kn * 6) - 1) < num < (Kn * 7):
        return Kn * 7
    elif ((Kn * 7) - 1) < num < (Kn * 8):
        return Kn * 8
    elif ((Kn * 8) - 1) < num < (Kn * 9):
        return Kn * 9
    elif ((Kn * 9) - 1) < num < (Kn * 10):
        return Kn * 10
    elif ((Kn * 9) - 1) < num < (Kn * 10):
        return Kn * 10
    elif ((Kn * 10) - 1) < num < (Kn * 11):
        return Kn * 11
    else:
        return Kn


if __name__ == '__main__':
    # Path del archivo a leer, si se quiere analizar otro dataset, se debe cambiar la direccion
    FILE_NAME_PATH = 'files/CSV_ETS295_class_smote_5_100.csv'
    # Lectura del archivo csv.
    df_original = pd.read_csv(FILE_NAME_PATH)
    # print('\033[32m****** Dataset sin normalizar ******\033[039m\n')
    # print(df_original.to_string())
    # Normalización de los datos.
    # print('\033[32m****** Dataset normalizado ******\033[039m\n')
    df_normalized = (df_original - df_original.min()) / (df_original.max() - df_original.min())
    # print(df_normalized.round(3).to_string())
    df_normalized.to_csv('files/dataset_normalized.csv')

    # Se extrae los valores del dataset original y del normalizado para poder realizar la clusterizacion
    original_values = df_original.values
    normalized_values = df_normalized.values
    # print('\033[32m*************** Affinity Propagation ***************\033[039m\n')
    # # Creación de un diccionario con las preferencias,
    # results = {
    #     'preferences': [-65, -45, -30, -12, -11, -10, -8.35, -8.25, -8, -7, -6.25]
    # }
    # # Se crea diccionario de todos los resultados de los clusters para Affinity propagation
    # affinity_propagation = dict()
    # k = 1;
    # for p in results['preferences']:
    #     clustering = AffinityPropagation(preference=p).fit(df_normalized)
    #     clustering_centers_indices = clustering.cluster_centers_indices_
    #     clustering_labels = clustering.labels_
    #     affinity_propagation[f'K{k + 1}'] = clustering_labels
    #     k += 1
    # # Se hace llamado a la funcion write_to_csv() para poder guardar el fichero
    # write_to_csv(affinity_propagation, 'AffinityPropagation_result')
    #
    # print('\033[32m*************** Gaussian Mixture ***************\033[039m\n')
    # # Se crea diccionario de todos los resultados de los clusters para Gaussian Mixture
    # gaussian_mixture = dict()
    # for k in range(2, 13):
    #     gaus_mix = GaussianMixture(n_components=k, n_init=10, max_iter=100).fit(X)
    #     gaussian_mixture[f'K{k}'] = gaus_mix.predict(original_values)
    # # Se hace llamado a la funcion write_to_csv() para poder guardar el fichero
    # write_to_csv(gaussian_mixture, 'GaussianMixture_result')
    #
    # print('\033[32m*************** Mean Shift Clustering ***************\033[039m\n')
    # precomputed_quantiles = {
    #     2: 0.4142142142142142,
    #     3: 0.4117117117117117,
    #     4: 0.38143143143143143,
    #     5: 0.34364364364364364,
    #     6: 0.33338338338338336,
    #     7: 0.32837837837837835,
    #     8: 0.24254254254254254,
    #     9: 0.23503503503503503,
    #     10: 0.22502502503503502,
    #     11: 0.20225225225225227,
    #     12: 0.20225225225225227,
    #     13: 0.2,
    # }
    # mean_shift = dict()
    # for k, v in precomputed_quantiles.items():
    #     bandwidth = estimate_bandwidth(df_normalized, quantile=v)
    #     model = MeanShift(bandwidth=bandwidth).fit(df_normalized)
    #     mean_shift[f'K{k}'] = model.predict(df_normalized)
    # write_to_csv(mean_shift, 'MeanShift_result')
    #
    # print('\033[32m*************** BIRCH Clustering ***************\033[039m\n')
    # birch = dict()
    # for i in range(2, 13, 1):
    #     model = Birch(branching_factor=50, n_clusters=i, threshold=0.5)
    #     model.fit(data_X)
    #     birch[f'K{i}'] = model.predict(data_X)
    # write_to_csv(birch, 'BIRCH_result')

    # print('\033[32m*************** DBSCAN Clustering ***************\033[039m\n')
    # dbscan = dict()
    # for k in range(2, 13):
    #     clusters = DBSCAN(eps=0.6, min_samples=k).fit(data_X)
    #     dbscan[f'K{k}'] = clusters.labels_
    # write_to_csv(dbscan, 'DBSCAN_result')
    #
    # print('\033[32m*************** OPTICS Clustering ***************\033[039m\n')
    # optics = dict()
    # for k in range(2, 13):
    #     clustering = OPTICS(min_cluster_size=k, ).fit(data_X)
    #     optics[f'K{k}'] = clustering.labels_
    # write_to_csv(optics, 'OPTICS_result')

    # print('\033[32m***************  Spectral Clustering ***************\033[039m\n')
    # spectral = dict()
    # for k in range(2, 13):
    #     clustering = SpectralClustering(n_clusters=k, assign_labels='discretize', random_state=0).fit(data_X)
    #     spectral[f'K{k}'] = clustering.labels_
    # write_to_csv(spectral, 'Spectral_result')
    # print('\033[32m***************  MiniBatchKMeans Clustering ***************\033[039m\n')
    # mini_Barch_means = dict()
    # for k in range(2, 13):
    #     kmeans = MiniBatchKMeans(n_clusters=k)
    #     kmeans.fit(data_X)
    #     mini_Barch_means[f'K{k}'] = kmeans.labels_
    # write_to_csv(mini_Barch_means, 'MiniBatchKMeans_result')
    # print('\033[32m***************  Agglomerative Hierarchical Clustering Clustering ***************\033[039m\n')
    # agglomerative_hierarchical_clustering = dict()
    # for k in range(2, 13):
    #     cluster_ea = AgglomerativeClustering(n_clusters=k, linkage='ward', affinity='euclidean')
    #     predict = cluster_ea.fit_predict(data_X)
    #     agglomerative_hierarchical_clustering[f'K{k}'] = predict
    # write_to_csv(agglomerative_hierarchical_clustering, 'AgglomerativeClustering_result')
    # print('\033[32m***************   KMeans ***************\033[039m\n')
    # kmeans = dict()
    # for k in range(2, 13):
    #     cluster = KMeans(n_clusters=k).fit(data_X)
    #     kmeans[f'K{k}'] = cluster.labels_
    # write_to_csv(kmeans, 'KMeans_result')
    #
    # print('\033[32m***************   CURE ***************\033[039m\n')
    # cure = dict()
    # alpha = 0.1
    # numRepPoints = 5
    # for k in range(2, 13):
    #     cluster = runCURE(data_X, numRepPoints, alpha, k)
    #     cure[f'K{k}'] = cluster
    # write_to_csv(cure, 'CURE_result')
    #
    # print('\033[32m*************** Expectation-Maximation ***************\033[039m\n')
    # # Se crea diccionario de todos los resultados de los clusters para Expectation-Maximation
    # expectation_max = dict()
    # for k in range(2, 13):
    #     exp_max = GaussianMixture(n_components=k, n_init=10, max_iter=100).fit(original_values)
    #     expectation_max[f'K{k}'] = exp_max.predict(original_values)
    # # Se hace llamado a la funcion write_to_csv() para poder guardar el fichero
    # write_to_csv(expectation_max, 'ExpectatioMaximization_result')

    # Método de intersección para encontrar el mejor K
    # Se consulta cuantos algoritmos se desea analizar
    numAlgorithm = input("Cuantos algoritmos desea analizar? =>  ")
    # Se consulta cuantos cluster se desea analizar
    numClusters = input("Cuantos clusters desea analizar? => ")
    # variable para poder analizar los porcentajes, se usa para escoger porcentages mayores al numero escogido en %
    average = 50
    # Se lee los algoritmos para poder analizarlos dependiendo de los datos ingresados anteriormente
    algorithms_analyze, selected_clusters, selected_algorithms = read_algorithm(numAlgorithm, numClusters)
    # Se hace llamado a la funcion separated_cluster_per_id() par poder tener los resultados de los cluster separados
    algorithms_analyze_cluster = separated_cluster_per_id()
    best_k = dict()
    # Se inicia la matriz con los algoritmos y clusters seleccionados
    Kn_average_algo = pd.DataFrame(index=selected_algorithms, columns=selected_clusters)
    # Bucle para analizar cada cluster empieza desde K=2 hasta el maximo num ingresado (max=12)
    for n in range(2, int(numClusters) + 2):
        dict_values_cluster = dict()
        list_matrix_values = list()
        index_matrix = list()
        filter_cluster_per_group(algorithms_analyze_cluster, n)
        print(f'\033[32m xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \033[039m\n')
        # Se crea la matriz original de cada cluster para luego realizar la sumatoria
        df_original = pd.DataFrame.from_dict(dict_values_cluster, orient='index')
        # Se guarda la matriz tranpuesta para usos futuros
        df_transpose = df_original.transpose()
        print(f'\033[32m xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \033[039m\n\n')
        # funcion para obtener la matrix con los porcentajes
        print(f'\033[32m ----- Matriz K{n} Average \033[039m\n')
        df_result = complete_matrix(df_original, df_transpose, n)
        print(df_result.to_string())
        df_result.to_excel(f'Result_Matrix/Result_K{n}.xlsx')
        print(f'\n\n\033[32m -----  Matriz K{n} Average below {average}% \033[039m\n')
        # Funcion para obtener matrix mayores a result > 50%
        df_max = matrix_max_average(n, int(average))
        # funcion para obtener la frecuencia total del Kn
        df_best_k = average_per_column(df_max)
        print(df_best_k)
        df_best_k.to_excel(f'Result_Matrix/Result_Average_K{n}.xlsx')
        avergae_per_Kn_algorithm(df_best_k, n)
        print(f'\033[32m xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \033[039m\n\n')
        best_k[f'K{n}'] = df_best_k.at['Total', 'Average Freq'] / n

    print(f'\033[32m xxxxxxxxxxxxxxxxxxxx Mejor K segun los algoritmos xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \033[039m\n')
    print(f'\033[32m Matriz de todos los algoritmos y los resultados: \033[039m')
    # Se suma por columna todos los resultados para poder visualizar el mejor K
    sum_column = Kn_average_algo.sum(axis=0)
    Kn_average_algo.loc['Total'] = sum_column.to_numpy()
    print(Kn_average_algo.to_string())

    # Se analiza el maximo valor de los resultados
    max_k_value = max(best_k.values())
    max_k_name = [key for key, value in best_k.items() if value == max(best_k.values())]
    print(f'\n\033[32m ==> Mejor K es {max_k_name} con: {max_k_value} \033[039m\n')

    K_selected = int(max_k_name[len(max_k_name) - 1].split('K')[1])

    # Se hace llamado a la funcion graph_tsne para visualizar la clusterinzacion de los algoritmos selecionados
    # con el mejor k, K-1 y K+1
    graph_tsne(K_selected)
