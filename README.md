# Unsupervised-Classification-Rai-Diaz

El programa desarrollado es basado en clustering, de lo cual se implementa una nueva metrica para encontrar el mejor K. Esta técnia es denominada "Intersección". 

El algoritmo presente muestra el analisis paso por paso para poder entender mejor como funciona esta métrica implementada.

Para esto se debe de tomar en cuenta ciertos aspectos antes de correr el programa.

 - Se debe crear dos carpetas Result_csv y Result_xlx
 - Si se quiere guardar la matriz que bota cada cluster se debe crear la carpeta Result_matrix y descomentar las lineas de codigo respectivas.

Por otro lado, se debe seguir los siguientes pasos:

1. Primero, si quiere remplazar el dataset analizar se debe de cambair el path asignado en la linea 405, en la variable FILE_NAME_PATH. En esta variable ingresan la nueva direccion del dataset analizar.
2. Segundo, se debe descomentar el código desde la linea 419 hasta la 532. Con la finalidad de realizar la clusterizacion del dataset con los 12 algoritmos implementados. 

NOTA: El paso 1 y 2 pueden ser opcionales, ya que si se realiza el analisis con el mismo dataset no se debe cambiar nada. Si se desea tener en carpetas estos archivos se puede mantener como el ejemplo. Pero, si se realiza un cambio o si no se crea alguna de las carpetas descritas anteriormente, se debe cambiar la direccion de donde se guarda los archivos.

3. Tercero, el programa pide que se ingrese el numero de algoritmos analizar de lo cual toma un rango de 1-12. También, pide el numero de cluster para analizar que toma un rango de 1-11. 

NOTA:
Algoritmos:
1 -> Affinity Propagation
2 -> Agglomerative Hierarchical Clustering
3 -> BIRCH
4 -> DBSCAN
5 -> Gaussian Mixture Models
6 -> Mean Shift Clustering
7 -> Mini-Batch K-Means
8 -> OPTICS
9 -> Spectral-clustering
10 -> Expectation-Maximization
11 -> k-means
12 -> CURE
*******************************************************************************
Cluster:
1 -> k = 2
2 -> k = 3
3 -> k = 4
4 -> k = 5
5 -> k = 6
6 -> k = 7
7 -> k = 8
8 -> k = 9
9 -> k = 10
10 -> k = 11
11 -> k = 12
