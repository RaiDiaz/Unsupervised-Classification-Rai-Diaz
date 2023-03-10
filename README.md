# Unsupervised-Classification-Rai-Diaz

El programa desarrollado es basado en clustering, de lo cual se implementa una nueva metrica para encontrar el mejor K. Esta técnia es denominada "Intersección". 

El algoritmo presente muestra el analisis paso por paso para poder entender mejor como funciona esta métrica implementada.

Para esto se debe de tomar en cuenta ciertos aspectos antes de correr el programa.

1. Primero, si quiere remplazar el dataset analizar se debe de cambiar el path asignado en la linea 405, en la variable FILE_NAME_PATH. En esta variable ingresan la nueva direccion del dataset analizar.
2. Segundo, deben de considerar que si el dataset analizar tiene una columna donde se encuentra las clases. deben quitarla. Esto lo pueden hacer ustedes removiendo toda la columna o caso contrario usen la siguiente linea de código 
```ruby
df_original.drop(<nombre de la columna a eliminar>, inplace=True, axis=1)
```
3. Tercero, se debe descomentar el código desde la linea 419 hasta la 532. Con la finalidad de realizar la clusterizacion del dataset con los 12 algoritmos implementados. 
4. Cuarto, una vez realizada la clusterización el programa pide el usuario que ingrese el numero de algoritmos analizar con el numero de clusters (tabla 1 y 2). 

NOTA: El paso 1 y 2 pueden ser opcionales, ya que si se realiza el analisis con el mismo dataset no se debe cambiar nada. Si se desea tener en carpetas estos archivos se puede mantener como el ejemplo. Pero, si se realiza un cambio o si no se crea alguna de las carpetas descritas anteriormente, se debe cambiar la direccion de donde se guarda los archivos.

3. Tercero, el programa pide que se ingrese el numero de algoritmos analizar de lo cual toma un rango de 1-12. También, pide el numero de cluster para analizar que toma un rango de 2-12. 

NOTA:
Algoritmos:
| Input  | Output |
| -------| ------------- |
|   1    | Affinity Propagation  |
|   2    | Agglomerative Hierarchical Clustering  |
|   3    |  BIRCH |
|   4    |  DBSCAN |
|   5    | Gaussian Mixture Models |
|   6    | Mean Shift Clustering |
|   7    | Mini-Batch K-Means |
|   8    | OPTICS |
|   9    | Spectral-clustering |
|   10   | Expectation-Maximization |
|   11   | k-means |
|   12   | CURE |
*******************************************************************************
Cluster:
| Input  | Output |
| -------| ------------- |
|   2    | k=2 |
|   3    | k=3 |
|   4    | k=4 |
|   5    | k=5 |
|   6    | k=6 |
|   7    | k=7 |
|   8    | k=8 |
|   9    | k=9 |
|   10    | k=10 |
|   11   | k=11 |
|   12   | k=12 |


Por otro lado, al momento de descargar el código. Se debe configurar e instalar las librerias. Para esto se creo un archivo requirements.txt, en este archivo genera el interpreter para poder correr el programa sin ningun problema.
