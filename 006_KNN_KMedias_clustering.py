
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
datos = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Algoritmo K-Means para agrupación
kmeans = KMeans(n_clusters=2)
kmeans.fit(datos)

# Centroides de los grupos
centroides = kmeans.cluster_centers_
etiquetas = kmeans.labels_

# Algoritmo k-NN para encontrar vecinos más cercanos
knn = NearestNeighbors(n_neighbors=2)
knn.fit(datos)

# Puntos más cercanos
distancias, indices = knn.kneighbors(datos)

# Gráficos
plt.scatter(datos[:,0], datos[:,1], c=etiquetas)
plt.scatter(centroides[:,0], centroides[:,1], marker='x', s=200)
plt.show()

print("Centroides:", centroides)
print("Puntos más cercanos:", indices)
