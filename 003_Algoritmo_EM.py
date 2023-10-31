import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generar datos de muestra
np.random.seed(0)
n_samples = 300
mean1 = [2, 2]
cov1 = [[1, 0], [0, 1]]
mean2 = [5, 5]
cov2 = [[1, 0], [0, 1]]
X = np.concatenate([np.random.multivariate_normal(mean1, cov1, int(n_samples / 2)),
                    np.random.multivariate_normal(mean2, cov2, int(n_samples / 2))])

# Inicializar y ajustar el modelo de mezcla gaussiana
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(X)

# Resultados del modelo
print("Parámetros del modelo:")
print("Pesos de las componentes:", gmm.weights_)
print("Medias de las componentes:", gmm.means_)
print("Covarianzas de las componentes:\n", gmm.covariances_)

# Predicción de las asignaciones de clase (componente)
labels = gmm.predict(X)

# Visualización de los datos y las asignaciones de clase
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Ajuste del modelo de mezcla gaussiana')
plt.show()
