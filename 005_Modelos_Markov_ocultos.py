pip install hmmlearn

from hmmlearn import hmm
import numpy as np

# Definir el modelo HMM
model = hmm.MultinomialHMM(n_components=2)

# Datos de entrenamiento (secuencia de observaciones)
X = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0]]).T  # Ejemplo de secuencia de observaciones binarias

# Entrenar el modelo HMM
model.fit(X)

# Generar una secuencia de observaciones ocultas
hidden_states, observed_states = model.sample(n_samples=10)

print("Secuencia de estados ocultos:", hidden_states)
print("Secuencia de observaciones:", observed_states)
