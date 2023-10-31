from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# Generar datos de ejemplo
X = np.random.rand(1000, 10)
y = np.random.randint(2, size=1000)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear un modelo de red neuronal simple
modelo = Sequential()
modelo.add(Dense(8, input_dim=10, activation='relu'))
modelo.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluar el modelo en datos de prueba
_, accuracy = modelo.evaluate(X_test, y_test)
print(f'Precisión en datos de prueba: {accuracy * 100:.2f}%')
