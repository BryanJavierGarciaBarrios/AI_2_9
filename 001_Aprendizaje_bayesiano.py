pip install scikit-learn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Datos de ejemplo: mensajes de correo electrónico y sus etiquetas
emails = ["Oferta especial de hoy. ¡Gana dinero rápido!",
          "Reunión de la junta directiva a las 3 pm.",
          "¡Descuento en tu próxima compra!",
          "Informe financiero trimestral adjunto.",
          "Ganaste un millón de dólares en la lotería.",
          "Confirmación de la reserva de vuelo.",
          "Píldoras milagrosas para bajar de peso.",
          "Factura adjunta, por favor pague a tiempo."]

labels = ["spam", "no spam", "spam", "no spam", "spam", "no spam", "spam", "no spam"]

# Crear un modelo de clasificación Naive Bayes
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Realizar predicciones
y_pred = naive_bayes.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Precisión del modelo:", accuracy)
print("Matriz de confusión:")
print(confusion)
