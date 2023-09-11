import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos desde un archivo CSV
dataSet = pd.read_csv('/Users/kathbejarano/Desktop/IA/PerceptronB/dsPerceptron.csv')

# Separar las características (features) de las etiquetas (labels)
X = dataSet.iloc[:, :-1]  # Todas las columnas excepto la última
y = dataSet.iloc[:, -1]   # La última columna

# Dividir el conjunto de datos en entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Crear y entrenar el perceptrón en el conjunto de entrenamiento
clf = Perceptron()
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de validación para calcular métricas
y_pred_validation = clf.predict(X_validation)

# Calcular métricas de evaluación en el conjunto de validación
precision_validation = precision_score(y_validation, y_pred_validation)
recall_validation = recall_score(y_validation, y_pred_validation)
accuracy_validation = accuracy_score(y_validation, y_pred_validation)

# Imprimir las métricas de validación
print("Métricas de Validación:")
print("Precision:", precision_validation)
print("Recall:", recall_validation)
print("Accuracy:", accuracy_validation)

# Realizar predicciones en el conjunto de prueba
y_pred_test = clf.predict(X_test)

# Calcular métricas de evaluación en el conjunto de prueba
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Imprimir las métricas de prueba
print("\nMétricas de Prueba:")
print("Precision:", precision_test)
print("Recall:", recall_test)
print("Accuracy:", accuracy_test)

# Agregar la matriz de confusión
confusion = confusion_matrix(y_test, y_pred_test)
print("\nMatriz de Confusión:")
print(confusion)

# Crear un gráfico de barras para las métricas de validación y prueba
metrics_names = ['Precision', 'Recall', 'Accuracy']
validation_metrics_values = [precision_validation, recall_validation, accuracy_validation]
test_metrics_values = [precision_test, recall_test, accuracy_test]

width = 0.35
x = np.arange(len(metrics_names))

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, validation_metrics_values, width, label='Validación', color='blue')
rects2 = ax.bar(x + width/2, test_metrics_values, width, label='Prueba', color='green')

ax.set_ylabel('Valor')
ax.set_title('Métricas de Evaluación (Validación y Prueba)')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()

# Mostrar el gráfico de barras de validación y prueba juntas
plt.show()