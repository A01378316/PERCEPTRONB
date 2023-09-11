#Kathia Bejarano Zamora
#A01378316

#**************************IMPORTACIÓN DE MIS LIBRERIAS**************************************
#Importante mencionr que para este código ya se implementó el uso de sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

#*******************************PREPARACIÓN DE MIS DATOS*********************************************************

# Cargo mi dataset desde un csv (Usaremos el mismo que la entrega pasada para evaluar y oder comparar)
dataSet = pd.read_csv('/Users/kathbejarano/Desktop/IA/PerceptronB/dsPerceptron.csv')

# Separo mis columnas para entender cuales son características y cuales no

X = dataSet.iloc[:, :-1]  # Todas las columnas excepto la última pues no debemos mezclar con los resultados esperados
y = dataSet.iloc[:, -1]   # La última columna que es el resultado esperado

# Dividir el conjunto de datos en entrenamiento, validación y prueba
#Importante mencionar que para esta entrega ya se hace uso del split para hacerlo de manera automática
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#*******************************PERCEPTRÓN*********************************************************

# Crear y entrenar el perceptrón en el conjunto de entrenamiento
#Para este momento ya mandaremos llamar al "Perceptron" que es precisamente la funcuión que se encarga de crear la instancia
#de un clasificador de perceptorn
setA = Perceptron()
#Utilizamos .fit para poder tomar nuestro conjunto x y nuestro conjutno y(entradas y salidas)
setA.fit(X_train, y_train)

#Después de usar .fit nuestro modelo está listo para poderlo someter a una predicción

# Realizar predicciones en el conjunto de validación para calcular métricas
#importante mencionar que aquí ya estamos usando .predict 
y_pred_validation = setA.predict(X_validation)

# Calcular métricas de evaluación en el conjunto de validación
#Al igual que lo he trabajado durante todo el código, aquí ya se usan las funciones de cada métrica
#para obtener los resultados de manera automática
precision_validation = precision_score(y_validation, y_pred_validation)
recall_validation = recall_score(y_validation, y_pred_validation)
accuracy_validation = accuracy_score(y_validation, y_pred_validation)

# Imprimimos las métricas de validación
print("Métricas de Validación:")
print("Precision:", precision_validation)
print("Recall:", recall_validation)
print("Accuracy:", accuracy_validation)

# Realizar predicciones en el conjunto de prueba
y_pred_test = setA.predict(X_test)

# Calcular métricas de evaluación en el conjunto de prueba
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Imprimir las métricas de prueba
print("\nMétricas de Prueba:")
print("Precision:", precision_test)
print("Recall:", recall_test)
print("Accuracy:", accuracy_test)

# Agregamos la matriz de confusión
confusion = confusion_matrix(y_test, y_pred_test)
print("\nMatriz de Confusión:")
print(confusion)

# Imprimir las etiquetas reales y las predicciones en el conjunto de prueba
print("\nEtiquetas Reales en el Conjunto de Prueba:")
print(y_test.values)
print("\nPredicciones en el Conjunto de Prueba:")
print(y_pred_test)

#*******************************VISUALIZACIÓN*********************************************************
# Crear un gráfico de barras para las métricas de validación y prueba para poder tener una comparativa
metrics_names = ['Precision', 'Recall', 'Accuracy']
validation_metrics_values = [precision_validation, recall_validation, accuracy_validation]
test_metrics_values = [precision_test, recall_test, accuracy_test]

width = 0.35
x = np.arange(len(metrics_names))

fig, ax = plt.subplots()
#Hacemos ajustes para tener bien nuestos labels
rects1 = ax.bar(x - width/2, validation_metrics_values, width, label='Validación', color='blue')
rects2 = ax.bar(x + width/2, test_metrics_values, width, label='Prueba', color='green')

ax.set_ylabel('Valor')
ax.set_title('Métricas de Evaluación (Validación y Prueba)')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()

# Mostrar el gráfico de barras de validación y prueba juntas
plt.show()
