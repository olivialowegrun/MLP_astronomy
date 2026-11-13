import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# CARGA Y LIMPIEZA DEL DATASET
# Cargamos del dataset
df = pd.read_csv("star_classification.csv")

# Visualizamos las columnas para posteriormente selecionar aquellas con las que trabajar.
print(list(df.columns))

# Para el preprocesamiento seleccionamos estrictamente las bandas fotométricas y el redshift.
selected_features = ["u", "g", "r", "i", "z", "redshift"]
df_subset = df[selected_features].copy() # Copiamos en el subset las características que hemos seleccionado...
df_subset["class"] = df["class"] # y la clasificación.

df_subset["u_g"] = df_subset["u"] - df_subset["g"] # Añadimos nuevas variables de colores
df_subset["g_r"] = df_subset["g"] - df_subset["r"]
df_subset["r_i"] = df_subset["r"] - df_subset["i"]
df_subset["i_z"] = df_subset["i"] - df_subset["z"]

selected_features = ["u", "g", "r", "i", "z", "redshift", "u_g", "g_r", "r_i", "i_z"] # Actualizamos features

# Primero comprobamos si realmente existen valores nulos en nuestro subset.
print("\nContamos los valores nulos = ")
null_counts = df_subset.isnull().sum()
print(null_counts) # Como no hay valores nulos, no tenemos que eliminar los NaN

# Separación de características (X) y etiquetas (y)
X = df_subset[selected_features]

# Convertimos las clases de texto a (0, 1, 2) y obtenemos las etiquetas (y) reemplazadas por números.
class_mapping = {"GALAXY": 0, "STAR": 1, "QSO": 2}
y = df_subset["class"].map(class_mapping)

# Se normalizan los datos numéricos para la estabilidad del descenso del gradiente.
media = np.mean(X, axis=0)
desviacion = np.std(X, axis=0)
X_scaled = (X - media) / (desviacion + 1e-8) # Le sumamos un valor muy pequeño para que no quede igual a 0

# Revolvemos aleatoriamente los datos.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42, stratify=y)
# 'stratify=y' garantiza que las proporciones de las 3 clases sean idénticas en train y test.


# FUNCIONES
def softmax(z):
    # Estabilidad numérica restando el máximo por fila
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # Le restamos el máximo para no tener overflow por exp altas
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y_labels, num_classes): # Devuelve una matriz identidad tamaño num_clases...
    return np.eye(num_classes)[y_labels] # cambiando las filas según y_labels

# ENTRENAMIENTO DEL MODELO
np.random.seed(42) # Para la reproducibilidad
n_samples, n_features = X_train.shape # Devuelve una tupla con las dimensiones de la matriz
n_classes = 3  # Galaxia, estrella, cuásar

# Inicialización de pesos (W) y sesgos o umbrales (b) 
W = np.zeros((n_features, n_classes))
b = np.zeros((1, n_classes))

learning_rate = 0.1 # Tasa de aprendizaje
epocas = 500
Y_train_oh = one_hot(y_train, n_classes)

print("\nEntrenando la red neuronal con descenso de gradiente aplicando Softmax")

for epoca in range(epocas):
    # Forward propagation
    scores = np.dot(X_train, W) + b # Z = XW + b
    probs = softmax(scores) # Le aplicamos la función de activación

    # Backpropagation
    error = probs - Y_train_oh
    dW = (1 / n_samples) * np.dot(X_train.T, error) # La transpuesta para poder realizar el producto 
    db = (1 / n_samples) * np.sum(error, axis=0, keepdims=True)

    # Actualización de parámetros
    W -= learning_rate * dW
    b -= learning_rate * db

# EVALUACIÓN DEL MODELO
test_scores = np.dot(X_test, W) + b
test_probs = softmax(test_scores)
y_pred = np.argmax(test_probs, axis=1) # Para devolver la clase con porcentaje mayor

# Cálculo del Accuracy
fallos = np.sum(y_pred != y_test)  # Cuenta cuántas veces la predicción no fue la real
total_muestras = len(y_test)
accuracy = 1.0 - (fallos / total_muestras)

# Cálculo de la matriz de confusión numérica
cm = confusion_matrix(y_test, y_pred)

print(f"\nEntrenamiento completado")
print(f"Precisión o accuracy: {accuracy * 100:.2f}%")
print("\nMatriz de Confusión:")
print(cm)

# VISUALIZACIÓN DE RESULTADOS
print("\nVisualizaciones de resultados")

class_names = ["Galaxia", "Estrella", "Cuásar"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title("Matriz de Confusión - MLP con Softmax")
plt.xlabel("Clase Predicha")
plt.ylabel("Clase Real")
plt.grid(False)  # Evita que salgan líneas de rejilla sobre el mapa de calor
plt.show()