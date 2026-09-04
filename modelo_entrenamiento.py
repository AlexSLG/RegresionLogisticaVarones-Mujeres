"""
================================================================================
MODELO DE REGRESIÓN LOGÍSTICA PARA CLASIFICACIÓN DE GÉNERO (VARÓN / MUJER)
================================================================================

Este script contiene el pipeline completo de Machine Learning para el proyecto
de clasificación binaria de género a partir de imágenes faciales.

FLUJO DEL SCRIPT:
    1. Carga de imágenes del dataset
    2. Análisis Exploratorio de Datos (EDA)
    3. Preprocesamiento de imágenes
    4. Entrenamiento del modelo de Regresión Logística (manual + sklearn)
    5. Evaluación con métricas completas
    6. Guardado del modelo serializado (.pkl)

EJECUCIÓN:
    python modelo_entrenamiento.py

SALIDA:
    - modelo/logistic_model.pkl  → Modelo entrenado
    - modelo/scaler.pkl          → Estandarizador ajustado
    - Gráficos de EDA y métricas en pantalla
================================================================================
"""

# =================================================================================
# SECCIÓN 1: IMPORTACIÓN DE LIBRERÍAS
# =================================================================================
# NumPy: librería para operaciones matemáticas con arreglos multidimensionales.
#   Se usa para manipulatear matrices de píxeles, vectores de pesos, y cálculos
#   numéricos de la regresión logística (producto punto, sumas, etc.)
#
# Pandas: librería para análisis y manipulación de datos tabulares.
#   Se usa para crear DataFrames con las métricas y resúmenes del modelo.
#
# OpenCV (cv2): librería de visión por computadora.
#   Se usa para leer imágenes del disco, convertirlas a escala de grises,
#   y redimensionarlas a un tamaño fijo (64x64 píxeles).
#
# Matplotlib: librería de visualización de datos.
#   Se usa para generar gráficos: histogramas, curvas ROC, matrices de confusión.
#
# Seaborn: librería de visualización estadística basada en Matplotlib.
#   Se usa para gráficos más estilizados (heatmaps, distribuciones).
#
# Scikit-learn: librería de Machine Learning.
#   LogisticRegression: implementación optimizada de regresión logística.
#   GridSearchCV: búsqueda de hiperparámetros óptimos con validación cruzada.
#   StandardScaler: estandarización de features (media=0, desviación=1).
#   Métricas: accuracy_score, precision_score, recall_score, f1_score,
#             confusion_matrix, roc_curve, roc_auc_score.
#
# Joblib: librería para serializar/deserializar objetos Python (modelos).
#   Se usa para guardar el modelo entrenado y el scaler en archivos .pkl.

import numpy as np
import pandas as pd
import cv2
import os
import sys
import io

# Configurar salida estandar a UTF-8 para evitar errores de codificacion
# en consolas Windows (cp1252 por defecto)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # Backend no-interactivo para entornos sin display
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report,
)

# Suprimir advertencias para una salida más limpia
warnings.filterwarnings("ignore")

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


# =================================================================================
# SECCIÓN 2: CARGA DE IMÁGENES
# =================================================================================
# Este bloque lee todas las imágenes del dataset organizado en carpetas:
#   Dataset/train/female/ → imágenes de mujeres (etiqueta: 1)
#   Dataset/train/male/   → imágenes de varones (etiqueta: 0)
#   Dataset/test/female/  → imágenes de mujeres para prueba
#   Dataset/test/male/    → imágenes de varones para prueba
#
# Cada imagen pasa por tres transformaciones:
#   1. cv2.imread(ruta, cv2.IMREAD_GRAYSCALE) → convierte a escala de grises
#      (1 canal en lugar de 3 canales RGB, reduciendo dimensionalidad)
#   2. cv2.resize(img, (64, 64)) → redimensiona a 64x64 píxeles para
#      que todas las imágenes tengan el mismo tamaño de entrada
#   3. La imagen queda como una matriz NumPy de forma (64, 64)

# Constante global: tamaño al que se redimensionan todas las imágenes
# 64x64 = 4096 píxeles por imagen (estas serán nuestras features/variables)
IMAGE_SIZE = 64

# Rutas a las carpetas del dataset
# Nota: Se asume que el dataset está en la carpeta 'Dataset' relativa al script
TRAIN_FEMALE = os.path.join("Dataset", "train", "female")
TRAIN_MALE = os.path.join("Dataset", "train", "male")
TEST_FEMALE = os.path.join("Dataset", "test", "female")
TEST_MALE = os.path.join("Dataset", "test", "male")


def cargar_imagenes(directorio, label):
    """
    Carga todas las imágenes .jpg de un directorio y las convierte a matrices
    NumPy en escala de grises de 64x64 píxeles.

    PARÁMETROS:
        directorio (str): ruta a la carpeta con las imágenes
        label (int): etiqueta de clase (1=mujer, 0=varón)

    RETORNA:
        imagenes (np.array): arreglo de shape (N, 64, 64) con N imágenes
        etiquetas (np.array): arreglo de shape (N,) con las etiquetas
    """
    imagenes = []
    etiquetas = []

    for nombre_archivo in tqdm(os.listdir(directorio), desc=f"Cargando {os.path.basename(directorio)}"):
        # Construir la ruta completa al archivo de imagen
        ruta = os.path.join(directorio, nombre_archivo)

        # Leer la imagen en escala de grises (1 canal, valores 0-255)
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

        # Verificar que la imagen se leyó correctamente (no es None)
        if img is not None:
            # Redimensionar a 64x64 píxeles
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            imagenes.append(img)
            etiquetas.append(label)

    return np.array(imagenes), np.array(etiquetas)


def cargar_dataset_completo():
    """
    Carga el dataset completo de entrenamiento y prueba.

    RETORNA:
        train_images: np.array de shape (N_train, 64, 64)
        train_labels: np.array de shape (N_train,) — 1=mujer, 0=varón
        test_images: np.array de shape (N_test, 64, 64)
        test_labels: np.array de shape (N_test,) — 1=mujer, 0=varón
    """
    print("=" * 60)
    print("CARGANDO DATASET")
    print("=" * 60)

    # Cargar imágenes de entrenamiento
    train_f, label_f = cargar_imagenes(TRAIN_FEMALE, label=1)  # Female = 1
    train_m, label_m = cargar_imagenes(TRAIN_MALE, label=0)    # Male = 0

    # Unir ambas clases en un solo conjunto
    train_images = np.concatenate((train_f, train_m), axis=0)
    train_labels = np.concatenate((label_f, label_m), axis=0)

    # Cargar imágenes de prueba
    test_f, label_ft = cargar_imagenes(TEST_FEMALE, label=1)
    test_m, label_mt = cargar_imagenes(TEST_MALE, label=0)

    test_images = np.concatenate((test_f, test_m), axis=0)
    test_labels = np.concatenate((label_ft, label_mt), axis=0)

    print(f"\nImagenes de entrenamiento: {train_images.shape[0]} "
          f"({np.sum(train_labels == 1)} mujeres, {np.sum(train_labels == 0)} varones)")
    print(f"Imagenes de prueba: {test_images.shape[0]} "
          f"({np.sum(test_labels == 1)} mujeres, {np.sum(test_labels == 0)} varones)")

    return train_images, train_labels, test_images, test_labels


# =================================================================================
# SECCIÓN 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =================================================================================
# El EDA (Exploratory Data Analysis) es el proceso de investigar el dataset
# antes de modelar. Permite entender:
#   - ¿Está balanceado el dataset? (¿igual cantidad de hombres y mujeres?)
#   - ¿Hay imágenes corruptas o con valores atípicos?
#   - ¿Cómo se distribuyen los valores de intensidad de píxeles?
#   - ¿Qué ejemplos representativos hay de cada clase?

def realizar_eda(train_images, train_labels, test_images, test_labels):
    """
    Realiza el Análisis Exploratorio de Datos y genera gráficos informativos.
    """
    print("\n" + "=" * 60)
    print("ANALISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 60)

    # --- Gráfico 1: Distribución de clases ---
    # Verificar si el dataset está balanceado. Un dataset desbalanceado
    # puede causar que el modelo favorezca la clase mayoritaria.
    n_female_train = np.sum(train_labels == 1)
    n_male_train = np.sum(train_labels == 0)
    n_female_test = np.sum(test_labels == 1)
    n_male_test = np.sum(test_labels == 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribución en entrenamiento
    axes[0].bar(["Mujer (1)", "Varón (0)"], [n_female_train, n_male_train],
                color=["#FF6B9D", "#4ECDC4"], edgecolor="black", linewidth=0.8)
    axes[0].set_title("Distribución de Clases - Entrenamiento", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Cantidad de Imágenes")
    axes[0].set_xlabel("Género")
    for i, v in enumerate([n_female_train, n_male_train]):
        axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold", fontsize=12)

    # Distribución en prueba
    axes[1].bar(["Mujer (1)", "Varón (0)"], [n_female_test, n_male_test],
                color=["#FF6B9D", "#4ECDC4"], edgecolor="black", linewidth=0.8)
    axes[1].set_title("Distribución de Clases - Prueba", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Cantidad de Imágenes")
    axes[1].set_xlabel("Género")
    for i, v in enumerate([n_female_test, n_male_test]):
        axes[1].text(i, v + 1, str(v), ha="center", fontweight="bold", fontsize=12)

    plt.tight_layout()
    plt.savefig("eda_distribucion_clases.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("-> Guardado: eda_distribucion_clases.png")

    # --- Gráfico 2: Ejemplos de imágenes de cada clase ---
    # Mostrar algunas imágenes de referencia para verificar que se cargaron bien
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Ejemplos de Imágenes del Dataset", fontsize=16, fontweight="bold")

    # Índices aleatorios para mostrar ejemplos variados
    np.random.seed(42)
    idx_female = np.random.choice(np.where(train_labels == 1)[0], 5, replace=False)
    idx_male = np.random.choice(np.where(train_labels == 0)[0], 5, replace=False)

    for i, idx in enumerate(idx_female):
        axes[0, i].imshow(train_images[idx], cmap="gray")
        axes[0, i].set_title("Mujer", color="#FF6B9D", fontweight="bold")
        axes[0, i].axis("off")

    for i, idx in enumerate(idx_male):
        axes[1, i].imshow(train_images[idx], cmap="gray")
        axes[1, i].set_title("Varón", color="#4ECDC4", fontweight="bold")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("eda_ejemplos_imagenes.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("-> Guardado: eda_ejemplos_imagenes.png")

    # --- Gráfico 3: Histograma de intensidades de píxeles ---
    # Analizar cómo se distribuyen los valores de brillo (0=negro, 255=blanco)
    # Esto ayuda a entender si hay diferencias claras en brillo entre clases
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Aplanar todas las imágenes para obtener un vector de píxeles por clase
    female_pixels = train_images[train_labels == 1].flatten()
    male_pixels = train_images[train_labels == 0].flatten()

    axes[0].hist(female_pixels, bins=50, color="#FF6B9D", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Distribución de Intensidad - Mujeres", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Valor de Píxel (0-255)")
    axes[0].set_ylabel("Frecuencia")

    axes[1].hist(male_pixels, bins=50, color="#4ECDC4", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Distribución de Intensidad - Varones", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Valor de Píxel (0-255)")
    axes[1].set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.savefig("eda_histograma_intensidades.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("-> Guardado: eda_histograma_intensidades.png")

    # --- Estadísticas descriptivas ---
    print("\n--- Estadisticas de Intensidad de Pixeles ---")
    print(f"Mujeres - Promedio: {female_pixels.mean():.2f}, "
          f"Desviación Estándar: {female_pixels.std():.2f}")
    print(f"Varones - Promedio: {male_pixels.mean():.2f}, "
          f"Desviación Estándar: {male_pixels.std():.2f}")

    # Verificar datos nulos
    total_nulos = np.sum(np.isnan(train_images.astype(float)))
    print(f"\nValores nulos en dataset de entrenamiento: {total_nulos}")

    # Verificar dimensiones
    print(f"Forma de imagen individual: {train_images[0].shape} "
          f"({train_images[0].shape[0]}x{train_images[0].shape[1]} píxeles)")
    print(f"Total de features por imagen (al aplanar): {IMAGE_SIZE * IMAGE_SIZE} "
          f"(= {IMAGE_SIZE}x{IMAGE_SIZE})")


# =================================================================================
# SECCIÓN 4: PREPROCESAMIENTO DE IMÁGENES
# =================================================================================
# Antes de entrenar el modelo, las imágenes deben transformarse:
#
# PASO 1: Aplanamiento (Flatten)
#   Cada imagen de 64x64 se convierte en un vector de 4096 elementos.
#   Razón: La regresión logística espera un vector 1D de features,
#   no una matriz 2D. Es como "desenrollar" la imagen fila por fila.
#   Ejemplo: imagen [[1,2],[3,4]] → vector [1, 2, 3, 4]
#
# PASO 2: Normalización / Estandarización
#   Los valores de píxel van de 0 a 255. StandardScaler transforma cada
#   feature para que tenga media = 0 y desviación estándar = 1.
#   Razón: La regresión logística converge más rápido y es más estable
#   cuando las features están en escalas similares.
#   Fórmula: z = (x - media) / desviación_estándar
#
# PASO 3: Separación Train/Test
#   El dataset ya viene separado en carpetas. Train se usa para ajustar
#   el modelo, test se usa para evaluar su rendimiento con datos nuevos.

def preprocesar_imagenes(train_images, test_images):
    """
    Preprocesa las imágenes: aplanamiento y estandarización.

    PARÁMETROS:
        train_images: np.array de shape (N_train, 64, 64)
        test_images: np.array de shape (N_test, 64, 64)

    RETORNA:
        x_train: np.array de shape (N_train, 4096) — features de entrenamiento
        x_test: np.array de shape (N_test, 4096) — features de prueba
        scaler: StandardScaler ajustado (se guarda para uso en predicción)
    """
    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO DE IMAGENES")
    print("=" * 60)

    # --- Paso 1: Aplanar imágenes de (N, 64, 64) a (N, 4096) ---
    # reshape(-1, IMAGE_SIZE*IMAGE_SIZE) convierte cada matriz 64x64
    # en un vector unidimensional de 4096 elementos
    n_train = train_images.shape[0]
    n_test = test_images.shape[0]

    x_train = train_images.reshape(n_train, IMAGE_SIZE * IMAGE_SIZE)
    x_test = test_images.reshape(n_test, IMAGE_SIZE * IMAGE_SIZE)

    print(f"Forma despues de aplanar:")
    print(f"  x_train: {x_train.shape} ({n_train} imagenes x {IMAGE_SIZE * IMAGE_SIZE} features)")
    print(f"  x_test:  {x_test.shape} ({n_test} imagenes x {IMAGE_SIZE * IMAGE_SIZE} features)")

    # --- Paso 2: Estandarización con StandardScaler ---
    # StandardScaler ajusta (fit) solo con datos de entrenamiento para evitar
    # "data leakage" (filtración de información del test al entrenamiento).
    # Luego transforma (transform) ambos conjuntos usando los mismos parámetros.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)  # fit + transform en train
    x_test = scaler.transform(x_test)        # solo transform en test (usa stats de train)

    print(f"\nEstandarizacion aplicada:")
    print(f"  Media de x_train despues de escalar: {x_train.mean():.6f} (~ 0)")
    print(f"  Desviacion de x_train despues de escalar: {x_train.std():.2f} (~ 1)")

    return x_train, x_test, scaler


# =================================================================================
# SECCIÓN 5: REGRESIÓN LOGÍSTICA DESDE CERO (Implementación Manual)
# =================================================================================
# Esta sección implementa la regresión logística sin usar scikit-learn,
# solo con NumPy. Esto permite entender a fondo las matemáticas del modelo.
#
# --- CONCEPTOS FUNDAMENTALES ---
#
# REGRESIÓN LOGÍSTICA:
#   Es un algoritmo de clasificación binaria. A pesar de llamarse "regresión",
#   se usa para clasificar (predecir categorías, no valores continuos).
#   Funciona calculando una probabilidad usando la función sigmoide.
#
# FUNCIÓN SIGMOIDE σ(z):
#   σ(z) = 1 / (1 + e^(-z))
#
#   Donde z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b (suma ponderada de features)
#     - w = pesos (coeficientes): indican cuánto influye cada feature
#     - b = sesgo (bias): término independiente que ajusta el umbral
#     - x = features (valores de píxeles estandarizados)
#
#   La sigmoide transforma cualquier número real z en un valor entre 0 y 1.
#   Si σ(z) ≥ 0.5 → predicción = 1 (mujer)
#   Si σ(z) < 0.5 → predicción = 0 (varón)
#
# FUNCIÓN DE COSTE (Log Loss / Binary Cross-Entropy):
#   J(w,b) = -1/m * Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
#
#   Donde:
#     - m = número de ejemplos de entrenamiento
#     - y = etiqueta real (0 o 1)
#     - ŷ = predicción del modelo (probabilidad entre 0 y 1)
#
#   Esta función mide qué tan lejos están las predicciones de las etiquetas reales.
#   Valores más bajos indican mejor ajuste.
#
# GRADIENT DESCENT (Descenso de Gradiente):
#   Algoritmo de optimización que ajusta los pesos iterativamente para
#   minimizar la función de coste. En cada paso:
#     w_nuevo = w_viejo - learning_rate * ∂J/∂w
#     b_nuevo = b_viejo - learning_rate * ∂J/∂b
#
#   El learning_rate (tasa de aprendizaje) controla el tamaño del paso.
#   Si es muy grande → el modelo puede "pasarse" del mínimo.
#   Si es muy pequeño → el modelo tarda mucho en converger.
#
# FORWARD PROPAGATION:
#   Calcular z = w.T @ x + b, luego ŷ = σ(z), luego el coste.
#
# BACKWARD PROPAGATION:
#   Calcular las derivadas (gradientes) de la función de coste respecto
#   a los pesos y el sesgo, para saber en qué dirección ajustarlos.


def sigmoid(z):
    """
    Función Sigmoide: σ(z) = 1 / (1 + e^(-z))

    Transforma cualquier número real z en un valor entre 0 y 1.
    Esto permite interpretar el resultado como una probabilidad.

    Ejemplos:
        σ(0) = 0.5   → frontera de decisión
        σ(10) ≈ 1.0  → casi seguro de que es clase 1
        σ(-10) ≈ 0.0 → casi seguro de que es clase 0

    PARÁMETROS:
        z: np.array — suma ponderada (w.T @ x + b)

    RETORNA:
        np.array con valores entre 0 y 1 (probabilidades)
    """
    # np.clip evita overflow en exp() cuando z es muy negativo
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def inicializar_pesos_y_bias(dimension):
    """
    Inicializa los pesos (w) y el sesgo (bias, b) del modelo.

    Los pesos se inicializan en 0.01 para cada feature (4096 features).
    El sesgo se inicializa en 0.0.

    ¿Por qué no inicializar en 0?
    Si todos los pesos fueran 0, la derivada del coste sería igual para
    todas las features y el modelo no aprendería diferencias entre ellas.
    Inicializar con valores pequeños rompe esta simetría.

    PARÁMETROS:
        dimension (int): número de features (4096 = 64*64)

    RETORNA:
        w: np.array de forma (4096, 1) — vector de pesos
        b: float — sesgo (bias)
    """
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b


def forward_backward_propagation(w, b, x_train, y_train):
    """
    Realiza la propagación hacia adelante (forward) y hacia atrás (backward).

    FORWARD:
        1. Calcula z = w.T @ x + b (suma ponderada)
        2. Calcula ŷ = σ(z) (probabilidad predicha)
        3. Calcula el coste J usando log loss

    BACKWARD:
        4. Calcula las derivadas parciales ∂J/∂w y ∂J/∂b
        5. Estos gradientes indican cómo ajustar w y b para reducir el error

    PARÁMETROS:
        w: np.array (4096, 1) — pesos actuales
        b: float — sesgo actual
        x_train: np.array (4096, m) — datos de entrenamiento (features en filas)
        y_train: np.array (1, m) — etiquetas reales

    RETORNA:
        cost: float — valor de la función de coste
        gradientes: dict con "derivative_weight" y "derivative_bias"
    """
    m = x_train.shape[1]  # número de ejemplos

    # --- FORWARD PROPAGATION ---
    # z = suma ponderada de features × pesos + sesgo
    z = np.dot(w.T, x_train) + b

    # ŷ = probabilidad predicha por la sigmoide
    y_head = sigmoid(z)

    # Función de coste: Log Loss (Binary Cross-Entropy)
    # Evitamos log(0) usando np.clip
    y_head_clipped = np.clip(y_head, 1e-10, 1 - 1e-10)
    cost = (-1 / m) * np.sum(
        y_train * np.log(y_head_clipped) + (1 - y_train) * np.log(1 - y_head_clipped)
    )

    # --- BACKWARD PROPAGATION ---
    # Derivada parcial del coste respecto a w: ∂J/∂w = (1/m) * x @ (ŷ - y)
    derivative_weight = (1 / m) * np.dot(x_train, (y_head - y_train).T)

    # Derivada parcial del coste respecto a b: ∂J/∂b = (1/m) * Σ(ŷ - y)
    derivative_bias = (1 / m) * np.sum(y_head - y_train)

    gradientes = {
        "derivative_weight": derivative_weight,
        "derivative_bias": derivative_bias,
    }

    return cost, gradientes


def entrenar_modelo_manual(x_train, y_train, x_test, y_test,
                           learning_rate=0.01, num_iterations=1500):
    """
    Entrena el modelo de Regresión Logística desde cero usando descenso de gradiente.

    PARÁMETROS:
        x_train: np.array (4096, m_train) — features de entrenamiento
        y_train: np.array (1, m_train) — etiquetas de entrenamiento
        x_test: np.array (4096, m_test) — features de prueba
        y_test: np.array (1, m_test) — etiquetas de prueba
        learning_rate: float — tasa de aprendizaje (controla el tamaño del paso)
        num_iterations: int — número de iteraciones del descenso de gradiente

    RETORNA:
        w: np.array — pesos optimizados
        b: float — sesgo optimizado
        cost_list: list — historial de costes por iteración
    """
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO MANUAL - REGRESION LOGISTICA")
    print("=" * 60)
    print(f"  Learning rate: {learning_rate}")
    print(f"  Iteraciones: {num_iterations}")
    print(f"  Features: {x_train.shape[0]}")
    print(f"  Ejemplos de entrenamiento: {x_train.shape[1]}")

    # Inicializar pesos y sesgo
    dimension = x_train.shape[0]
    w, b = inicializar_pesos_y_bias(dimension)

    cost_list = []

    # --- DESCENSO DE GRADIENTE ---
    # En cada iteración:
    #   1. Se calcula la predicción actual (forward)
    #   2. Se calcula el error y los gradientes (backward)
    #   3. Se actualizan los pesos y el sesgo en dirección opuesta al gradiente
    for i in range(num_iterations):
        cost, gradientes = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)

        # Actualización de pesos: w = w - lr * ∂J/∂w
        w = w - learning_rate * gradientes["derivative_weight"]

        # Actualización de sesgo: b = b - lr * ∂J/∂b
        b = b - learning_rate * gradientes["derivative_bias"]

        # Mostrar progreso cada 300 iteraciones
        if i % 300 == 0:
            print(f"  Iteracion {i:4d} | Coste: {cost:.6f}")

    # --- EVALUACIÓN ---
    # Usar los pesos y sesgo optimizados para predecir en train y test
    y_pred_test = predecir(w, b, x_test)
    y_pred_train = predecir(w, b, x_train)

    acc_train = accuracy_score(y_train.flatten(), y_pred_train.flatten())
    acc_test = accuracy_score(y_test.flatten(), y_pred_test.flatten())

    print(f"\n  Exactitud Entrenamiento: {acc_train * 100:.2f}%")
    print(f"  Exactitud Prueba: {acc_test * 100:.2f}%")

    # Graficar la curva de convergencia del coste
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_iterations), cost_list, color="#FF6B9D", linewidth=2)
    plt.title("Convergencia del Coste durante el Entrenamiento", fontsize=14, fontweight="bold")
    plt.xlabel("Número de Iteración")
    plt.ylabel("Coste (Log Loss)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("coste_convergencia.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("-> Guardado: coste_convergencia.png")

    return w, b, cost_list


def predecir(w, b, x):
    """
    Realiza predicciones usando los pesos y sesgo entrenados.

    PARÁMETROS:
        w: np.array (4096, 1) — pesos optimizados
        b: float — sesgo optimizado
        x: np.array (4096, m) — datos a predecir

    RETORNA:
        y_pred: np.array (1, m) — predicciones (0 o 1)
    """
    z = sigmoid(np.dot(w.T, x) + b)
    y_pred = np.zeros((1, z.shape[1]))

    # Umbral de decisión: si σ(z) ≥ 0.5 → clase 1, si no → clase 0
    for i in range(z.shape[1]):
        y_pred[0, i] = 1 if z[0, i] >= 0.5 else 0

    return y_pred


# =================================================================================
# SECCIÓN 6: REGRESIÓN LOGÍSTICA CON SCIKIT-LEARN (Implementación Optimizada)
# =================================================================================
# Scikit-learn ofrece una implementación optimizada de Regresión Logística que:
#   - Usa solvers numéricos más avanzados (LBFGS, liblinear, etc.)
#   - Incluye regularización L1/L2 para evitar overfitting
#   - Permite búsqueda automática de hiperparámetros
#
# HIPERPARÁMETROS CLAVE:
#
#   C (parámetro de regularización):
#     Inverso del strength de regularización.
#     C alto → menos regularización → puede hacer overfitting
#     C bajo → más regularización → puede hacer underfitting
#     Valores típicos: 0.001 a 1000
#
#   penalty (tipo de regularización):
#     L2 (Ridge): penaliza la suma de cuadrados de los pesos
#       → w_nuevo = w - lr * (∂J/∂w + λ·w)
#       → Tiende a distribuir los pesos de forma más uniforme
#     L1 (Lasso): penaliza la suma de valores absolutos de los pesos
#       → Puede llevar algunos pesos a exactamente 0 (selección de features)
#
#   solver (algoritmo de optimización):
#     'lbfgs': bueno para dataset pequeños/medianos, usa approximations
#     'liblinear': bueno para datasets pequeños, usa coordinate descent
#
# VALIDACIÓN CRUZADA (GridSearchCV):
#   Prueba diferentes combinaciones de hiperparámetros usando K-Fold.
#   Divide el train en K partes, entrena con K-1 y valida con 1,
#   rotando hasta que todas las partes hayan sido validación.
#   Esto da una estimación más robusta del rendimiento.

def entrenar_con_sklearn(x_train, y_train, x_test, y_test):
    """
    Entrena un modelo de Regresión Logística con scikit-learn y
    optimiza hiperparámetros con GridSearchCV.

    RETORNA:
        mejor_modelo: LogisticRegression ajustado con los mejores hiperparámetros
        scaler: StandardScaler ajustado (ya fue ajustado antes, pero se retorna para consistencia)
    """
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO CON SCIKIT-LEARN + GRIDSEARCHCV")
    print("=" * 60)

    # Aplanar etiquetas de (1, N) a (N,) para compatibilidad con sklearn
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    # --- Búsqueda de hiperparámetros ---
    # C: prueba valores en escala logarítmica de 0.001 a 1000
    # penalty: prueba regularización L1 y L2
    param_grid = {
        "C": np.logspace(-3, 3, 7),  # [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],  # liblinear soporta tanto L1 como L2
    }

    lr_base = LogisticRegression(random_state=42, max_iter=1000)

    # GridSearchCV con 10 folds (validación cruzada)
    # cv=10 significa que el train se divide en 10 partes:
    #   9 para entrenar, 1 para validar, rotando 10 veces
    print("Buscando mejores hiperparametros con GridSearchCV (10-fold)...")
    grid_search = GridSearchCV(
        lr_base,
        param_grid,
        cv=10,                # 10-fold validación cruzada
        scoring="accuracy",   # métrica de evaluación: exactitud
        n_jobs=-1,            # usar todos los cores del procesador
        verbose=0,
    )
    grid_search.fit(x_train, y_train_flat)

    # Mostrar resultados de la búsqueda
    print(f"\n  Mejores hiperparametros encontrados:")
    print(f"    C = {grid_search.best_params_['C']}")
    print(f"    Penalty = {grid_search.best_params_['penalty']}")
    print(f"    Solver = {grid_search.best_params_['solver']}")
    print(f"    Accuracy (CV): {grid_search.best_score_ * 100:.2f}%")

    # --- Entrenar el modelo final con los mejores hiperparámetros ---
    mejor_modelo = LogisticRegression(
        C=grid_search.best_params_["C"],
        penalty=grid_search.best_params_["penalty"],
        solver=grid_search.best_params_["solver"],
        random_state=42,
        max_iter=1000,
    )
    mejor_modelo.fit(x_train, y_train_flat)

    # Evaluar en train y test
    acc_train = mejor_modelo.score(x_train, y_train_flat)
    acc_test = mejor_modelo.score(x_test, y_test_flat)

    print(f"\n  Exactitud Entrenamiento (sklearn): {acc_train * 100:.2f}%")
    print(f"  Exactitud Prueba (sklearn): {acc_test * 100:.2f}%")

    return mejor_modelo


# =================================================================================
# SECCIÓN 7: ANÁLISIS DE COEFICIENTES
# =================================================================================
# En Regresión Logística, cada feature (píxel) tiene un coeficiente (peso) que
# indica cuánto influye en la predicción.
#
#   Coeficiente POSITIVO grande → el píxel "encendido" (claro) favorece clase 1
#   Coeficiente NEGATIVO grande → el píxel "encendido" favorece clase 0
#   Coeficiente cercano a 0 → el píxel tiene poca influencia
#
# Al visualizar los coeficientes como una imagen 64x64, podemos ver qué
# regiones del rostro son más discriminativas para el género.
# Generalmente: pelo largo, forma del rostro, cejas, etc.

def analizar_coeficientes(modelo):
    """
    Analiza y visualiza los coeficientes del modelo de Regresión Logística
    para entender qué regiones de la imagen son más importantes para
    la clasificación de género.
    """
    print("\n" + "=" * 60)
    print("ANALISIS DE COEFICIENTES")
    print("=" * 60)

    # Obtener los coeficientes del modelo
    # Forma: (1, 4096) → se reorganiza a (64, 64) para visualizar como imagen
    coeficientes = modelo.coef_[0]

    # Reorganizar los coeficientes como imagen 64x64
    coef_img = coeficientes.reshape(IMAGE_SIZE, IMAGE_SIZE)

    # --- Gráfico 1: Mapa de calor de coeficientes ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mapa de calor: regiones rojas = favorecen mujer, azules = favorecen varón
    im = axes[0].imshow(coef_img, cmap="RdBu_r", aspect="auto")
    axes[0].set_title("Coeficientes del Modelo\n(Rojo = Mujer, Azul = Varón)",
                      fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Columna de píxel")
    axes[0].set_ylabel("Fila de píxel")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # --- Gráfico 2: Distribución de coeficientes ---
    axes[1].hist(coeficientes, bins=80, color="#8B5CF6", alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero line")
    axes[1].set_title("Distribución de Coeficientes", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Valor del Coeficiente")
    axes[1].set_ylabel("Frecuencia")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("analisis_coeficientes.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("-> Guardado: analisis_coeficientes.png")

    # Estadísticas de coeficientes
    print(f"\n  Total de coeficientes: {len(coeficientes)}")
    print(f"  Coeficiente promedio: {coeficientes.mean():.6f}")
    print(f"  Coeficiente maximo (mas feminizante): {coeficientes.max():.6f}")
    print(f"  Coeficiente minimo (mas masculinizante): {coeficientes.min():.6f}")
    print(f"  Coeficientes positivos: {np.sum(coeficientes > 0)} "
          f"({np.sum(coeficientes > 0) / len(coeficientes) * 100:.1f}%)")
    print(f"  Coeficientes negativos: {np.sum(coeficientes < 0)} "
          f"({np.sum(coeficientes < 0) / len(coeficientes) * 100:.1f}%)")


# =================================================================================
# SECCIÓN 8: EVALUACIÓN COMPLETA DE MÉTRICAS
# =================================================================================
# Métricas de evaluación para clasificación binaria:
#
# --- MATRIZ DE CONFUSIÓN ---
#                      Predicho: Mujer    Predicho: Varón
#   Real: Mujer        [  TP  ]          [  FN  ]
#   Real: Varón        [  FP  ]          [  TN  ]
#
#   TP (True Positive):  correctly predijo "Mujer" cuando era Mujer
#   TN (True Negative):  correctly predijo "Varón" cuando era Varón
#   FP (False Positive): predijo "Mujer" cuando era Varón (Error Tipo I)
#   FN (False Negative): predijo "Varón" cuando era Mujer (Error Tipo II)
#
# --- EXACTITUD (Accuracy) ---
#   Accuracy = (TP + TN) / (TP + TN + FP + FN)
#   Proporción de predicciones correctas sobre el total.
#   Limitación: no es confiable si el dataset está desbalanceado.
#
# --- PRECISIÓN (Precision) ---
#   Precision = TP / (TP + FP)
#   De todos los que predije como "Mujer", ¿cuántos realmente eran Mujer?
#   Alta precisión = pocos falsos positivos.
#
# --- SENSIBILIDAD / RECALL ---
#   Recall = TP / (TP + FN)
#   De todos los que realmente eran "Mujer", ¿cuántos logré detectar?
#   Alto recall = pocos falsos negativos.
#
# --- PUNTUACIÓN F1 (F1-Score) ---
#   F1 = 2 * (Precision * Recall) / (Precision + Recall)
#   Media armónica de Precision y Recall. Penaliza valores extremos.
#   F1 alto = buen balance entre precisión y sensibilidad.
#
# --- CURVA ROC Y AUC ---
#   ROC (Receiver Operating Characteristic): gráfico de Tasa de Verdaderos
#   Positivos (TPR/Recall) vs Tasa de Falsos Positivos (FPR) en distintos umbrales.
#
#   TPR = TP / (TP + FN) = Recall
#   FPR = FP / (FP + TN)
#
#   AUC (Area Under the Curve): área bajo la curva ROC.
#   AUC = 0.5 → modelo aleatorio (no sirve)
#   AUC = 1.0 → modelo perfecto
#   AUC > 0.7 → aceptable
#   AUC > 0.8 → bueno
#   AUC > 0.9 → excelente

def evaluar_modelo(modelo, x_test, y_test, scaler):
    """
    Realiza una evaluación completa del modelo con todas las métricas requeridas.

    PARÁMETROS:
        modelo: LogisticRegression entrenado
        x_test: np.array — features de prueba
        y_test: np.array — etiquetas reales
        scaler: StandardScaler — estandarizador (no se usa aquí pero se mantiene por consistencia)
    """
    print("\n" + "=" * 60)
    print("EVALUACION COMPLETA DE METRICAS")
    print("=" * 60)

    y_test_flat = y_test.flatten()

    # --- Predicciones ---
    y_pred = modelo.predict(x_test)                   # predicciones de clase (0 o 1)
    y_prob = modelo.predict_proba(x_test)[:, 1]       # probabilidades de clase 1

    # --- Cálculo de métricas ---
    accuracy = accuracy_score(y_test_flat, y_pred)
    precision = precision_score(y_test_flat, y_pred)
    recall = recall_score(y_test_flat, y_pred)
    f1 = f1_score(y_test_flat, y_pred)
    auc_roc = roc_auc_score(y_test_flat, y_prob)

    print("\n+---------------------------------------------------+")
    print("|          METRICAS DE EVALUACION                    |")
    print("+---------------------------------------------------+")
    print(f"|  Exactitud (Accuracy):  {accuracy * 100:6.2f}%                   |")
    print(f"|  Precision (Precision): {precision * 100:6.2f}%                   |")
    print(f"|  Sensibilidad (Recall): {recall * 100:6.2f}%                   |")
    print(f"|  Puntuacion F1:         {f1 * 100:6.2f}%                   |")
    print(f"|  AUC-ROC:               {auc_roc * 100:6.2f}%                   |")
    print("+---------------------------------------------------+")

    # --- Reporte de clasificacion detallado ---
    print("\nReporte de Clasificacion:")
    print(classification_report(
        y_test_flat, y_pred,
        target_names=["Varon (0)", "Mujer (1)"]
    ))

    # --- MATRIZ DE CONFUSIÓN ---
    cm = confusion_matrix(y_test_flat, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("Matriz de Confusion:")
    print(f"  TP (Verdadero Positivo):  {tp} -- Mujer predicha correctamente")
    print(f"  TN (Verdadero Negativo):  {tn} -- Varon predicho correctamente")
    print(f"  FP (Falso Positivo):      {fp} -- Varon confundido con Mujer")
    print(f"  FN (Falso Negativo):      {fn} -- Mujer confundida con Varon")

    # Gráfico de la matriz de confusión
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap de la matriz de confusión
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Varón (0)", "Mujer (1)"],
                yticklabels=["Varón (0)", "Mujer (1)"],
                ax=axes[0], linewidths=0.5, linecolor="black")
    axes[0].set_title("Matriz de Confusión", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Etiqueta Real")

    # --- CURVA ROC ---
    fpr, tpr, thresholds = roc_curve(y_test_flat, y_prob)

    axes[1].plot(fpr, tpr, color="#FF6B9D", linewidth=2.5,
                 label=f"ROC (AUC = {auc_roc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
                 label="Clasificador Aleatorio (AUC = 0.5)")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#FF6B9D")
    axes[1].set_title("Curva ROC", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[1].set_ylabel("Tasa de Verdaderos Positivos (TPR/Recall)")
    axes[1].legend(loc="lower right", fontsize=11)
    axes[1].set_xlim([-0.02, 1.02])
    axes[1].set_ylim([-0.02, 1.02])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluacion_metricas.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("-> Guardado: evaluacion_metricas.png")

    # Retornar métricas para uso en la app
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# =================================================================================
# SECCIÓN 9: GUARDADO DEL MODELO
# =================================================================================
# El modelo entrenado se serializa (guarda) en un archivo .pkl usando joblib.
# Esto permite cargarlo después sin necesidad de reentrenar.
#
# Se guardan dos archivos:
#   1. logistic_model.pkl → el modelo LogisticRegression entrenado
#   2. scaler.pkl → el StandardScaler ajustado (necesario para preprocesar
#      nuevas imágenes de la misma forma que las de entrenamiento)
#
# ¿Por qué guardar el scaler?
#   Porque al predecir una imagen nueva, se debe estandarizar con la misma
#   media y desviación estándar que se usó durante el entrenamiento.

def guardar_modelo(modelo, scaler, carpeta="modelo"):
    """
    Guarda el modelo entrenado y el scaler en archivos .pkl.
    """
    print("\n" + "=" * 60)
    print("GUARDADO DEL MODELO")
    print("=" * 60)

    # Crear la carpeta si no existe
    os.makedirs(carpeta, exist_ok=True)

    # Guardar el modelo de regresión logística
    ruta_modelo = os.path.join(carpeta, "logistic_model.pkl")
    joblib.dump(modelo, ruta_modelo)
    print(f"  Modelo guardado en: {ruta_modelo}")

    # Guardar el estandarizador
    ruta_scaler = os.path.join(carpeta, "scaler.pkl")
    joblib.dump(scaler, ruta_scaler)
    print(f"  Scaler guardado en: {ruta_scaler}")

    # Verificar tamaños de archivo
    tam_modelo = os.path.getsize(ruta_modelo) / 1024
    tam_scaler = os.path.getsize(ruta_scaler) / 1024
    print(f"  Tamano modelo: {tam_modelo:.1f} KB")
    print(f"  Tamano scaler: {tam_scaler:.1f} KB")


# =================================================================================
# SECCIÓN 10: FUNCIÓN PRINCIPAL (MAIN)
# =================================================================================
# Este bloque se ejecuta cuando se corre el script directamente:
#   python modelo_entrenamiento.py

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MODELO DE REGRESION LOGISTICA PARA CLASIFICACION DE GENERO  ")
    print("=" * 60)

    # 1. Cargar dataset
    train_images, train_labels, test_images, test_labels = cargar_dataset_completo()

    # 2. Análisis Exploratorio de Datos (EDA)
    realizar_eda(train_images, train_labels, test_images, test_labels)

    # 3. Preprocesamiento
    x_train, x_test, scaler = preprocesar_imagenes(train_images, test_images)

    # Preparar etiquetas en formato (1, N) para implementación manual
    y_train_manual = train_labels.reshape(1, -1)
    y_test_manual = test_labels.reshape(1, -1)

    # 4. Entrenamiento manual (para entender las matemáticas)
    w_manual, b_manual, cost_list = entrenar_modelo_manual(
        x_train.T, y_train_manual, x_test.T, y_test_manual,
        learning_rate=0.01, num_iterations=1500
    )

    # 5. Entrenamiento con sklearn (mejor rendimiento)
    modelo_sklearn = entrenar_con_sklearn(x_train, train_labels, x_test, test_labels)

    # 6. Análisis de coeficientes
    analizar_coeficientes(modelo_sklearn)

    # 7. Evaluación completa de métricas
    metricas = evaluar_modelo(modelo_sklearn, x_test, test_labels, scaler)

    # 8. Guardar el modelo y scaler para la aplicación web
    guardar_modelo(modelo_sklearn, scaler)

    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("  Archivos generados:")
    print("    - modelo/logistic_model.pkl")
    print("    - modelo/scaler.pkl")
    print("  Siguiente paso: ejecutar 'streamlit run app.py'")
    print("=" * 60)
