"""
APLICACION WEB - CLASIFICACION DE GENERO CON REGRESION LOGISTICA

Permite subir una imagen facial y obtener prediccion de genero (Varon / Mujer)
con la probabilidad asignada por el modelo de Regresion Logistica.

Ejecucion local:  streamlit run app.py
Despliegue:       Streamlit Community Cloud (conectado a repositorio GitHub)
"""

import streamlit as st
import numpy as np
import cv2
import joblib
import os

# =================================================================================
# CONFIGURACION DE LA PAGINA
# =================================================================================
st.set_page_config(
    page_title="Clasificacion de Genero - Regresion Logistica",
    page_icon="\U0001f9ec",
    layout="centered",
)

# =================================================================================
# CARGA DEL MODELO Y SCALER
# =================================================================================
# Los archivos .pkl contienen el modelo entrenado y el estandarizador.
# Se cargan una sola vez y se almacenan en session_state para no recargarlos
# en cada interaccion de la app.

@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo de Regresion Logistica y el StandardScaler desde archivos .pkl.
    @st.cache_resource asegura que se carguen solo una vez (optimizacion).
    """
    modelo = joblib.load(os.path.join("modelo", "logistic_model.pkl"))
    scaler = joblib.load(os.path.join("modelo", "scaler.pkl"))
    return modelo, scaler

modelo, scaler = cargar_modelo()

# =================================================================================
# FUNCIONES DE PROCESAMIENTO DE IMAGEN
# =================================================================================
# Para que una imagen subida por el usuario sea procesada por el modelo,
# debe pasar por la misma transformacion que las imagenes de entrenamiento:
#   1. Convertir a escala de grises (1 canal, valores 0-255)
#   2. Redimensionar a 64x64 pixeles (mismo tamano que entrenamiento)
#   3. Aplanar a vector unidimensional de 4096 elementos (64*64)
#   4. Estandarizar con StandardScaler (media=0, desviacion=1)
#
# IMAGE_SIZE = 64 corresponde al tamano fijo usado durante el entrenamiento.
# El modelo fue entrenado con vectores de 4096 features (64*64 = 4096).

IMAGE_SIZE = 64

def preprocesar_imagen(imagen_bytes):
    """
    Convierte bytes de imagen a feature vector preprocesado para el modelo.

    PASOS:
        1. Decodificar bytes a imagen OpenCV (BGR)
        2. Convertir BGR a escala de grises (1 canal)
        3. Redimensionar a 64x64 pixeles
        4. Aplanar de matriz (64,64) a vector (4096,)
        5. Reshape a (1, 4096) para compatibilidad con sklearn
        6. Estandarizar con el scaler ajustado durante entrenamiento

    PARAMETROS:
        imagen_bytes: bytes de la imagen subida por el usuario

    RETORNA:
        imagen_gris: imagen en escala de grises para mostrar en la app
        features: vector (1, 4096) estandarizado listo para prediccion
    """
    # Decodificar bytes a imagen numpy usando OpenCV
    # np.frombuffer convierte bytes a array; cv2.imdecode lee la imagen
    nparr = np.frombuffer(imagen_bytes, np.uint8)
    imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convertir de BGR (formato OpenCV) a escala de grises
    # cv2.COLOR_BGR2GRAY: convierte imagen de 3 canales (BGR) a 1 canal (gris)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Redimensionar a 64x64 pixeles (mismo tamano usado en entrenamiento)
    imagen_redim = cv2.resize(imagen_gris, (IMAGE_SIZE, IMAGE_SIZE))

    # Aplanar la matriz de 64x64 a un vector de 4096 elementos
    # reshape(1, -1) lo pone en formato (1 ejemplo, 4096 features)
    features = imagen_redim.reshape(1, IMAGE_SIZE * IMAGE_SIZE)

    # Estandarizar usando el scaler previamente ajustado
    # IMPORTANTE: se usa transform() NO fit_transform() porque el scaler
    # ya fue ajustado con los datos de entrenamiento
    features = scaler.transform(features)

    return imagen_gris, features

# =================================================================================
# INTERFAZ DE USUARIO - HEADER
# =================================================================================
st.title("\U0001f9ec Clasificacion de Genero con Regresion Logistica")
st.markdown("""
### Clasificacion Binaria: Varon vs Mujer

Este aplicativo utiliza un modelo de **Regresion Logistica** entrenado con imagenes
faciales para predecir el genero de una persona a partir de su fotografia.

**Como funciona:**
1. Sube una imagen de un rostro
2. El modelo procesa la imagen (escala de grises, 64x64 pixeles)
3. Se aplica la funcion sigmoide: `sigma(z) = 1 / (1 + e^(-z))`
4. Si `sigma(z) >= 0.5` -> Mujer (1), si no -> Varon (0)
""")

st.divider()

# =================================================================================
# INTERFAZ DE USUARIO - BARRA LATERAL (SIDEBAR)
# =================================================================================
# La barra lateral muestra informacion tecnica del modelo y las metricas
# obtenidas durante el entrenamiento.

with st.sidebar:
    st.header("\U0001f4ca Informacion del Modelo")

    st.markdown("""
    **Algoritmo:** Regresion Logistica

    **Variable objetivo:**
    - Mujer = 1 (clase positiva)
    - Varon = 0 (clase negativa)

    **Features de entrada:**
    - 4,096 pixeles (64x64 escala de grises)
    - Estandarizados con StandardScaler

    **Funcion de decision:**
    ```
    z = w1*x1 + w2*x2 + ... + w4096*x4096 + b
    sigma(z) = 1 / (1 + e^(-z))
    ```
    """)

    st.divider()
    st.subheader("\U0001f3af Metricas del Modelo")

    # Las metricas se calculan durante el entrenamiento y se hardcodean aqui.
    # Para actualizar, ejecutar modelo_entrenamiento.py y copiar los valores.
    st.markdown("""
    | Metrica | Valor |
    |---------|-------|
    | Accuracy | 89.00% |
    | Precision | 89.00% |
    | Recall | 89.00% |
    | F1-Score | 89.00% |
    | AUC-ROC | 93.71% |
    """)

    st.info("Modelo entrenado con 3,491 imagenes (1,747 mujeres, 1,744 varones). Hyperparametros: C=0.001, penalty=l2, solver=liblinear.")

    st.divider()
    st.subheader("\U0001f4d1 Glosario de Metricas")

    st.markdown("""
    **Accuracy (Exactitud):**
    Proporción de predicciones correctas sobre el total.

    **Precision:**
    De todos los predichos como Mujer, cuantos realmente lo son.

    **Recall (Sensibilidad):**
    De todos los que realmente son Mujer, cuantos detecto.

    **F1-Score:**
    Media armonica de Precision y Recall.

    **AUC-ROC:**
    Area bajo la curva ROC. Mide capacidad de discriminacion.
    - 0.5 = aleatorio
    - 1.0 = perfecto
    """)

# =================================================================================
# INTERFAZ DE USUARIO - CARGA DE IMAGEN
# =================================================================================
st.subheader("\U0001f4f7 Subir Imagen para Clasificar")

# file_uploader: widget de Streamlit que permite seleccionar un archivo.
# - type=["jpg", "jpeg", "png"]: formatos de imagen aceptados
# - key: identificador unico del widget
uploaded_file = st.file_uploader(
    "Seleccione una imagen facial (JPG, JPEG o PNG)",
    type=["jpg", "jpeg", "png"],
    key="uploaded_image",
)

if uploaded_file is not None:
    # Leer los bytes de la imagen subida
    imagen_bytes = uploaded_file.read()

    # Preprocesar la imagen para el modelo
    imagen_gris, features = preprocesar_imagen(imagen_bytes)

    # =================================================================================
    # PREDICCION
    # =================================================================================
    # modelo.predict(): devuelve la prediccion de clase (0 o 1)
    # modelo.predict_proba(): devuelve probabilidades [P(clase=0), P(clase=1)]
    prediccion = modelo.predict(features)[0]
    probabilidades = modelo.predict_proba(features)[0]

    # Probabilidad de la clase predicha
    prob_clase_1 = probabilidades[1]  # Probabilidad de ser Mujer
    prob_clase_0 = probabilidades[0]  # Probabilidad de ser Varon

    # Interpretar la prediccion
    if prediccion == 1:
        genero_predicho = "Mujer"
        color = "#FF6B9D"
        emoji = "\u2640\ufe0f"
    else:
        genero_predicho = "Varon"
        color = "#4ECDC4"
        emoji = "\u2642\ufe0f"

    # =================================================================================
    # MOSTRAR RESULTADOS
    # =================================================================================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Imagen Original")
        st.image(imagen_bytes, caption="Imagen subida", use_container_width=True)

    with col2:
        st.markdown("### Imagen Preprocesada")
        st.image(imagen_gris, caption="Escala de grises 64x64", use_container_width=True)

    st.divider()

    # Resultado de la clasificacion
    st.markdown(f"## Resultado: {emoji} **{genero_predicho}**")
    st.markdown(f"**Probabilidad:** {max(prob_clase_0, prob_clase_1) * 100:.2f}%")

    # Barra de probabilidad visual
    # st.progress: muestra una barra de progreso que indica la probabilidad
    st.progress(max(prob_clase_0, prob_clase_1))

    # Detalle de probabilidades
    st.markdown("### Probabilidades por Clase")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Varon (0)", f"{prob_clase_0 * 100:.2f}%")
    with col_b:
        st.metric("Mujer (1)", f"{prob_clase_1 * 100:.2f}%")

    # =================================================================================
    # EXPLICACION DE LA PREDICCION
    # =================================================================================
    with st.expander("\U0001f4a1 Explicacion de la Prediccion", expanded=False):
        st.markdown("""
        ### Como funciona la Regresion Logistica

        **Paso 1: Procesamiento de imagen**
        - La imagen se convierte a escala de grises (1 canal, valores 0-255)
        - Se redimensiona a 64x64 pixeles (4,096 features)
        - Se estandariza con StandardScaler (media=0, desviacion=1)

        **Paso 2: Calculo de z (suma ponderada)**
        ```
        z = w1*x1 + w2*x2 + ... + w4096*x4096 + b
        ```
        Donde:
        - w = pesos del modelo (coeficientes aprendidos)
        - x = valores de pixeles estandarizados
        - b = sesgo (bias)

        **Paso 3: Funcion Sigmoide**
        ```
        sigma(z) = 1 / (1 + e^(-z))
        ```
        Transforma z en una probabilidad entre 0 y 1.

        **Paso 4: Decision**
        - Si sigma(z) >= 0.5 -> Mujer (1)
        - Si sigma(z) < 0.5 -> Varon (0)
        """)

else:
    # Mensaje cuando no se ha subido imagen
    st.info("\U0001f447 Suba una imagen facial para obtener una prediccion de genero.")

# =================================================================================
# PIE DE PAGINA
# =================================================================================
st.divider()
st.markdown("""
**Proyecto:** Regresion Logistica para Clasificacion Binaria de Genero
**Herramientas:** Python, Scikit-learn, OpenCV, Streamlit
**Modelo:** Regresion Logistica con optimizacion de hiperparametros (GridSearchCV)
""")
