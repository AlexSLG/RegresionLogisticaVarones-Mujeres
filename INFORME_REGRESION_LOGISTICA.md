# INFORME: Clasificacion de Genero con Regresion Logistica

---

## 1. CARATULA

**Proyecto:** Clasificacion Binaria de Genero (Varon / Mujer) con Regresion Logistica

**Integrantes:**
- [Nombre Integrante 1] - [Codigo Universitario]
- [Nombre Integrante 2] - [Codigo Universitario]
- [Nombre Integrante 3] - [Codigo Universitario]
- [Nombre Integrante 4] - [Codigo Universitario]

**Universidad:** [Nombre de la Universidad]

**Curso:** [Nombre del Curso]

**Docente:** [Nombre del Docente]

**Fecha:** [Fecha de Entrega]

---

## 2. RESUMEN EJECUTIVO

### Problema
Se busca desarrollar un sistema de clasificacion binaria que, a partir de imagenes faciales, determine si la persona en la imagen es **Varon** o **Mujer**. Este es un problema analogo al clasico "Perros vs. Gatos", donde se discrimina entre dos clases mutuamente excluyentes a partir de caracteristicas de la imagen.

### Modelo Utilizado
Se implemento un modelo de **Regresion Logistica**, uno de los algoritmos mas fundamentales del Machine Learning supervisado. El modelo fue entrenado con un dataset de aproximadamente 3,691 imagenes faciales (1,747 mujeres y 1,744 varones en entrenamiento) y evaluado con 200 imagenes adicionales (100 de cada clase).

### Resultados Principales
- El modelo fue entrenado exitosamente con GridSearchCV (10-fold Cross-Validation) para encontrar los mejores hiperparametros.
- Se implemento tanto una version manual (desde cero con NumPy) como una version optimizada con Scikit-learn.
- Se obtuvieron metricas de accuracy, precision, recall, F1-score y AUC-ROC.
- Se despliego una aplicacion web funcional con Streamlit para predicciones en tiempo real.

### Entregables
- Script de entrenamiento (`modelo_entrenamiento.py`)
- Aplicacion web (`app.py`)
- Modelo serializado (`modelo/logistic_model.pkl`, `modelo/scaler.pkl`)
- Documentacion completa (este archivo)

---

## 3. DEFINICION DEL PROBLEMA

### Contexto
La clasificacion automatica de genero a partir de imagenes faciales es un problema fundamental en computer vision y tiene multiples aplicaciones:
- **Seguridad:** Identificacion automatizada en sistemas de acceso
- **Marketing:** Analisis demografico de audiencias
- **Medicina:** Asistencia en diagnostico medico (algunas condiciones son especificas de genero)
- **Redes sociales:** Etiquetado automatico en plataformas sociales

### Formulacion del Problema
- **Tipo:** Clasificacion binaria
- **Clases:** Varon (0) y Mujer (1)
- **Features (variables de entrada):** 4,096 valores de intensidad de pixel (imagen en escala de grises de 64x64 pixeles)
- **Variable objetivo (Y):** Genero (0 = Varon, 1 = Mujer)

### Valor del Problema
La automatizacion de esta tarea elimina la necesidad de revision manual, permitiendo procesar miles de imagenes por segundo con precision medible. Aunque la regresion logistica no es el estado del arte para vision por computadora, su interpretabilidad y simplicidad la hacen ideal para fines educativos y como linea base (baseline) para modelos mas complejos.

---

## 4. ANALISIS EXPLORATORIO Y PREPARACION DE DATOS

### 4.1 Estructura del Dataset

El dataset esta organizado en carpetas con la siguiente estructura:

```
Dataset/
  train/
    female/    -> 1,747 imagenes
    male/      -> 1,744 imagenes
  test/
    female/    -> 100 imagenes
    male/      -> 100 imagenes
  valid/
    female/    -> 100 imagenes
    male/      -> 100 imagenes
```

**Total:** 3,891 imagenes (1,947 mujeres, 1,944 varones)

### 4.2 Balanceo de Clases

| Conjunto | Mujeres | Varones | Total |
|----------|---------|---------|-------|
| Entrenamiento | 1,747 | 1,744 | 3,491 |
| Prueba | 100 | 100 | 200 |
| Validacion | 100 | 100 | 200 |

El dataset esta **practicamente balanceado** (50.04% mujeres, 49.96% varones), lo cual es ideal porque evita que el modelo favorezca una clase por tener mas ejemplos. Un dataset desbalanceado puede causar que el modelo aprenda a predecir siempre la clase mayoritaria y aun asi obtenga alta accuracy.

### 4.3 Analisis Exploratorio (EDA)

El EDA (Exploratory Data Analysis) es el proceso de investigar el dataset antes de modelar. Se generaron los siguientes graficos:

1. **Distribucion de clases:** Grafico de barras que muestra la cantidad de imagenes por genero en entrenamiento y prueba. Confirma el balanceo del dataset.

2. **Ejemplos de imagenes:** Muestra aleatoria de 5 mujeres y 5 varones para verificar que las imagenes se cargan correctamente y son rostros humanos.

3. **Histograma de intensidades:** Distribucion de valores de pixel (0=negro, 255=blanco) para cada clase. Ayuda a identificar diferencias en brillo entre generos.

**Observaciones del EDA:**
- No se encontraron imagenes corruptas o nulas.
- Las imagenes son rostros recortados con fondo relativamente uniforme.
- La distribucion de intensidades es similar entre ambos generos, lo que indica que la clasificacion depende de patrones estructurales, no solo de brillo.

### 4.4 Preprocesamiento de Datos

El preprocesamiento consta de 4 pasos fundamentales:

#### Paso 1: Conversion a Escala de Grises
```python
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
```
Las imagenes se convierten de 3 canales (BGR) a 1 canal (gris). Esto reduce la dimensionalidad de 3 canales a 1, lo que disminuye la cantidad de features y acelera el entrenamiento sin perder informacion relevante para la clasificacion de genero.

**Termino tecnico - Escala de grises:** Representacion de una imagen donde cada pixel tiene un valor entre 0 (negro) y 255 (blanco), sin informacion de color. Se usa OpenCV con el codigo `cv2.IMREAD_GRAYSCALE`.

#### Paso 2: Redimensionamiento
```python
imagen = cv2.resize(imagen, (64, 64))
```
Todas las imagenes se redimensionan a 64x64 pixeles para garantizar que tengan el mismo tamano. La regresion logistica requiere que todas las entradas tengan el mismo numero de features.

**Por que 64x64?** Es un tamano que preserva suficiente detalle facial para clasificacion mientras mantiene el numero de features manageable (64x64 = 4,096). Tamano mayor aumentaria computacionalmente; tamano menor perderia informacion.

#### Paso 3: Aplanamiento (Flatten)
```python
features = imagen.reshape(1, 64 * 64)  # De matriz (64,64) a vector (4096,)
```
La imagen 2D se convierte en un vector 1D de 4,096 elementos. Esto es necesario porque la regresion logistica opera con vectores unidimensionales: cada pixel se convierte en una feature (variable de entrada).

**Analogia:** Es como "desenrollar" la imagen fila por fila. La primera fila se convierte en los primeros 64 elementos, la segunda fila en los siguientes 64, y asi sucesivamente.

#### Paso 4: Estandarizacion
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Ajusta y transforma en train
X_test = scaler.transform(X_test)        # Solo transforma en test
```
Los valores de pixel (0-255) se transforman para que tengan **media = 0** y **desviacion estandar = 1**.

**Formula:** `z = (x - media) / desviacion_estandar`

**Por que es necesaria?**
- La regresion logistica usa gradient descent, que converge mas rapido cuando las features estan en escalas similares.
- Sin estandarizacion, features con valores grandes (ej: 255) dominarian sobre features con valores pequeños.
- `fit_transform()` solo se aplica a entrenamiento para evitar "data leakage" (filtracion de informacion del conjunto de prueba).

**Termino tecnico - StandardScaler:** Transformador de scikit-learn que estandariza features eliminando la media y escalando a varianza unitaria. Es decir, cada feature tendra media=0 y desviacion estandar=1.

**Termino tecnico - Data Leakage:** Ocurre cuando informacion del conjunto de prueba se filtra al conjunto de entrenamiento. Esto causa una sobreestimacion del rendimiento del modelo. Por eso, el scaler se ajusta (fit) SOLO con datos de entrenamiento.

### 4.5 Division Train/Test

| Conjunto | Cantidad | Proporcion | Uso |
|----------|----------|------------|-----|
| Entrenamiento (Train) | 3,491 | 89.7% | Ajustar los pesos del modelo |
| Prueba (Test) | 200 | 5.1% | Evaluar el rendimiento final |
| Validacion (Valid) | 200 | 5.1% | No se uso en este proyecto |

**Termino tecnico - Train/Test Split:** Separacion del dataset en dos partes: entrenamiento (para que el modelo aprenda) y prueba (para evaluar con datos nunca vistos). Esto evita el **overfitting**, que es cuando el modelo memoriza los datos de entrenamiento pero falla con datos nuevos.

---

## 5. MODELADO CON REGRESION LOGISTICA

### 5.1 Que es la Regresion Logistica?

La Regresion Logistica es un algoritmo de **Machine Learning supervisado** utilizado para **clasificacion binaria**. A pesar de llamarse "regresion", se usa para predecir categorias (clases), no valores continuos.

**Como funciona:**
1. Recibe las features de entrada (4,096 valores de pixel)
2. Calcula una suma ponderada: `z = w1*x1 + w2*x2 + ... + w4096*x4096 + b`
3. Aplica la **funcion sigmoide** para obtener una probabilidad
4. Si la probabilidad >= 0.5, predice clase 1; si no, predice clase 0

### 5.2 La Funcion Sigmoide

La funcion sigmoide es el corazon de la regresion logistica:

```
sigma(z) = 1 / (1 + e^(-z))
```

Donde:
- `z` = suma ponderada de features (w^T * x + b)
- `e` = numero de Euler (constante matematica ~2.71828)
- `sigma(z)` = probabilidad de que la instancia pertenezca a la clase 1

**Propiedades de la sigmoide:**
- Transforma cualquier numero real z en un valor entre 0 y 1
- Si z = 0, sigma(0) = 0.5 (frontera de decision)
- Si z es muy positivo, sigma(z) se acerca a 1
- Si z es muy negativo, sigma(z) se acerca a 0
- Es continua y diferenciable (necesario para gradient descent)

**Interpretacion grafica:** La sigmoide tiene forma de "S". Para valores negativos grandes de z, se acerca a 0. Para valores positivos grandes, se acerca a 1. En z = 0, cruza exactamente por 0.5, que es el umbral de decision por defecto.

### 5.3 Funcion de Coste (Log Loss / Binary Cross-Entropy)

Para medir que tan bien esta funcionando el modelo, se usa la funcion de coste:

```
J(w,b) = -1/m * SUMA[y_i * log(sigma(z_i)) + (1 - y_i) * log(1 - sigma(z_i))]
```

Donde:
- `m` = numero de ejemplos de entrenamiento
- `y_i` = etiqueta real del ejemplo i (0 o 1)
- `sigma(z_i)` = probabilidad predicha por el modelo
- `log` = logaritmo natural (base e)

**Por que esta funcion?**
- Si y = 1 y sigma(z) se acerca a 1, el coste se acerca a 0 (prediccion correcta)
- Si y = 1 y sigma(z) se acerca a 0, el coste tiende a infinito (prediccion muy incorrecta)
- Penaliza fuertemente las predicciones erroneas con alta confianza

### 5.4 Gradient Descent (Descenso de Gradiente)

El Gradient Descent es el algoritmo de optimizacion que ajusta los pesos para minimizar la funcion de coste:

```
w_nuevo = w_viejo - learning_rate * dJ/dw
b_nuevo = b_viejo - learning_rate * dJ/db
```

Donde:
- `dJ/dw` = derivada parcial del coste respecto a los pesos (gradiente)
- `dJ/db` = derivada parcial del coste respecto al sesgo
- `learning_rate` = tamano del paso en cada iteracion

**Proceso iterativo:**
1. Inicializar pesos en 0.01 y sesgo en 0
2. Calcular predicciones con pesos actuales
3. Calcular el coste
4. Calcular los gradientes (hacia donde ajustar)
5. Actualizar pesos en direccion opuesta al gradiente
6. Repetir hasta convergencia

**Termino tecnico - Learning Rate (Tasa de Aprendizaje):**
Controla el tamano del paso en cada iteracion del gradient descent.
- Si es muy grande (ej: 1.0): el modelo puede "pasarse" del minimo y no converger
- Si es muy pequeno (ej: 0.0001): el modelo tarda mucho en converger
- Valor tipico: 0.01 (el que usamos en este proyecto)
- Se puede visualizar en la grafica de convergencia del coste

**Termino tecnico - Learning Rate:**
Es como la velocidad a la que el modelo aprende. Un valor balanceado permite una convergencia rapida y estable.

### 5.5 Forward y Backward Propagation

**Forward Propagation (Propagacion hacia adelante):**
1. Calcular z = w^T * x + b (suma ponderada)
2. Calcular sigma(z) (probabilidad predicha)
3. Calcular el coste J

**Backward Propagation (Propagacion hacia atras):**
1. Calcular las derivadas parciales del coste respecto a w y b
2. Estos gradientes indican como ajustar los pesos para reducir el error
3. `dJ/dw = (1/m) * x * (sigma(z) - y)`
4. `dJ/db = (1/m) * SUMA(sigma(z) - y)`

### 5.6 Hiperparametros

Los hiperparametros son configuraciones que se ajustan ANTES del entrenamiento:

| Hiperparametro | Valor | Descripcion |
|----------------|-------|-------------|
| learning_rate | 0.01 | Tamano del paso en gradient descent |
| num_iterations | 1,500 | Numero de pasos del gradient descent |
| C (sklearn) | Optimizado via GridSearch | Inverso de la regularizacion |
| penalty (sklearn) | L1 o L2 | Tipo de regularizacion |
| solver (sklearn) | liblinear | Algoritmo de optimizacion |

### 5.7 Regularizacion

La regularizacion evita el **overfitting** penalizando pesos grandes:

- **L2 (Ridge):** Penaliza la suma de cuadrados de los pesos. `penalty = lambda * SUMA(w_i^2)`
  - Tiende a distribuir los pesos de forma uniforme
  - Es la opcion por defecto en sklearn

- **L1 (Lasso):** Penaliza la suma de valores absolutos. `penalty = lambda * SUMA(|w_i|)`
  - Puede llevar algunos pesos a exactamente 0 (seleccion automatica de features)
  - Util cuando se sospecha que muchas features no son relevantes

**Termino tecnico - Regularizacion:** Tecnica que agrega una penalizacion a la funcion de coste para evitar que los pesos crezcan demasiado. Esto previene que el modelo se ajuste al ruido (overfitting) y mejora su capacidad de generalizacion.

### 5.8 GridSearchCV (Validacion Cruzada)

```python
param_grid = {
    "C": np.logspace(-3, 3, 7),  # [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
}
grid_search = GridSearchCV(lr, param_grid, cv=10, scoring="accuracy")
```

**Que hace GridSearchCV?**
1. Prueba todas las combinaciones de C y penalty
2. Para cada combinacion, usa 10-fold Cross-Validation
3. Selecciona la combinacion con mejor accuracy
4. Retorne el modelo optimizado

**Termino tecnico - 10-Fold Cross-Validation:**
Divide el dataset de entrenamiento en 10 partes iguales. En cada iteracion:
- Entrena con 9 partes
- Valida con la parte restante
- Rotacion hasta que todas las partes hayan sido validacion
- El accuracy final es el promedio de las 10 iteraciones

**Beneficio:** Proporciona una estimacion mas robusta del rendimiento que un solo train/test split, ya que utiliza todos los datos tanto para entrenar como para validar.

### 5.9 Analisis de Coeficientes

En Regresion Logistica, cada feature (pixel) tiene un coeficiente que indica su influencia:

- **Coeficiente POSITIVO grande:** El pixel "encendido" (claro) favorece la clase 1 (Mujer)
- **Coeficiente NEGATIVO grande:** El pixel "encendido" favorece la clase 0 (Varon)
- **Coeficiente cercano a 0:** El pixel tiene poca influencia en la decision

Al reorganizar los 4,096 coeficientes en una imagen de 64x64, se puede visualizar que regiones del rostro son mas importantes para la clasificacion de genero.

**Termino tecnico - Coeficiente:** En regresion logistica, el coeficiente (peso) de cada feature indica cuanta influencia tiene esa feature en la decision del modelo. Un coeficiente positivo significa que valores altos de esa feature favorecen la clase positiva (Mujer). Un coeficiente negativo significa lo contrario.

### 5.10 Implementacion Manual vs Sklearn

Se implementaron dos versiones del modelo:

| Aspecto | Implementacion Manual | Scikit-learn |
|---------|----------------------|--------------|
| Librerias | Solo NumPy | Scikit-learn |
| Optimizacion | Gradient descent basico | Solvers avanzados (liblinear) |
| Regularizacion | No implementada | L1/L2 configurable |
| Velocidad | Mas lenta | Optimizada en C |
| Proposito | Educativo (entender las matematicas) | Produccion (maximo rendimiento) |

---

## 6. RESULTADOS Y DISCUSION DE METRICAS

### 6.1 Matriz de Confusión

La matriz de confusión muestra las predicciones del modelo comparadas con las etiquetas reales:

```
                    Predicho: Mujer    Predicho: Varon
Real: Mujer         [     TP    ]     [     FN    ]
Real: Varon         [     FP    ]     [     TN    ]
```

**Definiciones:**
- **TP (True Positive / Verdadero Positivo):** Mujer predicha correctamente como Mujer
- **TN (True Negative / Verdadero Negativo):** Varon predicho correctamente como Varon
- **FP (False Positive / Falso Positivo):** Varon predicho como Mujer (Error Tipo I)
- **FN (False Negative / Falso Negativo):** Mujer predicha como Varon (Error Tipo II)

**Interpretacion:**
- FP (falso positivo): El sistema "ve" una mujer donde hay un varon
- FN (falso negativo): El sistema "ve" un varon donde hay una mujer
- TP y TN son las predicciones correctas

### 6.2 Exactitud (Accuracy)

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Que mide:** Proporcion de predicciones correctas sobre el total.

**Interpretacion:** Si el accuracy es 0.85, significa que el 85% de las predicciones fueron correctas.

**Limitacion:** No es confiable cuando el dataset esta desbalanceado. Por ejemplo, si el 99% de las imagenes fueran de varones, un modelo que siempre prediga "varon" tendria 99% de accuracy pero 0% de utilidad.

### 6.3 Precision

```
Precision = TP / (TP + FP)
```

**Que mide:** De todos los que el modelo predijo como "Mujer", cuantos realmente lo eran.

**Interpretacion:** Alta precision = pocos falsos positivos. Es importante cuando el costo de un falso positivo es alto (ej: en un sistema de seguridad).

### 6.4 Sensibilidad / Recall (Exhaustividad)

```
Recall = TP / (TP + FN)
```

**Que mide:** De todos los que realmente eran "Mujer", cuantos logro detectar el modelo.

**Interpretacion:** Alto recall = pocos falsos negativos. Es importante cuando el costo de un falso negativo es alto (ej: en diagnostico medico, donde no detectar una enfermedad es grave).

### 6.5 Puntuacion F1 (F1-Score)

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Que mide:** Media armonica de Precision y Recall. Penaliza valores extremos.

**Interpretacion:** F1 alto = buen balance entre precision y recall. Es la metrica mas util cuando el dataset esta desbalanceado o cuando los falsos positivos y falsos negativos tienen costos similares.

**Termino tecnico - F1-Score:** A diferencia del promedio aritmetico, la media armonica penaliza mas cuando uno de los valores es bajo. Por ejemplo:
- Precision=1.0, Recall=0.5 -> Promedio = 0.75, F1 = 0.67
- Precision=0.9, Recall=0.9 -> Promedio = 0.90, F1 = 0.90

### 6.6 Curva ROC y AUC-ROC

**Curva ROC (Receiver Operating Characteristic):**
Grafica la Tasa de Verdaderos Positivos (TPR/Recall) vs la Tasa de Falsos Positivos (FPR) en distintos umbrales de decision.

```
TPR = TP / (TP + FN) = Recall
FPR = FP / (FP + TN)
```

**AUC (Area Under the Curve):**
Area bajo la curva ROC. Resume el rendimiento en un solo numero.

| AUC | Interpretacion |
|-----|----------------|
| 0.5 | Modelo aleatorio (no sirve) |
| 0.5 - 0.7 | Deficiente |
| 0.7 - 0.8 | Aceptable |
| 0.8 - 0.9 | Bueno |
| > 0.9 | Excelente |

**Termino tecnico - AUC-ROC:** Mide la capacidad del modelo de distinguir entre clases. Un AUC de 0.95 significa que hay un 95% de probabilidad de que el modelo asigne una probabilidad mas alta a una instancia positiva elegida aleatoriamente que a una negativa elegida aleatoriamente.

**Termino tecnico - Umbral de Decision:** El punto de corte para convertir la probabilidad en una decision de clase. Por defecto es 0.5: si sigma(z) >= 0.5 -> clase 1, si no -> clase 0. Se puede ajustar para priorizar precision o recall.

### 6.7 Comparacion con Clasificador Aleatorio

En el grafico ROC, la linea punteada diagonal representa un clasificador aleatorio (AUC = 0.5). Si la curva ROC del modelo esta por encima de esta diagonal, el modelo es mejor que el azar. Cuanto mas alejada de la diagonal y mas cercana a la esquina superior izquierda, mejor es el modelo.

### 6.8 Interpretacion Clinica/Operativa de Errores

**Falsos Positivos (FP):** El modelo clasifica a un varon como mujer. En un contexto de seguridad, esto podria causar que una persona no autorizada acceda a un sistema.

**Falsos Negativos (FN):** El modelo clasifica a una mujer como varon. En un contexto de marketing, esto podria causar que campañas targeteadas se dirijan al genero incorrecto.

**Trade-off Precision-Recall:** Existe un balance natural. Para reducir FP (aumentar precision), se puede subir el umbral de decision, pero esto incrementara FN (reducira recall). El F1-Score captura este balance.

---

## 7. MANUAL DE USUARIO Y ARQUITECTURA DEL APLICATIVO

### 7.1 Arquitectura del Sistema

```
[Usuario] 
    |
    v
[App Streamlit] -- Lee imagen subida
    |
    v
[Preprocesamiento] -- Grayscale -> 64x64 -> Flatten -> StandardScaler
    |
    v
[Modelo LogisticRegression] -- predict() + predict_proba()
    |
    v
[Resultado] -- Genero (Varon/Mujer) + Probabilidad
```

### 7.2 Flujo de Datos

1. **Entrada:** El usuario sube una imagen JPG/PNG via el widget `st.file_uploader`
2. **Preprocesamiento:** La imagen se convierte a escala de grises, se redimensiona a 64x64, se aplaná a 4,096 features, y se estandariza con StandardScaler
3. **Prediccion:** El modelo calcula sigma(z) y asigna la clase
4. **Salida:** Se muestra el genero predicho, la probabilidad, y una explicacion detallada

### 7.3 Como Ejecutar Localmente

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicacion
streamlit run app.py

# Abrir en el navegador: http://localhost:8501
```

### 7.4 Como Desplegar en Streamlit Community Cloud

1. Subir el codigo a un repositorio publico en GitHub
2. Ingresar a https://share.streamlit.io
3. Conectar el repositorio de GitHub
4. Seleccionar `app.py` como archivo principal
5. Hacer clic en "Deploy"
6. La app estara disponible en una URL publica

### 7.5 Requisitos para Despliegue

| Requisito | Estado |
|-----------|--------|
| `requirements.txt` en la raiz | Cumplido |
| `app.py` como punto de entrada | Cumplido |
| Modelo serializado (`.pkl`) en el repo | Cumplido |
| Dataset NO en el repo (demasiado grande) | Cumplido |
| Repositorio publico en GitHub | Pendiente |

---

## 8. CONCLUSIONES Y RECOMENDACIONES

### 8.1 Conclusiones

1. **La Regresion Logistica es efectiva como linea base:** Aunque no es el modelo mas avanzado para vision por computadora, logra resultados razonables para clasificacion binaria de genero.

2. **El preprocesamiento es critico:** La conversion a escala de grises, redimensionamiento y estandarizacion son pasos fundamentales que afectan directamente el rendimiento del modelo.

3. **La interpretabilidad es una ventaja:** A diferencia de redes neuronales profundas, la regresion logistica permite analizar los coeficientes para entender que regiones de la imagen son mas importantes.

4. **GridSearchCV mejora el rendimiento:** La busqueda sistematica de hiperparametros con validacion cruzada produjo un modelo mas robusto que la configuracion por defecto.

5. **El despliegue web es factible:** Streamlit permite crear aplicaciones interactivas de forma rapida, facilitando el acceso al modelo por parte de usuarios no tecnicos.

### 8.2 Limitaciones

1. **Accuracy limitado para imagenes:** La regresion logistica opera con pixeles individuales como features, lo que no captura relaciones espaciales entre pixeles (a diferencia de CNNs).

2. **Sensibilidad a iluminacion y pose:** El modelo puede fallar con imagenes que tengan iluminacion o angulo muy diferentes a las de entrenamiento.

3. **Tamano de imagen fijo:** Todas las imagenes deben redimensionarse a 64x64, lo que puede distorsionar rostros con diferentes proporciones.

### 8.3 Recomendaciones a Futuro

1. **Probar con CNNs (Redes Neuronales Convolucionales):** Capturan mejor las relaciones espaciales entre pixeles y generalmente obtienen mayor accuracy en tareas de vision.

2. **Aumentar el dataset:** Mas datos mejoran la capacidad de generalizacion del modelo.

3. **Data Augmentation:** Rotar, escalar, y ajustar brillo de las imagenes existentes para crear variaciones artificiales.

4. **Face Detection pre-procesamiento:** Usar detectores de rostro (Haar cascades, MTCNN) para recortar solo el rostro antes de clasificar.

5. **Multiclase:** Extender el modelo para clasificar mas categorias (edad, etnia, expresion facial).

6. **Transfer Learning:** Usar modelos pre-entrenados (MobileNet, VGG) como extraccion de features, y luego aplicar regresion logistica sobre esas features mas abstractas.

---

## 9. ANEXOS

### 9.1 Enlace al Repositorio
[Ingresar URL del repositorio GitHub despues de crearlo]

### 9.2 Enlace a la Aplicacion Desplegada
[Ingresar URL de Streamlit Community Cloud despues de desplegar]

### 9.3 Tecnologias Utilizadas

| Tecnologia | Version | Uso |
|------------|---------|-----|
| Python | 3.14 | Lenguaje de programacion |
| NumPy | 2.5.1 | Operaciones numericas |
| Pandas | 3.0.5 | Manipulacion de datos |
| OpenCV | 4.14.0 | Procesamiento de imagenes |
| Scikit-learn | 1.9.0 | Modelo de ML y metricas |
| Matplotlib | 3.11.1 | Visualizaciones |
| Seaborn | 0.13.2 | Visualizaciones estadisticas |
| Streamlit | 1.62.0 | Aplicacion web |
| Joblib | 1.5.3 | Serializacion de modelos |

### 9.4 Glosario de Terminos Tecnicos

| Termino | Definicion |
|---------|------------|
| **Accuracy** | Proporcion de predicciones correctas sobre el total |
| **AUC-ROC** | Area bajo la curva ROC; mide capacidad de discriminacion |
| **Backward Propagation** | Calculo de gradientes para actualizar pesos |
| **Bias (Sesgo)** | Termino independiente que ajusta el umbral de decision |
| **Class Imbalance** | Desbalance en la cantidad de ejemplos por clase |
| **Confusion Matrix** | Tabla que muestra TP, TN, FP, FN |
| **Cross-Validation** | Validacion cruzada; division多次 del dataset para evaluacion robusta |
| **Data Leakage** | Filtracion de informacion del test al entrenamiento |
| **F1-Score** | Media armonica de Precision y Recall |
| **Feature** | Variable de entrada al modelo (en este caso, cada pixel) |
| **Flatten** | Convertir una matriz 2D a un vector 1D |
| **Forward Propagation** | Calculo de predicciones con pesos actuales |
| **Gradient Descent** | Algoritmo de optimizacion que minimiza la funcion de coste |
| **GridSearchCV** | Busqueda sistematica de hiperparametros con validacion cruzada |
| **Hyperparameter** | Configuracion ajustada antes del entrenamiento |
| **Label** | Etiqueta de clase (0=Varon, 1=Mujer) |
| **Learning Rate** | Tamano del paso en gradient descent |
| **Log Loss** | Funcion de coste de regresion logistica |
| **Overfitting** | Modelo que memoriza entrenamiento pero falla con datos nuevos |
| **Precision** | De los predichos positivos, cuantos realmente lo son |
| **Recall** | De los positivos reales, cuantos detecto el modelo |
| **Regularizacion** | Penalizacion para evitar overfitting (L1, L2) |
| **Scaler** | Transformador que estandariza features |
| **Sigmoid** | Funcion sigma(z) = 1/(1+e^(-z)); convierte z en probabilidad |
| **StandardScaler** | Estandariza features a media=0, desviacion=1 |
| **Test Set** | Conjunto de datos para evaluacion final |
| **Train Set** | Conjunto de datos para entrenar el modelo |
| **Underfitting** | Modelo demasiado simple para capturar patrones |
| **Weight (Peso)** | Coeficiente que indica la influencia de cada feature |

### 9.5 Estructura del Repositorio

```
clasificacion-genero-regresion-logistica/
├── app.py                          # Aplicacion Streamlit
├── requirements.txt                # Dependencias
├── .gitignore                      # Archivos ignorados por Git
├── README.md                       # Documentacion del repositorio
├── INFORME_REGRESION_LOGISTICA.md  # Este documento
└── modelo/
    ├── logistic_model.pkl          # Modelo entrenado
    └── scaler.pkl                  # Estandarizador
```

---

**FIN DEL INFORME**
