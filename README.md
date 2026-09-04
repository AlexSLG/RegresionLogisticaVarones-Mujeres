# Clasificacion de Genero con Regresion Logistica

Clasificacion binaria de genero (Varon / Mujer) a partir de imagenes faciales usando Regresion Logistica.

## Despliegue

Esta aplicacion esta desplegada en **Streamlit Community Cloud**.

**URL de la aplicacion:** [Ingresar URL despues de desplegar]

## Resultados del modelo

| Metrica | Valor |
|---------|-------|
| Accuracy | 89.00% |
| Precision | 89.00% |
| Recall | 89.00% |
| F1-Score | 89.00% |
| AUC-ROC | 93.71% |

- **Mejores hiperparametros:** C=0.001, penalty=l2, solver=liblinear
- **Dataset de entrenamiento:** 3,491 imagenes (1,747 mujeres, 1,744 varones)
- **Dataset de prueba:** 200 imagenes (100 de cada clase)

## Como ejecutar localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU-USUARIO/clasificacion-genero-regresion-logistica.git
cd clasificacion-genero-regresion-logistica

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicacion
streamlit run app.py
```

## Estructura del repositorio

```
.
├── app.py                          # Aplicacion Streamlit
├── requirements.txt                # Dependencias
├── modelo/
│   ├── logistic_model.pkl          # Modelo entrenado
│   └── scaler.pkl                  # Estandarizador
└── INFORME_REGRESION_LOGISTICA.md  # Documentacion para informe PDF
```

## Tecnologias

- **Python 3.14**
- **Scikit-learn** - Regresion Logistica, metricas, preprocesamiento
- **OpenCV** - Procesamiento de imagenes
- **Streamlit** - Interfaz web interactiva
- **NumPy** - Operaciones numericas
- **Matplotlib / Seaborn** - Visualizaciones
