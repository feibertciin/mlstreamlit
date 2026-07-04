# Proyecto Predictivo de Engagement de Usuarios - CRISP-ML(Q)

Este proyecto implementa un modelo de Machine Learning supervisado para predecir la puntuación de engagement (`score`) de usuarios a partir de variables demográficas y espaciales. El proyecto sigue rigurosamente el proceso metodológico **CRISP-ML(Q)**.

## 📂 Estructura del Proyecto

- `ETL_+_EDA_Estu.ipynb`: Cuaderno Jupyter con las fases de Comprensión de Negocio, Comprensión de Datos, Preparación de Datos, Modelado y Evaluación.
- `app.py`: Aplicación web interactiva desarrollada en **Streamlit** para realizar predicciones.
- `index.html`: Landing Page con diseño premium (dark mode y glassmorphism) que explica el proyecto y su ciclo de vida.
- `sampledata.csv`: Dataset original con la información de los usuarios.
- `model.joblib`: Pipeline del modelo entrenado y serializado.
- `ar_dashboard.png`: Imagen de Realidad Aumentada generada por Inteligencia Artificial para la Landing Page.
- `feature_importance.png`: Gráfico que muestra la importancia de variables del modelo de Random Forest.

## 🛠️ Requisitos de Instalación

Para ejecutar la aplicación localmente, asegúrate de tener instalado Python 3.10 o superior y las dependencias correspondientes:

```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
```

## 🚀 Instrucciones de Ejecución

### 1. Ejecutar la Aplicación Streamlit
Para levantar el servidor interactivo de Streamlit, ejecuta en la consola:

```bash
streamlit run app.py
```

Esto abrirá la aplicación interactiva de manera automática en tu navegador (por defecto en `http://localhost:8501`).

### 2. Abrir la Landing Page
Puedes abrir directamente el archivo `index.html` en cualquier navegador web moderno de manera local, o servirlo en un servidor web estático.

## ⚙️ Ciclo de Vida Metodológico (CRISP-ML)

1. **Comprensión del Negocio:** Definición del objetivo del modelo: predecir el score de engagement para priorizar esfuerzos de marketing.
2. **Comprensión de Datos:** Análisis de distribución geográfica (lat/lon), edad y país de procedencia en `sampledata.csv`.
3. **Preparación de Datos:** Imputación robusta de valores nulos (mediana para edad, media para coordenadas) y normalización.
4. **Modelado:** Creación de un pipeline con `ColumnTransformer` (OneHotEncoder + StandardScaler) y entrenamiento de **Random Forest Regressor** optimizado con `GridSearchCV`.
5. **Evaluación:** Comparación con un modelo baseline de **Regresión Lineal**, evaluando con métricas de $R^2$, RMSE y MAE.
6. **Despliegue:** Serialización del modelo final a `model.joblib` para consumo en producción desde la aplicación de Streamlit.
