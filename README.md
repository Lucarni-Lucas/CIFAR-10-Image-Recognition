# CIFAR-10 Image Classifier

Este proyecto es un clasificador de imágenes basado en el conjunto de datos CIFAR-10, que contiene 60,000 imágenes de 32x32 píxeles en 10 categorías diferentes. El modelo utiliza una red neuronal convolucional (CNN) para realizar la clasificación.

## Requisitos

1. Python 3.8 o superior.
2. Instalación de las dependencias del archivo `requirements.txt`.


## Estructura del proyecto

- `src/`: Contiene el código fuente del proyecto.
- `data/`: Carpeta para almacenar el dataset CIFAR-10.
- `notebooks/`: Incluye análisis exploratorio y experimentos en Jupyter Notebook.
- `results/`: Contiene el modelo entrenado y gráficos de entrenamiento.

## Cómo ejecutar el proyecto

1. Instala las dependencias:
    ```bash
    pip install -r requirements.txt

2. Asegúrate de que el dataset CIFAR-10 esté descargado en la carpeta data/.

3. Entrena el modelo:
    ```bash
    python src/train.py

4. Evalúa el modelo:
    ```bash
    python src/evaluate.py

## Resultados obtenidos
• Accuracy final en entrenamiento: ~0.75 - ~0.78

• Accuracy en el conjunto de prueba: ~0.74 - ~0.77




