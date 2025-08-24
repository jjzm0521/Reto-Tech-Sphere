# Reto-Tech-Sphere: Clasificación de Artículos Médicos

Este proyecto es una solución para el desafío "AI + Data Challenge – Tech Sphere 2025".

## Descripción del Desafío

El "AI + Data Challenge – Tech Sphere 2025" es un desafío de clasificación biomédica con inteligencia artificial. El objetivo es construir una solución de IA para clasificar artículos médicos en uno o varios de los siguientes dominios:
- Cardiovascular
- Neurológico
- Hepatorenal
- Oncológico

La clasificación se debe basar únicamente en el **título** y el **resumen** del artículo.

## Público Objetivo

El desafío está dirigido a:
- Perfiles junior y recién graduados en carreras STEM.
- Profesionales en transición hacia la ciencia de datos.
- Desarrolladores incursionando en el campo de la inteligencia artificial.

## Stack Tecnológico Recomendado

- **Lenguaje:** Python
- **Librerías Principales:**
  - `pandas` para manipulación de datos.
  - `scikit-learn` para modelos de machine learning baseline.
  - `spacy` / `scispacy` para procesamiento de lenguaje natural (NLP).
  - `torch` y `transformers` para modelos avanzados de deep learning (BERT).

## Dataset

El dataset `challenge_data-18-ago.csv` contiene los datos para el entrenamiento y la evaluación de los modelos. Cada fila representa un artículo médico con su título, resumen y las categorías a las que pertenece.
