# Modelo Baseline

Este es un proyecto de data science para demostrar una estructura de proyecto bien organizada.

## Estructura del Proyecto

```
modelo-baseline/
│
├── .gitignore          # Archivos a ignorar por git
├── README.md           # Documentación del proyecto
├── requirements.txt    # Dependencias de Python
│
├── data/
│   ├── raw/            # Datos originales
│   └── processed/      # Datos procesados
│
├── src/
│   ├── data_preparation.py # Scripts para preparar los datos
│   ├── train_model.py      # Script para entrenar el modelo
│   └── predict.py          # Script para hacer predicciones
│
└── results/
    ├── figures/          # Gráficos y visualizaciones
    └── models/           # Modelos entrenados
```

## Uso

1.  Instalar dependencias: `pip install -r requirements.txt`
2.  Ejecutar la preparación de datos: `python src/data_preparation.py`
3.  Entrenar un modelo: `python src/train_model.py`
4.  Realizar predicciones: `python src/predict.py`

También se puede ejecutar todo el pipeline desde el `main.py` ubicado en la raíz del proyecto `modelo-baseline`.

`python main.py`
