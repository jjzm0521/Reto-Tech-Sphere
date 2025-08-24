# ==============================================================================
# CELDA 1: INSTALACIÓN, CARGA Y PREPARACIÓN DE DATOS (ADAPTADA)
# ==============================================================================

# 1. IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os

# 2. CARGA Y PREPARACIÓN DE LOS DATOS
PROCESSED_DATA_PATH = "processed_data.pkl"

try:
    df = pd.read_pickle(PROCESSED_DATA_PATH)
    print(f"\nDataset preprocesado cargado exitosamente desde: '{PROCESSED_DATA_PATH}'")
except FileNotFoundError:
    print(f"\nError: No se encontró el archivo en la ruta '{PROCESSED_DATA_PATH}'.")
    print("Por favor, ejecuta primero el script `src/train.py` para generar el archivo de datos preprocesado.")
    exit()

# 3. CONFIGURACIÓN Y PREPROCESAMIENTO
TEXT_COLUMN = 'text'
LABEL_COLUMNS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
SMOKE_TEST = True # Poner en False para entrenar con todos los datos

# El DataFrame ya tiene las columnas de etiquetas binarizadas
# Crear la columna 'labels' en el formato que espera el Trainer
df['labels'] = df[LABEL_COLUMNS].values.tolist()
df_model = df[[TEXT_COLUMN, 'labels']].copy()

# Dividir los datos en conjuntos de entrenamiento, validación y prueba (80%, 10%, 10%)
train_val_df, test_df = train_test_split(
    df_model,
    test_size=0.1,
    random_state=42
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.111,  # 0.1 / 0.9 = 0.111...
    random_state=42
)

if SMOKE_TEST:
    print("\n--- SMOKE TEST ACTIVADO ---")
    print("Usando un subconjunto de los datos y menos épocas para una ejecución rápida.")
    train_df = train_df.head(80)
    val_df = val_df.head(20)
    test_df = test_df.head(20)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print(f"\nTamaño del conjunto de entrenamiento: {len(train_dataset)}")
print(f"Tamaño del conjunto de validación:   {len(val_dataset)}")
print(f"Tamaño del conjunto de prueba:        {len(test_dataset)}")
print("\n¡Preparación de datos finalizada!")


# ==============================================================================
# CELDA 2: CONFIGURACIÓN DEL EXPERIMENTO, MODELO Y TOKENIZADOR
# ==============================================================================

# Modelo seleccionado: SciBERT (demostró mejor rendimiento)
MODEL_NAME = 'allenai/scibert_scivocab_cased'
EXPERIMENT_NAME = "scibert-finetune-cased"

print(f"\n--- Iniciando experimento: {EXPERIMENT_NAME} ---")
print(f"Modelo base seleccionado: {MODEL_NAME}")

# 1. CARGA DEL TOKENIZADOR
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples[TEXT_COLUMN], padding="max_length", truncation=True, max_length=512)

# 2. TOKENIZACIÓN DE LOS DATASETS
print("\nTokenizando los datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# 3. FORMATEO FINAL DE ETIQUETAS
def format_labels(dataset):
    labels = [np.array(label, dtype=np.float32) for label in dataset['labels']]
    dataset = dataset.remove_columns('labels')
    dataset = dataset.add_column('labels', labels)
    return dataset

tokenized_train_dataset = format_labels(tokenized_train_dataset)
tokenized_val_dataset = format_labels(tokenized_val_dataset)
tokenized_test_dataset = format_labels(tokenized_test_dataset)
print("¡Tokenización completa!")

# 4. CARGA DEL MODELO PRE-ENTRENADO
print(f"\nCargando modelo pre-entrenado '{MODEL_NAME}'...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_COLUMNS),
    problem_type="multi_label_classification"
)
print("\n¡Modelo y tokenizador listos para el entrenamiento!")


# ==============================================================================
# CELDA 3: ENTRENAMIENTO, EVALUACIÓN Y GUARDADO
# ==============================================================================

# 1. DEFINICIÓN DE MÉTRICAS
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    y_true = p.label_ids
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_true, probs, average='weighted')
    return {
        'f1_weighted': f1_weighted,
        'roc_auc_weighted': roc_auc
    }

# 2. CONFIGURACIÓN DEL ENTRENAMIENTO
OUTPUT_DIR = f"./{EXPERIMENT_NAME}-results"
FINAL_MODEL_PATH = f"./{EXPERIMENT_NAME}-final-model"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Reducido para ahorrar memoria
    per_device_eval_batch_size=2,   # Reducido para ahorrar memoria
    gradient_accumulation_steps=4,  # Acumular gradientes para simular un batch size de 8
    num_train_epochs=1 if SMOKE_TEST else 3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    save_total_limit=1,
    report_to="none",
)

# 3. CREACIÓN DEL OBJETO TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 4. ENTRENAMIENTO
print(f"\nIniciando el fine-tuning del modelo: {EXPERIMENT_NAME}...")
trainer.train()
print("¡Entrenamiento completado!")

# 5. EVALUACIÓN FINAL SOBRE EL CONJUNTO DE PRUEBA
print("\n--- Evaluación Final ---")
print("Evaluando el rendimiento del MEJOR modelo sobre el conjunto de PRUEBA...")
test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)

print("\nResultados Finales en el Conjunto de Prueba:")
print(test_results)

# 6. GUARDADO PERMANENTE DEL MODELO FINAL
print(f"\nGuardando el modelo final y el tokenizador en '{FINAL_MODEL_PATH}'...")
trainer.save_model(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)
print(f"¡Proceso completado! Modelo guardado en '{FINAL_MODEL_PATH}'.")
