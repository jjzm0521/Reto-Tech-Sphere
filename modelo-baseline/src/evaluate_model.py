import joblib
from sklearn.metrics import (f1_score, classification_report,
                             precision_score, recall_score, accuracy_score,
                             multilabel_confusion_matrix)
import pandas as pd
import os

# --- Configuration ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (src -> modelo-baseline)
project_root = os.path.dirname(script_dir)

MODEL_FILE = os.path.join(project_root, 'results', 'models', 'baseline_model.joblib')
TEST_DATA_FILE = os.path.join(project_root, 'results', 'models', 'test_data.joblib')
DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

# --- Main Evaluation Script ---
def evaluate_model():
    """
    Loads a trained model and test data, evaluates the model,
    and prints a formatted text report of the performance metrics.
    """
    # 1. Load Model and Data
    print("--- Loading Model and Data ---")
    try:
        pipeline = joblib.load(MODEL_FILE)
        test_data = joblib.load(TEST_DATA_FILE)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        print("Model and data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please run the training script (train_model.py) first to generate the model and test data.")
        return

    # 2. Make Predictions
    print("\n--- Making Predictions ---")
    y_pred = pipeline.predict(X_test)
    print("Predictions made on the test set.")

    # 3. Calculate All Metrics
    print("\n--- Calculating Metrics ---")
    # Global Metrics
    global_metrics = {
        "f1_score_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "subset_accuracy": accuracy_score(y_test, y_pred)
    }

    # Per-Category Metrics from classification_report
    class_report = classification_report(y_test, y_pred, target_names=DOMAINS, zero_division=0, output_dict=True)
    category_metrics = {k: v for k, v in class_report.items() if k in DOMAINS}

    # Confusion Matrices
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    confusion_matrices = {domain: mcm[i] for i, domain in enumerate(DOMAINS)}
    print("All metrics calculated.")

    # 4. Generate and Print Text Report
    print("\n--- Generating Final Report ---")
    text_report = generate_text_report(global_metrics, category_metrics, confusion_matrices)
    print("\n" + "="*80)
    print(text_report)
    print("="*80 + "\n")

    print("--- Evaluation script finished. ---")


def generate_text_report(global_metrics, category_metrics, confusion_matrices):
    """
    Generates a formatted text block with model performance metrics.
    """
    # Title
    report = "Análisis de Rendimiento del Modelo de Clasificación Biomédica\n\n"

    # Global Metrics Section
    report += "Métricas Globales\n"
    report += "-"*20 + "\n"
    report += f"- F1-Score Ponderado: {global_metrics['f1_score_weighted']:.3f}\n"
    report += f"- Precisión Ponderada: {global_metrics['precision_weighted']:.3f}\n"
    report += f"- Recall Ponderado: {global_metrics['recall_weighted']:.3f}\n"
    report += f"- Exactitud de Subconjunto: {global_metrics['subset_accuracy']:.3f}\n\n"

    # Per-Category Performance Section
    report += "Rendimiento por Categoría\n"
    report += "-"*28 + "\n"
    # Create a formatted table header
    header = f"{'Categoría':<15} | {'Precisión':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Soporte':>10}\n"
    report += header
    report += "-"*len(header) + "\n"
    # Create table rows
    for domain, metrics in category_metrics.items():
        report += (f"{domain:<15} | {metrics['precision']:>10.3f} | "
                   f"{metrics['recall']:>10.3f} | {metrics['f1-score']:>10.3f} | "
                   f"{metrics['support']:>10.0f}\n")
    report += "\n"

    # Confusion Matrices Section
    report += "Matrices de Confusión\n"
    report += "-"*24 + "\n"
    for domain, matrix in confusion_matrices.items():
        tn, fp, fn, tp = matrix.ravel()
        report += f"- Matriz para {domain}:\n"
        report += f"  [TN: {tn}, FP: {fp}]\n"
        report += f"  [FN: {fn}, TP: {tp}]\n\n"

    return report.strip()


if __name__ == '__main__':
    evaluate_model()
