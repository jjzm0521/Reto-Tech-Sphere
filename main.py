import train
import src.evaluate
import os

def main():
    """
    Main function to run the entire ML pipeline.
    1. Train the model.
    2. Evaluate the model.
    """
    print("--- Starting ML Pipeline ---")

    # Step 1: Train the model
    print("\n--- Running Training ---")
    train.train_baseline()

    # Step 2: Evaluate the model
    print("\n--- Running Evaluation ---")
    # Check if model and test data exist before evaluating
    if os.path.exists('baseline_model.joblib') and os.path.exists('test_data.joblib'):
        src.evaluate.evaluate_model()
    else:
        print("Evaluation skipped because model or test data was not found.")

    print("\n--- ML Pipeline Finished ---")

if __name__ == '__main__':
    main()
