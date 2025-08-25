import pandas as pd
import os
import joblib

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (src -> modelo-baseline)
project_root = os.path.dirname(script_dir)

model_path = os.path.join(project_root, 'results', 'models', 'baseline_model.joblib')

def predict():
    """
    Loads the trained model and makes a prediction on new data.
    """
    print("Making predictions...")

    # Load the model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = joblib.load(model_path)

    # Create some new data for prediction
    new_data = pd.DataFrame({'col1': [4, 5, 6]})

    # Make predictions
    predictions = model.predict(new_data)

    print("Predictions:", predictions)
    return predictions

if __name__ == '__main__':
    predict()
