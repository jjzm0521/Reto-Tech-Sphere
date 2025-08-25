import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (src -> modelo-baseline)
project_root = os.path.dirname(script_dir)

processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')
model_path = os.path.join(project_root, 'results', 'models', 'baseline_model.joblib')

# Create directories if they don't exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)

def train_model():
    """
    Reads processed data, trains a simple model, and saves it.
    """
    print("Training model...")

    # Read processed data
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data file not found at {processed_data_path}")
        return

    df = pd.read_csv(processed_data_path)

    # Simple model training example
    X = df[['col1']]
    y = df['col2']

    model = LogisticRegression()
    model.fit(X, y)

    # Save the model
    joblib.dump(model, model_path)

    print(f"Model trained and saved to {model_path}")

if __name__ == '__main__':
    train_model()
