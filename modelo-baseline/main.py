import sys
import os

# Add the 'src' directory to the Python path
# This is necessary so we can import the modules from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preparation import prepare_data
from train_model import train_model
from predict import predict

def main():
    """
    Main function to run the entire ML pipeline.
    1. Prepare the data.
    2. Train the model.
    3. Make predictions.
    """
    print("--- Starting ML Pipeline ---")

    # Step 1: Prepare the data
    print("\n--- Running Data Preparation ---")
    prepare_data()

    # Step 2: Train the model
    print("\n--- Running Training ---")
    train_model()

    # Step 3: Make predictions
    print("\n--- Running Prediction ---")
    predict()

    print("\n--- ML Pipeline Finished ---")

if __name__ == '__main__':
    main()
