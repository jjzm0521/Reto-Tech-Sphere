import pandas as pd
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (src -> modelo-baseline)
project_root = os.path.dirname(script_dir)

raw_data_path = os.path.join(project_root, 'data', 'raw', 'raw_data.csv')
processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')

# Create directories if they don't exist
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

def prepare_data():
    """
    Reads raw data, prepares it, and saves it to the processed data folder.
    """
    print("Preparing data...")

    # Read raw data
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at {raw_data_path}")
        # Create dummy raw data file if it doesn't exist
        print("Creating dummy raw data file...")
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        dummy_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        dummy_df.to_csv(raw_data_path, index=False)

    df = pd.read_csv(raw_data_path)

    # Simple data preparation example: add a new column
    df['new_col'] = df['col1'] * 2

    # Save processed data
    df.to_csv(processed_data_path, index=False)

    print(f"Data prepared and saved to {processed_data_path}")

if __name__ == '__main__':
    prepare_data()
