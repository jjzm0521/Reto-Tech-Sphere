import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report, hamming_loss
import os

# --- Configuration ---
MODEL_NAME = 'dmis-lab/biobert-base-cased-v1.1'
MODEL_PATH = 'biobert_model.bin'
TEST_DATA_PATH = 'bert_test_data.pkl'
DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
MAX_LEN = 256
BATCH_SIZE = 8 # Can be larger for evaluation

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- PyTorch Dataset Class (copied from train_bert.py) ---
class MedicalArticleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

# --- PyTorch Model Class (copied from train_bert.py) ---
class BiobertClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(BiobertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

# --- Evaluation Function ---
def get_predictions(model, data_loader, device):
    model = model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            fin_targets.extend(labels.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return np.array(fin_outputs), np.array(fin_targets)

# --- Main Evaluation Execution ---
def evaluate_bert_model():
    # 1. Check for necessary files
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Model file ('{MODEL_PATH}') or test data file ('{TEST_DATA_PATH}') not found.")
        print("Please run the BERT training script (`src/train_bert.py`) first.")
        return

    # 2. Load test data
    print(f"Loading test data from {TEST_DATA_PATH}...")
    df_test = pd.read_pickle(TEST_DATA_PATH)

    # 3. Setup Tokenizer and DataLoader
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = MedicalArticleDataset(
        texts=df_test.text.to_numpy(),
        labels=df_test[DOMAINS].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

    # 4. Load trained model
    print(f"Loading trained model from {MODEL_PATH}...")
    model = BiobertClassifier(n_classes=len(DOMAINS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)

    # 5. Get predictions
    print("Making predictions on the test set...")
    outputs, targets = get_predictions(model, test_data_loader, device)

    # Apply a 0.5 threshold to get binary predictions
    preds = outputs >= 0.5

    # 6. Calculate and Print Metrics
    print("\n--- BERT Model Evaluation Results ---")
    f1_weighted = f1_score(targets, preds, average='weighted')
    hamming = hamming_loss(targets, preds)

    print(f"\nWeighted F1-Score: {f1_weighted:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print("(Lower is better. It's the fraction of labels that are incorrectly predicted.)")

    print("\nClassification Report (per-class performance):")
    report = classification_report(targets, preds, target_names=DOMAINS, zero_division=0)
    print(report)
    print("--- Evaluation script finished. ---")


if __name__ == '__main__':
    evaluate_bert_model()
