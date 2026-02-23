# ===============================
# 1║ Mount Google Drive
# ===============================
from google.colab import drive
drive.mount('/content/drive')


# ===============================
# 2║ Imports
# ===============================
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold # Updated import
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===============================
# 3║ Load Metadata
# ===============================
base_path = "/content/drive/MyDrive/schizophrenia_dataset"

metadata_path = os.path.join(base_path, "Metadata.xlsx")
metadata = pd.read_excel(metadata_path)

print("Metadata columns:", metadata.columns)


# Keep only required columns
metadata = metadata[['Participant ID', 'diagnosis']]
metadata = metadata.dropna()

# Clean strings
metadata['diagnosis'] = metadata['diagnosis'].astype(str).str.strip().str.lower()

# Remove < and > from Participant ID
metadata['Participant ID'] = metadata['Participant ID'].str.replace('<', '', regex=False)
metadata['Participant ID'] = metadata['Participant ID'].str.replace('>', '', regex=False)

print("Unique diagnoses:")
print(metadata['diagnosis'].unique())

# Keep schizophrenia and none reported
metadata = metadata[
    metadata['diagnosis'].str.contains('schizophrenia') |
    metadata['diagnosis'].str.contains('none reported')
]

# Convert to binary labels
metadata['label'] = metadata['diagnosis'].apply(
    lambda x: 1 if 'schizophrenia' in x else 0
)

print("Total valid participants:", len(metadata))
print("\nFirst 5 Participant IDs from metadata:")
print(metadata['Participant ID'].head())

# Define cl_path and co_path before printing os.listdir(cl_path)
cl_path = os.path.join(base_path, "Speaker_Only_Raw_CL")
co_path = os.path.join(base_path, "Speaker_Only_Raw_CO")

print("\nFirst 5 files in CL folder:")
print(os.listdir(cl_path)[:5])




# ===============================
# 4║ Load Transcript Files
# ===============================
print("First 5 transcript files in CL:")
print(os.listdir(cl_path)[:5])

texts = []
labels = []

for _, row in metadata.iterrows():
    participant_id = row['Participant ID']
    label = row['label']

    filename = participant_id + "_Raw.txt"

    file_path = os.path.join(cl_path, filename)

    if not os.path.exists(file_path):
        file_path = os.path.join(co_path, filename)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            texts.append(text)
            labels.append(label)

print("Total loaded transcripts:", len(texts))


# ===============================
# Tokenization Function (moved here for scope)
# ===============================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

# ===============================
# Dual Linguistic Model Definition (Updated)
# ===============================
class DualLinguisticModel(nn.Module):
    def __init__(self, bert_model, electra_model):
        super().__init__()
        self.bert = bert_model
        self.electra = electra_model

        # Classifier takes concatenated output of BERT and ELECTRA (768 + 768 = 1536)
        self.classifier = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        # Removed torch.no_grad() to allow fine-tuning of unfrozen layers
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        electra_outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        bert_cls_embedding = bert_outputs.last_hidden_state[:, 0, :]
        electra_cls_embedding = electra_outputs.last_hidden_state[:, 0, :]

        # Concatenate the embeddings
        combined_embedding = torch.cat((bert_cls_embedding, electra_cls_embedding), dim=1)

        return self.classifier(combined_embedding)


# ===============================
# Cross-Validation Setup (Updated for Stratified K-Fold)
# ===============================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Replaced LeaveOneOut with StratifiedKFold

all_accuracies = []

texts_np = np.array(texts)
labels_np = np.array(labels)

for fold_idx, (train_index, test_index) in enumerate(skf.split(texts_np, labels_np)):

    print(f"\nFold {fold_idx+1}/{skf.get_n_splits()}")

    train_texts, test_texts = texts_np[train_index], texts_np[test_index]
    train_labels, test_labels = labels_np[train_index], labels_np[test_index]

    # Create HF dataset
    train_dataset = Dataset.from_dict({
        "text": list(train_texts),
        "label": list(train_labels)
    })

    test_dataset = Dataset.from_dict({
        "text": list(test_texts),
        "label": list(test_labels)
    })

    # Tokenize
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Reinitialize model EVERY fold
    bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
    electra = AutoModel.from_pretrained("google/electra-base-discriminator").to(device)

    for param in bert.parameters():
        param.requires_grad = False
    for param in electra.parameters():
        param.requires_grad = False

    # Unfreeze layers 4-11 (updated fine-tuning)
    for i in range(4, 12): # Changed from 6 to 4
        for param in bert.encoder.layer[i].parameters():
            param.requires_grad = True
        for param in electra.encoder.layer[i].parameters():
            param.requires_grad = True

    model = DualLinguisticModel(bert, electra).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "bert" in n and p.requires_grad],
            "lr": 3e-5
        },
        {
            "params": [p for n, p in model.named_parameters() if "electra" in n and p.requires_grad],
            "lr": 3e-5
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": 1e-3
        }
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters)

    # Train for few epochs (keep small to reduce overfitting)
    for epoch in range(8): # Changed from 4 to 8
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

    # Test on single subject
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

    fold_accuracy = correct / total
    all_accuracies.append(fold_accuracy)

    print("Fold Accuracy:", fold_accuracy)


print("\nFinal Stratified 5-Fold CV Accuracy (Improved Model):", np.mean(all_accuracies))

# Visualize the new performance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(all_accuracies) + 1), all_accuracies, color='skyblue')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('Stratified 5-Fold Cross-Validation Accuracies (Improved Model)')
plt.ylim(0, 1)
plt.xticks(range(1, len(all_accuracies) + 1))
plt.grid(axis='y', linestyle='--')
plt.show()
