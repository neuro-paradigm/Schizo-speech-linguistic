# ===============================
# 1⍡ Mount Google Drive
# ===============================
from google.colab import drive
drive.mount('/content/drive')


# ===============================
# 2⍡ Imports
# ===============================
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===============================
# 3⍡ Load Metadata
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

print("\nFirst 5 files in CL folder:")
print(os.listdir(cl_path)[:5])




# ===============================
# 4⍡ Load Transcript Files
# ===============================
cl_path = os.path.join(base_path, "Speaker_Only_Raw_CL")
co_path = os.path.join(base_path, "Speaker_Only_Raw_CO")
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


from sklearn.model_selection import LeaveOneOut
import numpy as np

loo = LeaveOneOut()

all_accuracies = []

texts = np.array(texts)
labels = np.array(labels)

for fold, (train_index, test_index) in enumerate(loo.split(texts)):

    print(f"\nFold {fold+1}/{len(texts)}")

    train_texts, test_texts = texts[train_index], texts[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

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

    for i in range(8, 12):
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
    for epoch in range(4):
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


print("\nFinal LOOCV Accuracy:", np.mean(all_accuracies))
