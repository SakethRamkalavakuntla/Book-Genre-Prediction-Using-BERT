

import pandas as pd

# Load the cleaned dataset
books = pd.read_csv('BooksDataSet.csv', on_bad_lines='skip', quoting=1, encoding='utf-8')

# Check the shape
print("Shape:", books.shape)

# Preview the data
print("\nSample Data:\n", books.head())

# Check for missing values
print("\nMissing values:\n", books.isnull().sum())

# Check class distribution
print("\nGenre distribution:\n", books['genre'].value_counts())

# Check summary lengths
books['summary_length'] = books['summary'].astype(str).apply(len)
print("\nSummary length stats:\n", books['summary_length'].describe())

# Optional: Histogram of summary lengths
import matplotlib.pyplot as plt
books['summary_length'].hist(bins=30)
plt.title("Distribution of Summary Lengths")
plt.xlabel("Length of Summary")
plt.ylabel("Number of Books")
plt.show()

# !pip install -U adapter-transformers
# !pip install datasets

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

books = pd.read_csv("BooksDataSet.csv")
books = books[['book_id', 'book_name', 'genre', 'summary']]

books = books[books['summary'].str.len() > 30].reset_index(drop=True)

label_encoder = LabelEncoder()
books['label'] = label_encoder.fit_transform(books['genre'])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

class BookSummaryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),  # [seq_len]
            'attention_mask': encoding['attention_mask'].squeeze(),  # [seq_len]
            'label': torch.tensor(label, dtype=torch.long)
        }

dataset = BookSummaryDataset(
    texts=books['summary'].tolist(),
    labels=books['label'].tolist(),
    tokenizer=tokenizer
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

from torch.optim import AdamW

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.optim import AdamW  # ✅ Fixed import

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels=6  # 6 genres
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

from sklearn.model_selection import train_test_split

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    books['summary'].tolist(),
    books['label'].tolist(),
    test_size=0.2,
    stratify=books['label'],
    random_state=42
)

# Use the same BookSummaryDataset class as before
train_dataset = BookSummaryDataset(train_texts, train_labels, tokenizer)
val_dataset = BookSummaryDataset(val_texts, val_labels, tokenizer)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

import torch
from tqdm import tqdm
import torch.nn.functional as F

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=3):
    model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()

        total_train_loss = 0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}]")
        for batch in train_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)

            train_accuracy = correct_train / total_train
            train_loop.set_postfix(loss=loss.item(), accuracy=train_accuracy)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # ------------------- VALIDATION STEP -------------------
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

train_model(model, train_loader, val_loader, optimizer, device, num_epochs=3)

import torch
from tqdm import tqdm
import torch.nn.functional as F
import os

def train_model_with_early_stopping(model, train_loader, val_loader, optimizer, device, num_epochs=10, patience=2, save_path='best_model.pt'):
    model.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()

        total_train_loss = 0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}]")
        for batch in train_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)

            train_accuracy = correct_train / total_train
            train_loop.set_postfix(loss=loss.item(), accuracy=train_accuracy)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # ------------------- VALIDATION STEP -------------------
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # ------------------- EARLY STOPPING CHECK -------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Validation loss improved. Model saved to: {save_path}")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("⛔ Early stopping triggered.")
            break

    print("Training complete.")

train_model_with_early_stopping(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=10,
    patience=2,
    save_path='best_bert_model.pt'
)

import pickle

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("LabelEncoder saved as label_encoder.pkl")

from google.colab import files

# Download model file
files.download("best_bert_model.pt")

from google.colab import files
import zipfile
import os

# Optional: Verify the file exists
assert os.path.exists("best_bert_model.pt"), "Model file not found."

# Step 1: Zip the model file
with zipfile.ZipFile("best_bert_model.zip", "w") as zipf:
    zipf.write("best_bert_model.pt")

# Step 2: Download the zipped model
files.download("best_bert_model.zip")


# # Step 3: Download the label encoder
# files.download("label_encoder.pkl")