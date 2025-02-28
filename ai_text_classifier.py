import pandas as pd
import numpy as np
import os
import re
import string
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['text'] = df['text'].apply(preprocess_text)
    return df['text'].values, df['label'].values

def create_data_loaders(X_train, y_train, X_val, y_val, tokenizer, max_len, batch_size):
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_len)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def train_model(model, train_loader, optimizer, device):
    model = model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, device):
    model = model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels

def main(data_file, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = model.to(device)

    texts, labels = load_data(data_file)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, tokenizer, max_len=128, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        print(f'Epoch {epoch + 1}/{3}')
        avg_train_loss = train_model(model, train_loader, optimizer, device)
        print(f'Train loss: {avg_train_loss}')

        predictions, true_labels = evaluate_model(model, val_loader, device)
        print(classification_report(true_labels, predictions))

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__ == "__main__":
    data_path = 'data/text_data.csv'  # Update path accordingly
    save_path = 'model/bert_text_classifier'
    main(data_path, save_path)