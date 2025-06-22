import json
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import os

BUCKET_PATH = "gs://jailbreak_dataset/"

######################################################
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = str(self.data.loc[index, 'prompt'])
        label = int(self.data.loc[index, 'label'])

        encoding = self.tokenizer.encode_plus(
            prompt,
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
            'labels': torch.tensor(label, dtype=torch.long)
        }


######################################################
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    correct_predictions = 0
    all_labels = []
    all_preds = []

    for data in tqdm(data_loader, desc="Training", leave=False):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='binary')
    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader), f1

######################################################
def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # Binary
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

    f1 = f1_score(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader), f1, auc, tpr, fnr

def load_data(file_path):
    _df = pd.read_csv(file_path) 
    _df.rename(columns={'jailbreak': 'label'}, inplace=True)
    _df.reset_index(drop=True, inplace=True)
    return _df

train_df = load_data(BUCKET_PATH + 'dataset.csv')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = torch.nn.CrossEntropyLoss().to(device)

EPOCHS = 4
NUM_RUNS = 30
batch_size = 16
max_len = 128
metrics_list = []


for run in range(1, NUM_RUNS + 1):
    print(f"\n====== Run {run}/{NUM_RUNS} ======")
    train_sample = train_df.sample(frac=0.8, random_state=run).reset_index(drop=True)
    val_sample = train_df.drop(train_sample.index).reset_index(drop=True)

    train_dataset = TextDataset(train_sample, tokenizer, max_len)
    val_dataset = TextDataset(val_sample, tokenizer, max_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                 num_training_steps=len(train_loader) * EPOCHS)


    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS} [Run {run}]")
        train_acc, train_loss, train_f1 = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        val_acc, val_loss, val_f1, val_auc, val_tpr, val_fnr = eval_model(model, val_loader, loss_fn, device)
        print(f" - Accuracy {val_acc.item()} AUC: {val_auc} F1 {val_f1}")
            
    best_metrics = {'accuracy': val_acc.item(),
                    'auc': val_auc, 'f1': val_f1, 
                    'tpr': val_tpr, 'fnr': val_fnr}
    metrics_list.append(best_metrics)


with open(BUCKET_PATH + 'metrics.pkl', 'wb') as f:  # open a text file
    pickle.dump(metrics_list, f) # serialize the list

