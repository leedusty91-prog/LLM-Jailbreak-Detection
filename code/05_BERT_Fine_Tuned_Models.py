"""Fine-tune bert-base-uncased for binary jailbreak detection.

The previous split drew an 80% sample, reset its index, and then dropped
those new positions from the parent frame. After the reset, the sample index
is ``0 .. n_train-1``, so the drop removes the first rows of the parent
instead of the sampled rows. Training and validation therefore shared most
prompts. The script also read one Cloud Storage object and reported that
contaminated validation slice, rather than the held-out test file.

Each run now cuts a fresh train/validation pair out of the training file with
``split_train_val`` (disjoint prompt strings) and reports Accuracy, F1, AUC,
TPR, and FNR on ``testing_data_binary.csv``. Validation AUC picks the epoch
whose weights are scored on the test set. The test set is not used for early
stopping.

Environment variables ``NUM_RUNS`` and ``EPOCHS`` override the paper settings
(30 runs, 4 epochs) for a CPU smoke check. Full paper numbers still require a
GPU.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from binary_data import load_binary_frames
from latex_tables import export_results
from paths import data_path
from splits import assert_no_prompt_overlap, repair_binary_frames, split_train_val


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = str(self.data.loc[index, "prompt"])
        label = int(self.data.loc[index, "label"])
        encoding = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    all_labels = []
    all_preds = []
    for batch in tqdm(data_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        predictions = outputs.logits.argmax(dim=1)
        correct_predictions += torch.sum(predictions == labels)
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(predictions.detach().cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader), f1


def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    all_labels = []
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            probabilities = torch.softmax(outputs.logits, dim=1)[:, 1]
            predictions = outputs.logits.argmax(dim=1)
            correct_predictions += torch.sum(predictions == labels)
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probabilities.detach().cpu().numpy())
            all_preds.extend(predictions.detach().cpu().numpy())
    matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    positive = tp + fn
    tpr = tp / positive if positive else 0.0
    fnr = fn / positive if positive else 0.0
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    return accuracy, total_loss / len(data_loader), f1, float(auc), float(tpr), float(fnr)


def load_detection_frames(train_path=None, test_path=None):
    """Load the binary files, drop leaked prompts, and rename the label column."""
    train_path = train_path or os.environ.get("BERT_TRAIN_CSV") or data_path(
        "binary_data", "training_data_binary.csv"
    )
    test_path = test_path or os.environ.get("BERT_TEST_CSV") or data_path(
        "binary_data", "testing_data_binary.csv"
    )
    train_df, test_df = load_binary_frames(train_path, test_path)
    train_df, test_df, report = repair_binary_frames(train_df, test_df)
    return _rename_label(train_df), _rename_label(test_df), report


def _rename_label(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    if "label" not in renamed.columns:
        if "jailbreak" not in renamed.columns:
            raise KeyError("Expected a jailbreak or label column")
        renamed = renamed.rename(columns={"jailbreak": "label"})
    renamed["label"] = renamed["label"].astype(int)
    return renamed.reset_index(drop=True)


def run_finetune(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_runs: int = 30,
    epochs: int = 4,
    batch_size: int = 16,
    max_len: int = 128,
    learning_rate: float = 2e-5,
):
    """Fine-tune one BERT classifier per run and score the held-out test set."""
    if train_df.empty or test_df.empty:
        raise ValueError("Train and test frames must both be non-empty")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for run in range(1, num_runs + 1):
        print(f"\n====== Run {run}/{num_runs} ======")
        _seed_everything(run)
        train_sample, val_sample = split_train_val(train_df, frac=0.8, random_state=run)
        assert_no_prompt_overlap(train_sample, val_sample, test_df)
        train_loader = torch.utils.data.DataLoader(
            TextDataset(train_sample, tokenizer, max_len), batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            TextDataset(val_sample, tokenizer, max_len), batch_size=batch_size
        )
        test_loader = torch.utils.data.DataLoader(
            TextDataset(test_df, tokenizer, max_len), batch_size=batch_size
        )
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs,
        )
        best_state = None
        best_val_auc = -1.0
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs} [Run {run}]")
            train_epoch(model, train_loader, optimizer, device, scheduler)
            _val_acc, _val_loss, _val_f1, val_auc, _val_tpr, _val_fnr = eval_model(
                model, val_loader, device
            )
            print(f" - validation AUC {val_auc:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError("Fine-tuning produced no validation checkpoint")
        model.load_state_dict(best_state)
        test_acc, _test_loss, test_f1, test_auc, test_tpr, test_fnr = eval_model(
            model, test_loader, device
        )
        print(
            f" - test accuracy {float(test_acc):.4f} AUC {test_auc:.4f} "
            f"F1 {test_f1:.4f} TPR {test_tpr:.4f} FNR {test_fnr:.4f}"
        )
        runs.append(
            {
                "features": "BERT",
                "model": "BERT",
                "auc": test_auc,
                "accuracy": float(test_acc),
                "fnr": test_fnr,
                "tpr": test_tpr,
                "f1": float(test_f1),
                "run": run,
                "best_validation_auc": best_val_auc,
            }
        )
    return runs


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    num_runs = int(os.environ.get("NUM_RUNS", "30"))
    epochs = int(os.environ.get("EPOCHS", "4"))
    train_df, test_df, report = load_detection_frames()
    print(
        "Binary split repair: "
        f"{report['shared_prompts_before']} shared prompts removed from evaluation, "
        f"{report['train_rows_after']} train rows, {report['test_rows_after']} test rows."
    )
    runs = run_finetune(train_df, test_df, num_runs=num_runs, epochs=epochs)
    paths = export_results("05_bert_finetuned", "binary", runs)
    print(f"Wrote {paths['tex']}")


if __name__ == "__main__":
    main()
