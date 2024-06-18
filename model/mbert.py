"""
Module for training SERENGETI LLM
"""

import time
import torch
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score
)
from models import load_data
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, random_split


logging.basicConfig(level=logging.INFO, filename='models.log', filemode='a')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def target_map(data):
    try:
        mapping = {1:1, 0:0, -1:2}
        data['label'] = data['label'].map(mapping)
        return data['label']
    except Exception as e:
        logging.error("Unable to map target")
        return None


def load_and_prepare_data(filepath):
    try:
        data = load_data(filepath)
        data['label'] = target_map(data=data)
        X = data['Comments'].values
        y = data['label'].values
        return X, y
    except Exception as e:
        logging.error(f"Error loading and preparing data: {e}")
        return None, None


def tokenize_texts(X, model_name, max_seq_length):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_texts = tokenizer(
            list(X), truncation=True, padding=True, return_tensors='pt', max_length=max_seq_length
        )
        return tokenized_texts.input_ids, tokenized_texts.attention_mask, tokenizer
    except Exception as e:
        logging.error(f"Error tokenizing texts: {e}")
        return None, None, None


def create_dataset(input_ids, attention_mask, labels):
    try:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
        return dataset
    except Exception as e:
        logging.error(f"Error creating dataset: {e}")
        return None


def split_dataset(dataset, train_ratio=0.8):
    try:
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset
    except Exception as e:
        logging.error(f"Error splitting dataset: {e}")
        return None, None


def create_dataloaders(train_dataset, val_dataset, batch_size):
    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader
    except Exception as e:
        logging.error(f"Error creating dataloaders: {e}")
        return None, None


def initialize_model(model_name, num_labels=3):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        model.to(device)
        return model
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        return None


def create_optimizer_and_scheduler(model, learning_rate, train_loader, epochs):
    try:
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs
        )
        return optimizer, scheduler
    except Exception as e:
        logging.error(f"Error creating optimizer and scheduler: {e}")
        return None, None


def train_model(model, train_loader, optimizer, scheduler, epochs):
    try:
        start = time.time()
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1}/{epochs} completed with mbert average loss: {avg_loss}")
        total = time.time() - start
        logging.info("The total amount of time it took to train the mbert is {}".format(total))
        return True
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return False


def evaluate_model(model, val_loader):
    try:
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        return val_labels, val_preds
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return None, None


def log_metrics(val_labels, val_preds):
    try:
        accuracy = accuracy_score(val_labels, val_preds)
        balanced = balanced_accuracy_score(val_labels, val_preds)
        rec = recall_score(val_labels, val_preds, average='weighted')
        f1 = f1_score(val_labels, val_preds, average='weighted')
        prec = precision_score(val_labels, val_preds, average='weighted')

        logging.info(f"Accuracy         : {accuracy}")
        logging.info(f"F1 Score         : {f1}")
        logging.info(f"Precision        : {prec}")
        logging.info(f"Balanced Accuracy: {balanced}")
        logging.info(f"Recall           : {rec}")
        return True
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return False


def main():
    try:
        filepath = 'Siswati_Sentiment.csv'
        model_name = 'bert-base-multilingual-uncased'
        batch_size = 32
        epochs = 10
        max_seq_length = 128
        learning_rate = 2e-5

        X, y = load_and_prepare_data(filepath)
        input_ids, attention_mask, tokenizer = tokenize_texts(X, model_name, max_seq_length)
        dataset = create_dataset(input_ids, attention_mask, y)
        train_dataset, val_dataset = split_dataset(dataset)
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size)
        model = initialize_model(model_name)
        optimizer, scheduler = create_optimizer_and_scheduler(model, learning_rate, train_loader, epochs)

        train_model(model, train_loader, optimizer, scheduler, epochs)
        val_labels, val_preds = evaluate_model(model, val_loader)
        log_metrics(val_labels, val_preds)
        return True
    except Exception as e:
        logging.error("Unable to run the main function! %s", e)
        return False
