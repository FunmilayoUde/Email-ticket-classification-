import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os

from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DistilBertTokenizer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads and preprocesses data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    try:
        data = pd.read_csv(filepath)
        data['subject'] = data['subject'].fillna('')
        data['body'] = data['body'].fillna('')
        data['combined_text'] = data['subject'].str.strip() + ' ' + data['body'].str.strip()
        return data
    except FileNotFoundError:
        logger.error(f"The file {filepath} was not found.")
        raise

def encode_labels(data: pd.DataFrame) -> (pd.DataFrame, dict, dict):
    
    labels = data['queue'].unique().tolist()
    labels = [s.strip() for s in labels]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    data['labels'] = data['queue'].apply(lambda x: label2id[x.strip()])
    return data, label2id, id2label

def tokenize_texts(texts: list, tokenizer: DistilBertTokenizer) -> dict:
    
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=128,
    )

class TicketDataset(Dataset):
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(pred):
    
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def parse_args():
    
    parser = argparse.ArgumentParser(description='Train DistilBERT model.')
    parser.add_argument('--data_path', type=str, default='helpdesk_customer_tickets.csv', help='Path to the CSV data file.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save model checkpoints.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info('Loading data...')
    data = load_data(args.data_path)
    
    logger.info('Encoding labels...')
    data, label2id, id2label = encode_labels(data)
    os.makedirs('./models', exist_ok=True)


    with open('./models/id2label.json', 'w') as f:
        json.dump(id2label, f)
    
    logger.info('Tokenizing texts...')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    tokenizer.save_pretrained('./tokenizer')
    
    logger.info('Splitting data...')
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['combined_text'].tolist(),
        data['labels'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=data['labels'],
    )
    
    logger.info('Tokenizing training data...')
    train_encodings = tokenize_texts(train_texts, tokenizer)
    logger.info('Tokenizing validation data...')
    val_encodings = tokenize_texts(val_texts, tokenizer)
    
    logger.info('Creating datasets...')
    train_dataset = TicketDataset(train_encodings, train_labels)
    val_dataset = TicketDataset(val_encodings, val_labels)
    
    logger.info('Loading model...')
    num_labels = len(label2id)
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-multilingual-cased',
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    
    logger.info('Setting up training arguments...')
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        evaluation_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
    )
    
    logger.info('Initializing Trainer...')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    logger.info('Starting training...')
    trainer.train()
    
    logger.info('Saving model...')
    trainer.save_model('./models')
    tokenizer.save_pretrained('./models')
    logger.info('Training completed and model saved.')

if __name__ == '__main__':
    main()
