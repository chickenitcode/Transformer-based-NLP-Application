"""
Data loader for AG News dataset - Google Colab Version
Handles dataset loading, preprocessing, and splitting with GPU optimization
"""

import torch
from datasets import load_dataset
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import config

class AGNewsDataset(Dataset):
    """Custom Dataset for AG News"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_ag_news_dataset():
    """Load AG News dataset from Hugging Face"""
    print("Loading AG News dataset...")
    dataset = load_dataset(config.DATASET_NAME)
    
    # AG News has labels 1-4, we need to convert to 0-3
    def convert_labels(example):
        # Ensure label is in valid range 1-4, then convert to 0-3
        if example['label'] not in [1, 2, 3, 4]:
            example['label'] = 1  # Default to label 1 if invalid
        example['label'] = example['label'] - 1
        return example
    
    dataset = dataset.map(convert_labels)
    
    print(f"Dataset loaded successfully!")
    print(f"Train: {len(dataset['train'])} samples")
    print(f"Test: {len(dataset['test'])} samples")
    
    return dataset

def prepare_data_loaders():
    """Prepare data loaders for training, validation, and testing"""
    
    # Load dataset
    dataset = load_ag_news_dataset()
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Split test set into validation and test
    test_dataset = dataset['test']
    test_size = len(test_dataset)
    val_size = int(test_size * config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT))
    test_size = test_size - val_size
    
    # Split the test set
    val_dataset, test_dataset = torch.utils.data.random_split(
        test_dataset, [val_size, test_size]
    )
    
    # Create custom datasets
    train_dataset = AGNewsDataset(
        texts=[item['text'] for item in dataset['train']],
        labels=[item['label'] for item in dataset['train']],
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    val_dataset = AGNewsDataset(
        texts=[dataset['test'][idx]['text'] for idx in val_dataset.indices],
        labels=[dataset['test'][idx]['label'] for idx in val_dataset.indices],
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    test_dataset = AGNewsDataset(
        texts=[dataset['test'][idx]['text'] for idx in test_dataset.indices],
        labels=[dataset['test'][idx]['label'] for idx in test_dataset.indices],
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    # Create data loaders with GPU optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Tăng num_workers cho GPU
        pin_memory=True  # Tối ưu cho GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,  # Tăng num_workers cho GPU
        pin_memory=True  # Tối ưu cho GPU
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,  # Tăng num_workers cho GPU
        pin_memory=True  # Tối ưu cho GPU
    )
    
    print(f"Data loaders created:")
    print(f"Train: {len(train_loader)} batches")
    print(f"Validation: {len(val_loader)} batches")
    print(f"Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, tokenizer

if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader, tokenizer = prepare_data_loaders()
    print("Data loading test completed successfully!")
