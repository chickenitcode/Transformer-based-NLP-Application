"""
Configuration file for Transformer-based NLP Application
Text Classification using DistilBERT
"""

import os

# Model Configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# Dataset Configuration
DATASET_NAME = "ag_news"  # AG News dataset for text classification
NUM_CLASSES = 4
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Training Configuration
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Paths
MODEL_SAVE_PATH = "models"
DATA_PATH = "data"
RESULTS_PATH = "results"

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True) 