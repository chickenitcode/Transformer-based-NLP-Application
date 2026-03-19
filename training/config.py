"""
Configuration file for Transformer-based NLP Application
Text Classification using DistilBERT - Google Colab Version with Drive
"""

import os

# Model Configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 32  # Tăng batch size cho GPU
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# Resume Training Configuration
RESUME_TRAINING = True  # Set to True to resume from checkpoint
CHECKPOINT_PATH = "models/best_model.pth"  # Path to checkpoint file
RESUME_EPOCH = 0  # Starting epoch (will be loaded from checkpoint)

# Dataset Configuration
DATASET_NAME = "ag_news"  # AG News dataset for text classification
NUM_CLASSES = 4
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# NUM_CLASSES = 3
# CLASS_NAMES = ["Class_0", "Class_1", "Class_2"]

# Training Configuration
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Paths - Local
MODEL_SAVE_PATH = "models"
DATA_PATH = "data"
RESULTS_PATH = "results"

# Paths - Google Drive
DRIVE_PATH = "/content/drive/MyDrive/transformer_project"
DRIVE_MODEL_PATH = os.path.join(DRIVE_PATH, "models")
DRIVE_RESULTS_PATH = os.path.join(DRIVE_PATH, "results")

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Create Drive directories (will be created after mounting)
# os.makedirs(DRIVE_PATH, exist_ok=True)
# os.makedirs(DRIVE_MODEL_PATH, exist_ok=True)
# os.makedirs(DRIVE_RESULTS_PATH, exist_ok=True)

