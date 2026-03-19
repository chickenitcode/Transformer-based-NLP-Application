"""
DistilBERT Classifier Model - Google Colab Version
Optimized for GPU training
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
import config

class DistilBertClassifier(nn.Module):
    """DistilBERT-based classifier for text classification"""
    
    def __init__(self, num_classes=None, dropout=0.1):
        super(DistilBertClassifier, self).__init__()
        
        # Use config.NUM_CLASSES if num_classes is not provided
        if num_classes is None:
            num_classes = config.NUM_CLASSES
        
        # Load pre-trained DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained(config.MODEL_NAME)
        
        # Freeze DistilBERT layers for faster training (optional)
        # Uncomment the following lines if you want to freeze the base model
        # for param in self.distilbert.parameters():
        #     param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)  # 768 is DistilBERT hidden size
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            logits: Classification logits
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_embeddings(self, input_ids, attention_mask):
        """
        Get embeddings from DistilBERT (for feature extraction)
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
            
        Returns:
            embeddings: Text embeddings
        """
        with torch.no_grad():
            outputs = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs.last_hidden_state[:, 0, :]
            return pooled_output

def create_model(device='cuda'):
    """
    Create and initialize the model
    
    Args:
        device: Device to place model on ('cuda' or 'cpu')
        
    Returns:
        model: Initialized DistilBERT classifier
    """
    model = DistilBertClassifier(
        num_classes=config.NUM_CLASSES,
        dropout=0.1
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024*1024):.1f} MB")
    print(f"Device: {device}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device)
    
    # Test forward pass
    batch_size = 2
    seq_length = 512
    input_ids = torch.randint(0, 30000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        print(f"Output shape: {outputs.shape}")
        print("Model test completed successfully!")
