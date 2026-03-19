"""
DistilBERT model for text classification
Custom classifier head on top of DistilBERT
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
import config

class DistilBertClassifier(nn.Module):
    """DistilBERT with custom classification head"""
    
    def __init__(self, num_classes=config.NUM_CLASSES, dropout=0.1):
        super(DistilBertClassifier, self).__init__()
        
        # Load DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained(config.MODEL_NAME)
        
        # Freeze DistilBERT layers (optional - can be unfrozen for fine-tuning)
        # for param in self.distilbert.parameters():
        #     param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Get DistilBERT outputs
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

def create_model():
    """Create and return the model"""
    model = DistilBertClassifier()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print("Model test completed successfully!") 