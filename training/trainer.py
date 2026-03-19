"""
Training script for DistilBERT Classifier - Google Colab Version with Drive Auto-Save
Optimized for GPU training with automatic Google Drive backup
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import time
import shutil
from tqdm import tqdm
import config
from data_loader import prepare_data_loaders
from model import create_model

def mount_google_drive():
    """Mount Google Drive to Colab"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Create project directories in Drive
        os.makedirs(config.DRIVE_PATH, exist_ok=True)
        os.makedirs(config.DRIVE_MODEL_PATH, exist_ok=True)
        os.makedirs(config.DRIVE_RESULTS_PATH, exist_ok=True)
        
        print(f"✅ Google Drive mounted successfully!")
        print(f"📁 Project folder: {config.DRIVE_PATH}")
        return True
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {str(e)}")
        print("⚠️  Results will only be saved locally")
        return False

def save_to_drive(local_path, drive_path):
    """Save file/folder to Google Drive"""
    try:
        if os.path.isfile(local_path):
            shutil.copy2(local_path, drive_path)
        elif os.path.isdir(local_path):
            if os.path.exists(drive_path):
                shutil.rmtree(drive_path)
            shutil.copytree(local_path, drive_path)
        print(f"✅ Saved to Drive: {drive_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save to Drive: {str(e)}")
        return False

class Trainer:
    """Training class for DistilBERT classifier with Drive backup"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.drive_mounted = False
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config.NUM_EPOCHS
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        val_accuracy = 100. * correct / total
        
        return val_loss, val_accuracy
    
    def save_checkpoint(self, epoch, model_path=None):
        """Save complete checkpoint including model, optimizer, scheduler states"""
        if model_path is None:
            model_path = config.CHECKPOINT_PATH
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy,
            'config': {
                'learning_rate': config.LEARNING_RATE,
                'batch_size': config.BATCH_SIZE,
                'num_epochs': config.NUM_EPOCHS
            }
        }
        
        # Save locally
        torch.save(checkpoint, model_path)
        
        # Save to Drive if mounted
        if self.drive_mounted:
            drive_checkpoint_path = os.path.join(config.DRIVE_MODEL_PATH, os.path.basename(model_path))
            save_to_drive(model_path, drive_checkpoint_path)
        
        print(f"✅ Checkpoint saved at epoch {epoch}: {model_path}")
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint and restore training state"""
        if checkpoint_path is None:
            checkpoint_path = config.CHECKPOINT_PATH
            
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
            
        try:
            print(f"🔄 Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract model state (support both full checkpoint and pure state_dict files)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Handle classifier size mismatch by dropping incompatible keys
            current_state = self.model.state_dict()
            removed_keys = []
            for key in list(state_dict.keys()):
                if key in current_state and state_dict[key].shape != current_state[key].shape:
                    removed_keys.append((key, tuple(state_dict[key].shape), tuple(current_state[key].shape)))
                    del state_dict[key]
            if removed_keys:
                print("⚠️  Detected incompatible parameter shapes; skipping these keys:")
                for key, old_shape, new_shape in removed_keys:
                    print(f"   - {key}: checkpoint {old_shape} → model {new_shape}")
            
            # Restore model state (allow missing keys for dropped head)
            self.model.load_state_dict(state_dict, strict=False)
            
            # Try to restore optimizer state if shapes match; otherwise skip
            loaded_opt = False
            if 'optimizer_state_dict' in checkpoint and not removed_keys:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    loaded_opt = True
                except Exception as e:
                    print(f"⚠️  Skipping optimizer state (incompatible): {str(e)}")
            else:
                if 'optimizer_state_dict' not in checkpoint:
                    print("ℹ️  No optimizer state in checkpoint.")
                if removed_keys:
                    print("ℹ️  Skipped optimizer state because model head changed.")
            
            # Try to restore scheduler state similarly
            if 'scheduler_state_dict' in checkpoint and loaded_opt and not removed_keys:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"⚠️  Skipping scheduler state: {str(e)}")
            else:
                if 'scheduler_state_dict' not in checkpoint:
                    print("ℹ️  No scheduler state in checkpoint.")
            
            # Restore training history when available
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            
            # Get starting epoch (if head changed, restart current epoch)
            start_epoch = checkpoint.get('epoch', 0)
            
            print(f"✅ Checkpoint loaded successfully!")
            print(f"📊 Resume from epoch: {start_epoch + 1}")
            print(f"📈 Best validation accuracy so far: {self.best_val_accuracy:.2f}%")
            print(f"📊 Training history: {len(self.train_losses)} epochs completed")
            if removed_keys:
                print("🔁 Note: Classifier head was reinitialized due to class count change; continuing with warm start.")
            
            return start_epoch
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {str(e)}")
            return False

    def save_model_with_backup(self, model_path, is_best=False):
        """Save model locally and to Drive if available"""
        # Save locally
        torch.save(self.model.state_dict(), model_path)
        
        # Save to Drive if mounted
        if self.drive_mounted:
            drive_model_path = os.path.join(config.DRIVE_MODEL_PATH, os.path.basename(model_path))
            save_to_drive(model_path, drive_model_path)
        
        if is_best:
            print(f"  ✅ New best model saved! (Val Acc: {self.best_val_accuracy:.2f}%)")
    
    def train(self, resume_from_checkpoint=False):
        """Complete training process with checkpoint support"""
        print(f"Starting training on {self.device}")
        print(f"Epochs: {config.NUM_EPOCHS}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print("-" * 50)
        
        # Handle resume training
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint()
            if start_epoch is False:
                print("⚠️  Failed to load checkpoint, starting from scratch")
                start_epoch = 0
            else:
                start_epoch += 1  # Start from next epoch
        
        start_time = time.time()
        
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print results
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint every epoch
            self.save_checkpoint(epoch)
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                best_model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')
                self.save_model_with_backup(best_model_path, is_best=True)
            
            print("-" * 50)
        
        # Save final model
        final_model_path = os.path.join(config.MODEL_SAVE_PATH, 'final_model.pth')
        self.save_model_with_backup(final_model_path)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
    
    def evaluate(self):
        """Evaluate on test set"""
        print("Evaluating on test set...")
        
        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')))
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_loss = total_loss / len(self.test_loader)
        test_accuracy = accuracy_score(all_labels, all_predictions) * 100
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Kiểm tra số lượng classes thực tế
        unique_labels = sorted(set(all_labels))
        actual_num_classes = len(unique_labels)
        
        print(f"Actual number of classes in test set: {actual_num_classes}")
        print(f"Unique labels: {unique_labels}")
        
        # Sử dụng class names phù hợp
        if actual_num_classes == len(config.CLASS_NAMES):
            target_names = config.CLASS_NAMES
        else:
            # Tạo class names tự động
            target_names = [f"Class_{i}" for i in range(actual_num_classes)]
            print(f"Using auto-generated class names: {target_names}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=target_names))
        
        return test_loss, test_accuracy, all_predictions, all_labels
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_PATH, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, predictions, labels):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=config.CLASS_NAMES, 
                   yticklabels=config.CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_PATH, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results_to_drive(self, results):
        """Save all results to Google Drive"""
        if self.drive_mounted:
            # Save results numpy file
            local_results_path = os.path.join(config.RESULTS_PATH, 'training_results.npy')
            drive_results_path = os.path.join(config.DRIVE_RESULTS_PATH, 'training_results.npy')
            np.save(local_results_path, results)
            save_to_drive(local_results_path, drive_results_path)
            
            # Save entire results folder
            save_to_drive(config.RESULTS_PATH, config.DRIVE_RESULTS_PATH)
            
            print(f"✅ All results saved to Google Drive: {config.DRIVE_PATH}")

def main():
    """Main training function with Drive integration and checkpoint support"""
    # Mount Google Drive
    drive_mounted = mount_google_drive()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, test_loader, tokenizer = prepare_data_loaders()
    
    # Create model
    model = create_model(device)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, device)
    trainer.drive_mounted = drive_mounted
    
    # Check if resume training is enabled
    if config.RESUME_TRAINING:
        print("🔄 Resume training mode enabled")
        print(f"📁 Looking for checkpoint: {config.CHECKPOINT_PATH}")
        
        # Check if checkpoint exists in Drive
        if drive_mounted:
            drive_checkpoint_path = os.path.join(config.DRIVE_MODEL_PATH, os.path.basename(config.CHECKPOINT_PATH))
            if os.path.exists(drive_checkpoint_path) and not os.path.exists(config.CHECKPOINT_PATH):
                print(f"📥 Downloading checkpoint from Drive: {drive_checkpoint_path}")
                save_to_drive(drive_checkpoint_path, config.CHECKPOINT_PATH)
    
    # Train with resume support
    trainer.train(resume_from_checkpoint=config.RESUME_TRAINING)
    
    # Evaluate
    test_loss, test_accuracy, predictions, labels = trainer.evaluate()
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(predictions, labels)
    
    # Save results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'best_val_accuracy': trainer.best_val_accuracy,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_accuracies': trainer.train_accuracies,
        'val_accuracies': trainer.val_accuracies
    }
    
    np.save(os.path.join(config.RESULTS_PATH, 'training_results.npy'), results)
    
    # Save to Drive
    trainer.save_results_to_drive(results)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Results saved in: {config.RESULTS_PATH}/")
    print(f"📁 Models saved in: {config.MODEL_SAVE_PATH}/")
    
    if drive_mounted:
        print(f"💾 Backup saved in Google Drive: {config.DRIVE_PATH}")
        print("✅ You can safely close Colab - results are saved in Drive!")

if __name__ == "__main__":
    main()

