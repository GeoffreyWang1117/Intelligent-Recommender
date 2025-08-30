"""
Training module for the LayerwiseAdapter model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import logging
import time
import json
import os
from tqdm import tqdm

try:
    from ..models.base import ModelConfig, UserProfile, ItemProfile, ExperimentLogger
    from ..models.layerwise_adapter import LayerwiseAdapter
    from ..models.teacher_adapters import MultiTeacherManager, create_teacher_manager
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models.base import ModelConfig, UserProfile, ItemProfile, ExperimentLogger
    from models.layerwise_adapter import LayerwiseAdapter
    from models.teacher_adapters import MultiTeacherManager, create_teacher_manager


class RecommendationDataset(Dataset):
    """Dataset for recommendation training."""
    
    def __init__(self, 
                 user_profiles: List[UserProfile],
                 item_profiles: List[ItemProfile], 
                 ratings: List[float]):
        self.user_profiles = user_profiles
        self.item_profiles = item_profiles
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        
        assert len(user_profiles) == len(item_profiles) == len(ratings)
    
    def __len__(self) -> int:
        return len(self.user_profiles)
    
    def __getitem__(self, idx: int) -> Tuple[UserProfile, ItemProfile, torch.Tensor]:
        return self.user_profiles[idx], self.item_profiles[idx], self.ratings[idx]


class LayerwiseTrainer:
    """Trainer for the LayerwiseAdapter model."""
    
    def __init__(self, 
                 config: ModelConfig,
                 model: LayerwiseAdapter,
                 teacher_manager: Optional[MultiTeacherManager] = None,
                 device: str = 'auto'):
        
        self.config = config
        self.model = model
        self.teacher_manager = teacher_manager
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Set teacher manager
        if teacher_manager is not None:
            self.model.set_teacher_manager(teacher_manager)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logger
        self.logger = logging.getLogger(__name__)
        self.experiment_logger = None
        
        self.logger.info(f"LayerwiseTrainer initialized on device: {self.device}")
    
    def setup_experiment_logging(self, experiment_name: str, log_dir: str) -> None:
        """Setup experiment logging."""
        self.experiment_logger = ExperimentLogger(experiment_name, log_dir)
        self.experiment_logger.log_model_info(self.model)
    
    def train(self,
              train_dataset: RecommendationDataset,
              val_dataset: Optional[RecommendationDataset] = None,
              save_dir: str = "./checkpoints") -> Dict[str, List[float]]:
        """
        Train the LayerwiseAdapter model.
        
        Returns:
            Training history with loss curves
        """
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self._collate_fn
            )
        
        # Training history
        history = {
            'train_loss': [],
            'train_task_loss': [],
            'train_distillation_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Load teachers if available
        if self.teacher_manager is not None:
            self.logger.info("Loading teacher models...")
            self.teacher_manager.load_all_teachers()
            ready_teachers = self.teacher_manager.get_ready_teachers()
            self.logger.info(f"Ready teachers: {ready_teachers}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Training loop
        self.logger.info("Starting training...")
        for epoch in range(self.config.max_epochs):
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            history['train_loss'].append(train_metrics['total_loss'])
            history['train_task_loss'].append(train_metrics.get('task_loss', 0.0))
            history['train_distillation_loss'].append(train_metrics.get('total_distillation', 0.0))
            history['learning_rate'].append(current_lr)
            
            if val_metrics:
                history['val_loss'].append(val_metrics['total_loss'])
            
            # Log to experiment logger
            if self.experiment_logger is not None:
                log_metrics = {
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'learning_rate': current_lr
                }
                log_metrics.update(train_metrics)
                log_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
                self.experiment_logger.log_metrics(log_metrics)
            
            # Print progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs} - "
                f"Train Loss: {train_metrics['total_loss']:.4f}"
                + (f" - Val Loss: {val_metrics['total_loss']:.4f}" if val_metrics else "")
            )
            
            # Early stopping
            if val_metrics:
                val_loss = val_metrics['total_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # Save best model
                    self._save_checkpoint(
                        os.path.join(save_dir, "best_model.pt"),
                        epoch, train_metrics, val_metrics
                    )
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                    epoch, train_metrics, val_metrics
                )
        
        # Save final model
        self._save_checkpoint(
            os.path.join(save_dir, "final_model.pt"),
            epoch, train_metrics, val_metrics
        )
        
        # Save training history
        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        
        self.logger.info("Training completed!")
        return history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_distillation_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (user_profiles, item_profiles, labels) in enumerate(pbar):
            
            # Move labels to device
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(user_profiles, item_profiles, return_layer_outputs=True)
            
            # Compute losses
            loss_dict = self.model.compute_distillation_loss(
                output, user_profiles, item_profiles, labels
            )
            
            total_loss_batch = loss_dict['total_loss']
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update distillation weights (progressive training)
            total_steps = len(train_loader) * self.config.max_epochs
            current_step = epoch * len(train_loader) + batch_idx
            self.model.update_distillation_weights(current_step, total_steps)
            
            # Accumulate metrics
            total_loss += total_loss_batch.item()
            total_task_loss += loss_dict.get('task_loss', torch.tensor(0.0)).item()
            total_distillation_loss += loss_dict.get('total_distillation', torch.tensor(0.0)).item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss_batch.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            self.global_step += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'task_loss': total_task_loss / num_batches,
            'total_distillation': total_distillation_loss / num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_distillation_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for user_profiles, item_profiles, labels in val_loader:
                
                # Move labels to device
                labels = labels.to(self.device)
                
                # Forward pass
                output = self.model(user_profiles, item_profiles, return_layer_outputs=True)
                
                # Compute losses
                loss_dict = self.model.compute_distillation_loss(
                    output, user_profiles, item_profiles, labels
                )
                
                # Accumulate metrics
                total_loss += loss_dict['total_loss'].item()
                total_task_loss += loss_dict.get('task_loss', torch.tensor(0.0)).item()
                total_distillation_loss += loss_dict.get('total_distillation', torch.tensor(0.0)).item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'task_loss': total_task_loss / num_batches,
            'total_distillation': total_distillation_loss / num_batches
        }
    
    def _collate_fn(self, batch: List[Tuple[UserProfile, ItemProfile, torch.Tensor]]) -> Tuple[List[UserProfile], List[ItemProfile], torch.Tensor]:
        """Collate function for data loader."""
        user_profiles, item_profiles, ratings = zip(*batch)
        
        # Convert ratings to tensor
        ratings_tensor = torch.stack(list(ratings))
        
        return list(user_profiles), list(item_profiles), ratings_tensor
    
    def _save_checkpoint(self, 
                        filepath: str, 
                        epoch: int,
                        train_metrics: Dict[str, float],
                        val_metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint
    
    def evaluate(self, test_dataset: RecommendationDataset) -> Dict[str, float]:
        """Evaluate model on test dataset."""
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        inference_times = []
        
        with torch.no_grad():
            for user_profiles, item_profiles, labels in tqdm(test_loader, desc="Evaluating"):
                
                # Measure inference time
                start_time = time.time()
                predictions, confidences = self.model.predict(user_profiles, item_profiles)
                end_time = time.time()
                
                inference_times.append((end_time - start_time) * 1000 / len(user_profiles))  # ms per sample
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_confidences.append(confidences.cpu())
        
        # Concatenate results
        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)
        confidences = torch.cat(all_confidences)
        
        # Compute metrics
        mse = torch.mean((predictions - labels) ** 2).item()
        mae = torch.mean(torch.abs(predictions - labels)).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        # Compute correlation
        pred_mean = torch.mean(predictions)
        label_mean = torch.mean(labels)
        numerator = torch.sum((predictions - pred_mean) * (labels - label_mean))
        denominator = torch.sqrt(torch.sum((predictions - pred_mean) ** 2) * 
                                torch.sum((labels - label_mean) ** 2))
        correlation = (numerator / denominator).item() if denominator > 0 else 0.0
        
        # Average inference time
        avg_inference_time = sum(inference_times) / len(inference_times)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'avg_confidence': torch.mean(confidences).item(),
            'avg_inference_time_ms': avg_inference_time
        }
        
        self.logger.info("Evaluation results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics


def create_trainer(config: ModelConfig, 
                  use_teachers: bool = True,
                  device: str = 'auto') -> LayerwiseTrainer:
    """Factory function to create a complete trainer."""
    
    # Create model
    model = LayerwiseAdapter(config)
    
    # Create teacher manager if requested
    teacher_manager = None
    if use_teachers:
        teacher_config = {
            'use_ensemble_teacher': True,
            'llm_models': ['llama3'],
            'embedding_dim': config.embedding_dim
        }
        teacher_manager = create_teacher_manager(teacher_config)
    
    # Create trainer
    trainer = LayerwiseTrainer(config, model, teacher_manager, device)
    
    return trainer
