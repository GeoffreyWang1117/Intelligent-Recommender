"""
Simplified Real LLM Distillation Experiment with Llama3.
This script demonstrates basic knowledge distillation from real Llama3 to a simple student model.
"""

import os
import sys
import torch
import torch.nn as nn
import time
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.real_llm_teacher import create_real_llm_teacher, RealLlamaTeacher
from models.base import UserProfile, ItemProfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleStudentModel(nn.Module):
    """Simple student model for distillation demonstration."""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        combined = torch.cat([user_emb, item_emb], dim=1)
        return self.fc(combined).squeeze()


class SimpleLLMDistillationExperiment:
    """Simple real LLM distillation experiment."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load data
        self.user_mapping = {}
        self.item_mapping = {}
        self.train_data = []
        self.test_data = []
        
        # Models
        self.teacher = None  # Will be RealLlamaTeacher
        self.student = None  # Will be SimpleStudentModel
        
        # Results
        self.results = {
            'experiment_start': time.strftime('%Y-%m-%d %H:%M:%S'),
            'baseline_rmse': None,
            'distillation_rmse': None,
            'teacher_inference_time': None,
            'student_inference_time': None,
            'improvement': None
        }
    
    def load_data(self) -> None:
        """Load and preprocess MovieLens data."""
        logger.info("Loading MovieLens data...")
        
        # Load ratings
        ratings_df = pd.read_csv(f"{self.data_path}/ratings.csv")
        
        # Create user and item mappings
        unique_users = sorted(ratings_df['userId'].unique())
        unique_items = sorted(ratings_df['movieId'].unique())
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        # Convert to mapped data
        mapped_data = []
        for _, row in ratings_df.iterrows():
            user_idx = self.user_mapping[row['userId']]
            item_idx = self.item_mapping[row['movieId']]
            rating = row['rating']
            mapped_data.append((user_idx, item_idx, rating))
        
        # Split data
        np.random.shuffle(mapped_data)
        split_idx = int(0.8 * len(mapped_data))
        self.train_data = mapped_data[:split_idx]
        self.test_data = mapped_data[split_idx:]
        
        logger.info(f"Loaded {len(unique_users)} users, {len(unique_items)} items")
        logger.info(f"Train: {len(self.train_data)}, Test: {len(self.test_data)}")
    
    def setup_teacher(self) -> None:
        """Setup real LLM teacher."""
        logger.info("Setting up Llama3 teacher...")
        
        self.teacher = create_real_llm_teacher("llama3", config={
            'embedding_dim': 384,
            'temperature': 0.7
        })
        
        try:
            self.teacher.load_model("")  # Empty string for Ollama
        except:
            self.teacher.load_model()  # Try without parameter
            
        if hasattr(self.teacher, 'load_movielens_data'):
            self.teacher.load_movielens_data(self.data_path)
        
        logger.info("✅ Llama3 teacher ready")
    
    def setup_student(self) -> None:
        """Setup student model."""
        logger.info("Setting up student model...")
        
        num_users = len(self.user_mapping)
        num_items = len(self.item_mapping)
        
        self.student = SimpleStudentModel(num_users, num_items, embedding_dim=64)
        self.student.to(self.device)
        
        logger.info(f"✅ Student model ready: {sum(p.numel() for p in self.student.parameters())} parameters")
    
    def train_baseline(self, num_epochs: int = 5) -> float:
        """Train student model without teacher guidance."""
        logger.info("\\n🎯 Training Baseline Student Model...")
        
        # Reset student model
        self.setup_student()
        
        optimizer = torch.optim.Adam(self.student.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            count = 0
            
            for user_idx, item_idx, rating in self.train_data:
                optimizer.zero_grad()
                
                user_tensor = torch.tensor([user_idx], device=self.device)
                item_tensor = torch.tensor([item_idx], device=self.device)
                rating_tensor = torch.tensor([rating], device=self.device, dtype=torch.float)
                
                prediction = self.student(user_tensor, item_tensor)
                loss = criterion(prediction, rating_tensor)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
                
                # Limit training for demo (process every 10th sample)
                if count % 10 == 0 and count > 1000:
                    break
            
            avg_loss = total_loss / count if count > 0 else 0
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluate baseline
        baseline_rmse = self.evaluate_student()
        logger.info(f"Baseline RMSE: {baseline_rmse:.4f}")
        
        return baseline_rmse
    
    def train_with_distillation(self, num_epochs: int = 3) -> float:
        """Train student model with teacher guidance."""
        logger.info("\\n🦙 Training with Llama3 Distillation...")
        
        # Reset student model
        self.setup_student()
        
        optimizer = torch.optim.Adam(self.student.parameters(), lr=0.01)
        mse_criterion = nn.MSELoss()
        kl_criterion = nn.KLDivLoss(reduction='batchmean')
        
        alpha = 0.7  # Weight for distillation loss
        temperature = 3.0
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            total_task_loss = 0
            total_distill_loss = 0
            count = 0
            
            for user_idx, item_idx, rating in self.train_data:
                optimizer.zero_grad()
                
                # Get teacher prediction (slower, so we sample)
                if count % 5 == 0:  # Sample every 5th example for teacher guidance
                    teacher_prediction = self.get_teacher_prediction(user_idx, item_idx)
                else:
                    teacher_prediction = None
                
                # Student prediction
                user_tensor = torch.tensor([user_idx], device=self.device)
                item_tensor = torch.tensor([item_idx], device=self.device)
                rating_tensor = torch.tensor([rating], device=self.device, dtype=torch.float)
                
                student_prediction = self.student(user_tensor, item_tensor)
                
                # Task loss (student vs ground truth)
                task_loss = mse_criterion(student_prediction, rating_tensor)
                
                # Distillation loss (student vs teacher)
                if teacher_prediction is not None:
                    teacher_tensor = torch.tensor([teacher_prediction], device=self.device, dtype=torch.float)
                    
                    # Soft targets with temperature
                    student_soft = torch.log_softmax(student_prediction.unsqueeze(0) / temperature, dim=0)
                    teacher_soft = torch.softmax(teacher_tensor.unsqueeze(0) / temperature, dim=0)
                    
                    distill_loss = kl_criterion(student_soft, teacher_soft) * (temperature ** 2)
                else:
                    distill_loss = torch.tensor(0.0, device=self.device)
                
                # Combined loss
                total_loss_sample = (1 - alpha) * task_loss + alpha * distill_loss
                
                total_loss_sample.backward()
                optimizer.step()
                
                total_loss += total_loss_sample.item()
                total_task_loss += task_loss.item()
                total_distill_loss += distill_loss.item()
                count += 1
                
                # Limit training for demo
                if count > 500:  # Fewer samples due to teacher overhead
                    break
            
            avg_loss = total_loss / count if count > 0 else 0
            avg_task_loss = total_task_loss / count if count > 0 else 0
            avg_distill_loss = total_distill_loss / count if count > 0 else 0
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Total: {avg_loss:.4f}, Task: {avg_task_loss:.4f}, Distill: {avg_distill_loss:.4f}")
        
        # Evaluate distilled model
        distilled_rmse = self.evaluate_student()
        logger.info(f"Distillation RMSE: {distilled_rmse:.4f}")
        
        return distilled_rmse
    
    def get_teacher_prediction(self, user_idx: int, item_idx: int) -> float:
        """Get teacher prediction for a user-item pair."""
        try:
            # Convert indices back to original IDs
            original_user_id = list(self.user_mapping.keys())[list(self.user_mapping.values()).index(user_idx)]
            original_item_id = list(self.item_mapping.keys())[list(self.item_mapping.values()).index(item_idx)]
            
            # Create profiles
            user_profile = UserProfile(
                user_id=original_user_id,
                feature_vector=torch.randn(64),
                interaction_history=[]
            )
            
            item_profile = ItemProfile(
                item_id=original_item_id,
                feature_vector=torch.randn(64),
                category="movie"
            )
            
            # Get teacher prediction
            output = self.teacher.predict(user_profile, [item_profile])
            return output.predictions[0].item()
            
        except Exception as e:
            logger.warning(f"Teacher prediction failed: {e}")
            return 3.0  # Default neutral rating
    
    def evaluate_student(self) -> float:
        """Evaluate student model on test set."""
        self.student.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for user_idx, item_idx, rating in self.test_data[:200]:  # Sample for speed
                user_tensor = torch.tensor([user_idx], device=self.device)
                item_tensor = torch.tensor([item_idx], device=self.device)
                
                prediction = self.student(user_tensor, item_tensor)
                
                predictions.append(prediction.item())
                targets.append(rating)
        
        predictions = torch.tensor(predictions)
        targets = torch.tensor(targets)
        
        mse = torch.mean((predictions - targets) ** 2)
        rmse = torch.sqrt(mse)
        
        self.student.train()
        return rmse.item()
    
    def benchmark_inference_speed(self) -> Tuple[float, float]:
        """Benchmark inference speed for teacher and student."""
        logger.info("\\n⚡ Benchmarking Inference Speed...")
        
        # Test data
        test_user_idx, test_item_idx, _ = self.test_data[0]
        
        # Teacher speed
        start_time = time.time()
        for _ in range(5):  # 5 predictions
            _ = self.get_teacher_prediction(test_user_idx, test_item_idx)
        teacher_time = (time.time() - start_time) / 5
        
        # Student speed
        user_tensor = torch.tensor([test_user_idx], device=self.device)
        item_tensor = torch.tensor([test_item_idx], device=self.device)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(1000):  # 1000 predictions
                _ = self.student(user_tensor, item_tensor)
        student_time = (time.time() - start_time) / 1000
        
        logger.info(f"Teacher inference: {teacher_time:.3f}s per prediction")
        logger.info(f"Student inference: {student_time:.6f}s per prediction")
        logger.info(f"Speedup: {teacher_time / student_time:.1f}x")
        
        return teacher_time, student_time
    
    def run_experiment(self) -> None:
        """Run complete distillation experiment."""
        logger.info("🚀 Starting Simple LLM Distillation Experiment...")
        
        # Setup
        self.load_data()
        self.setup_teacher()
        
        # Baseline training
        baseline_rmse = self.train_baseline()
        
        # Distillation training
        distilled_rmse = self.train_with_distillation()
        
        # Benchmark speed
        teacher_time, student_time = self.benchmark_inference_speed()
        
        # Calculate improvement
        improvement = (baseline_rmse - distilled_rmse) / baseline_rmse * 100
        
        # Store results
        self.results.update({
            'baseline_rmse': baseline_rmse,
            'distillation_rmse': distilled_rmse,
            'teacher_inference_time': teacher_time,
            'student_inference_time': student_time,
            'improvement': improvement
        })
        
        # Print summary
        logger.info("\\n📊 Experiment Results:")
        logger.info(f"  Baseline RMSE: {baseline_rmse:.4f}")
        logger.info(f"  Distillation RMSE: {distilled_rmse:.4f}")
        logger.info(f"  Improvement: {improvement:.2f}%")
        logger.info(f"  Teacher Speed: {teacher_time:.3f}s")
        logger.info(f"  Student Speed: {student_time:.6f}s")
        logger.info(f"  Speedup: {teacher_time / student_time:.1f}x")
        
        # Save results
        self.save_results()
        
        if improvement > 0:
            logger.info("\\n✅ Knowledge Distillation Successful!")
        else:
            logger.info("\\n⚠️  Knowledge Distillation showed no improvement (may need more data/epochs)")
    
    def save_results(self) -> None:
        """Save experiment results."""
        results_file = "simple_llm_distillation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to: {results_file}")


def main():
    """Main experiment entry point."""
    
    # Configuration
    data_path = "/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/small"
    
    # Check if data exists
    if not os.path.exists(data_path):
        logger.error(f"Data path not found: {data_path}")
        return
    
    # Run experiment
    experiment = SimpleLLMDistillationExperiment(data_path)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
