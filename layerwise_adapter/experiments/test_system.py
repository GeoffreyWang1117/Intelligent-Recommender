"""
Simple experiment to test the LayerwiseAdapter system.
"""

import torch
import numpy as np
import logging
import os
import sys
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.base import ModelConfig, UserProfile, ItemProfile
from models.layerwise_adapter import LayerwiseAdapter
from utils.trainer import RecommendationDataset, LayerwiseTrainer, create_trainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_data(num_users: int = 1000, 
                         num_items: int = 500, 
                         num_interactions: int = 10000,
                         feature_dim: int = 128) -> tuple:
    """Create synthetic recommendation data for testing."""
    
    logger.info(f"Creating synthetic data: {num_users} users, {num_items} items, {num_interactions} interactions")
    
    # Create user profiles
    user_profiles = []
    for user_id in range(num_users):
        feature_vector = torch.randn(feature_dim)
        interaction_history = np.random.choice(num_items, size=5, replace=False).tolist()
        
        user_profile = UserProfile(
            user_id=user_id,
            feature_vector=feature_vector,
            interaction_history=interaction_history
        )
        user_profiles.append(user_profile)
    
    # Create item profiles  
    item_profiles = []
    for item_id in range(num_items):
        feature_vector = torch.randn(feature_dim)
        
        item_profile = ItemProfile(
            item_id=item_id,
            feature_vector=feature_vector,
            category=f"category_{item_id % 10}"
        )
        item_profiles.append(item_profile)
    
    # Create interactions
    train_users = []
    train_items = []
    train_ratings = []
    
    for _ in range(num_interactions):
        user_id = np.random.randint(0, num_users)
        item_id = np.random.randint(0, num_items)
        
        # Simple rating generation (can be made more sophisticated)
        user_feat = user_profiles[user_id].feature_vector
        item_feat = item_profiles[item_id].feature_vector
        
        # Simulate rating based on feature similarity
        similarity = torch.cosine_similarity(user_feat, item_feat, dim=0)
        rating = 3.0 + 2.0 * similarity.item() + 0.2 * torch.randn(1).item()
        rating = max(1.0, min(5.0, rating))  # Clamp to [1, 5]
        
        train_users.append(user_profiles[user_id])
        train_items.append(item_profiles[item_id])
        train_ratings.append(rating)
    
    logger.info("Synthetic data created successfully")
    return train_users, train_items, train_ratings, user_profiles, item_profiles


def test_model_components():
    """Test individual model components."""
    logger.info("Testing model components...")
    
    # Create config
    config = ModelConfig()
    config.num_users = 100
    config.num_items = 50
    config.embedding_dim = 64
    config.hidden_dim = 128
    config.batch_size = 32
    config.max_epochs = 2
    
    # Create model
    model = LayerwiseAdapter(config)
    
    # Test model size calculation
    size_info = model.get_model_size()
    logger.info(f"Model size: {size_info}")
    
    # Create sample data
    user_profiles = []
    item_profiles = []
    
    for i in range(10):
        user_profile = UserProfile(
            user_id=i,
            feature_vector=torch.randn(config.embedding_dim),
            interaction_history=[1, 2, 3]
        )
        user_profiles.append(user_profile)
        
        item_profile = ItemProfile(
            item_id=i,
            feature_vector=torch.randn(config.embedding_dim),
            category=f"cat_{i%3}"
        )
        item_profiles.append(item_profile)
    
    # Test forward pass
    try:
        output = model(user_profiles[:5], item_profiles[:5])
        logger.info(f"Forward pass successful. Output shape: {output.recommendations.shape}")
        
        # Test prediction
        predictions, confidences = model.predict(user_profiles[:3], item_profiles[:3])
        logger.info(f"Prediction successful. Predictions: {predictions.shape}, Confidences: {confidences.shape}")
        
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        return False
    
    logger.info("Model component tests passed!")
    return True


def test_training_pipeline():
    """Test the complete training pipeline."""
    logger.info("Testing training pipeline...")
    
    # Create config
    config = ModelConfig()
    config.num_users = 100
    config.num_items = 50
    config.embedding_dim = 64
    config.hidden_dim = 128
    config.batch_size = 16
    config.max_epochs = 2
    config.learning_rate = 1e-3
    
    # Create synthetic data
    train_users, train_items, train_ratings, all_users, all_items = create_synthetic_data(
        num_users=config.num_users,
        num_items=config.num_items,
        num_interactions=500,
        feature_dim=config.embedding_dim
    )
    
    # Split data into train/val
    split_idx = int(0.8 * len(train_users))
    
    train_dataset = RecommendationDataset(
        train_users[:split_idx],
        train_items[:split_idx], 
        train_ratings[:split_idx]
    )
    
    val_dataset = RecommendationDataset(
        train_users[split_idx:],
        train_items[split_idx:],
        train_ratings[split_idx:]
    )
    
    logger.info(f"Created datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Create trainer (without teachers for simplicity)
    trainer = create_trainer(config, use_teachers=False)
    
    # Setup experiment logging
    os.makedirs("./experiment_logs", exist_ok=True)
    trainer.setup_experiment_logging("test_experiment", "./experiment_logs")
    
    try:
        # Train model
        logger.info("Starting training...")
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir="./test_checkpoints"
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
        
        # Evaluate on test set (use validation set for simplicity)
        test_metrics = trainer.evaluate(val_dataset)
        logger.info(f"Test metrics: {test_metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_teacher_integration():
    """Test teacher model integration (mock version)."""
    logger.info("Testing teacher integration...")
    
    # Create config
    config = ModelConfig()
    config.num_users = 50
    config.num_items = 30
    config.embedding_dim = 64
    config.hidden_dim = 128
    config.batch_size = 8
    config.max_epochs = 1
    
    try:
        # Create trainer with teachers (will use mock teachers due to import issues)
        trainer = create_trainer(config, use_teachers=True)
        
        # Create small dataset
        train_users, train_items, train_ratings, _, _ = create_synthetic_data(
            num_users=config.num_users,
            num_items=config.num_items,
            num_interactions=100,
            feature_dim=config.embedding_dim
        )
        
        train_dataset = RecommendationDataset(
            train_users[:50],
            train_items[:50],
            train_ratings[:50]
        )
        
        # Quick training test
        logger.info("Testing with teacher models...")
        history = trainer.train(
            train_dataset=train_dataset,
            save_dir="./test_teacher_checkpoints"
        )
        
        logger.info("Teacher integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Teacher integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=== LayerwiseAdapter System Test ===")
    
    # Test 1: Model components
    logger.info("\n1. Testing model components...")
    if not test_model_components():
        logger.error("Model component tests failed!")
        return
    
    # Test 2: Training pipeline
    logger.info("\n2. Testing training pipeline...")
    if not test_training_pipeline():
        logger.error("Training pipeline tests failed!")
        return
    
    # Test 3: Teacher integration
    logger.info("\n3. Testing teacher integration...")
    if not test_teacher_integration():
        logger.error("Teacher integration tests failed!")
        return
    
    logger.info("\n=== All tests passed! ===")
    logger.info("LayerwiseAdapter system is working correctly.")


if __name__ == "__main__":
    main()
