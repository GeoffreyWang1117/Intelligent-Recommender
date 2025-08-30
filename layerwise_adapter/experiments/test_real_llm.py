"""
Simple test script for Real LLM Teacher integration.
This script tests the basic functionality of Llama3 teacher.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_llm_teacher():
    """Test real LLM teacher functionality."""
    
    try:
        # Import the real LLM teacher
        from models.real_llm_teacher import RealLlamaTeacher
        from models.base import UserProfile, ItemProfile
        
        logger.info("🦙 Testing Real Llama3 Teacher...")
        
        # Initialize teacher
        config = {
            'embedding_dim': 384,
            'temperature': 0.7
        }
        
        teacher = RealLlamaTeacher(model_name="llama3", config=config)
        
        # Test model loading
        logger.info("Loading Llama3 model via Ollama...")
        teacher.load_model()
        
        if not teacher.is_ready:
            logger.error("❌ Teacher model not ready")
            return False
        
        logger.info("✅ Teacher model loaded successfully")
        
        # Load MovieLens data
        data_path = "/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/small"
        if os.path.exists(data_path):
            logger.info("Loading MovieLens data...")
            teacher.load_movielens_data(data_path)
            logger.info(f"✅ Loaded {len(teacher.item_profiles)} movies, {len(teacher.user_profiles)} users")
        else:
            logger.warning("MovieLens data not found, using fallback data")
        
        # Test prediction
        logger.info("Testing teacher prediction...")
        
        # Create test user profile
        user_profile = UserProfile(
            user_id=1,
            feature_vector=torch.randn(64),
            interaction_history=[1, 2, 3]
        )
        
        # Create test item profiles
        candidate_items = [
            ItemProfile(item_id=1, feature_vector=torch.randn(64), category="movie"),
            ItemProfile(item_id=2, feature_vector=torch.randn(64), category="movie"),
            ItemProfile(item_id=3, feature_vector=torch.randn(64), category="movie")
        ]
        
        # Get teacher prediction
        start_time = time.time()
        output = teacher.predict(user_profile, candidate_items)
        inference_time = time.time() - start_time
        
        logger.info(f"✅ Teacher prediction successful")
        logger.info(f"   Predictions: {output.predictions.tolist()}")
        logger.info(f"   Embeddings shape: {output.embeddings.shape}")
        logger.info(f"   Confidence: {output.confidence}")
        logger.info(f"   Inference time: {inference_time:.3f}s")
        logger.info(f"   Reasoning: {output.reasoning_chain}")
        
        # Test embeddings
        logger.info("Testing embedding generation...")
        user_emb, item_emb = teacher.get_embeddings(user_profile, candidate_items[0])
        logger.info(f"✅ Embeddings generated: user {user_emb.shape}, item {item_emb.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_availability():
    """Test if required models are available."""
    
    logger.info("🔍 Checking model availability...")
    
    try:
        import ollama
        
        # List available models
        models_response = ollama.list()
        logger.info(f"Ollama response: {models_response}")
        
        # Handle different response formats
        if hasattr(models_response, 'models'):
            models_list = models_response.models
        elif isinstance(models_response, dict) and 'models' in models_response:
            models_list = models_response['models']
        else:
            models_list = []
        
        available_models = []
        for model in models_list:
            if hasattr(model, 'model'):
                available_models.append(model.model)
            elif isinstance(model, dict):
                if 'model' in model:
                    available_models.append(model['model'])
                elif 'name' in model:
                    available_models.append(model['name'])
                else:
                    logger.info(f"Unknown model format: {model}")
            else:
                available_models.append(str(model))
        
        logger.info(f"Available Ollama models: {available_models}")
        
        # Check for required models
        required_models = ['llama3:latest', 'qwen3:latest']
        for model in required_models:
            if model in available_models:
                logger.info(f"✅ {model} is available")
            else:
                logger.warning(f"⚠️  {model} not found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to check models: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    
    logger.info("🚀 Starting Real LLM Teacher Test...")
    
    # Check model availability
    if not test_model_availability():
        logger.error("Model availability check failed")
        return
    
    # Test real LLM teacher
    if test_real_llm_teacher():
        logger.info("\\n✅ All tests passed! Real LLM Teacher is working.")
    else:
        logger.error("\\n❌ Tests failed!")

if __name__ == "__main__":
    import time
    main()
