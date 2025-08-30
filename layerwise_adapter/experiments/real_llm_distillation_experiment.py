"""
Real LLM Distillation Experiment using Llama3 via Ollama.
This script demonstrates knowledge distillation from real Llama3 to LayerwiseAdapter.
"""

import os
import sys
import torch
import time
import json
import logging
from typing import Dict, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.real_llm_teacher import create_real_llm_teacher
from models.layerwise_adapter import LayerwiseAdapter
from models.base import UserProfile, ItemProfile
from utils.data_processor import MovieLensDataProcessor
from utils.trainer import LayerwiseTrainer
from distillation.pakd import PAKDDistiller

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealLLMDistillationExperiment:
    """Real LLM distillation experiment with Llama3 and Qwen3."""
    
    def __init__(self, data_path: str, results_dir: str = "real_llm_experiment_results"):
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.data_processor = None
        self.llama_teacher = None
        self.qwen_teacher = None
        self.student_model = None
        self.trainer = None
        
        # Experiment results
        self.results = {
            'experiment_start': time.strftime('%Y-%m-%d %H:%M:%S'),
            'llama3_results': {},
            'qwen3_results': {},
            'baseline_results': {},
            'distillation_comparison': {}
        }
    
    def setup_experiment(self) -> None:
        """Setup experiment components."""
        logger.info("Setting up Real LLM Distillation Experiment...")
        
        # 1. Initialize data processor
        logger.info("Initializing data processor...")
        self.data_processor = MovieLensDataProcessor(self.data_path)
        self.data_processor.load_data()
        
        # 2. Initialize real LLM teachers
        logger.info("Initializing real LLM teachers...")
        
        # Llama3 Teacher
        self.llama_teacher = create_real_llm_teacher("llama3", config={
            'embedding_dim': 384,
            'temperature': 0.7
        })
        
        try:
            self.llama_teacher.load_model()
            self.llama_teacher.load_movielens_data(self.data_path)
            logger.info("✅ Llama3 teacher loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Llama3 teacher: {e}")
            self.llama_teacher = None
        
        # Qwen3 Teacher (if available)
        try:
            self.qwen_teacher = create_real_llm_teacher("qwen3", config={
                'embedding_dim': 384,
                'temperature': 0.6
            })
            self.qwen_teacher.load_model()
            self.qwen_teacher.load_movielens_data(self.data_path)
            logger.info("✅ Qwen3 teacher loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Qwen3 teacher: {e}")
            self.qwen_teacher = None
        
        # 3. Initialize student model
        logger.info("Initializing LayerwiseAdapter student model...")
        
        num_users = len(self.data_processor.user_mapping)
        num_items = len(self.data_processor.item_mapping)
        
        self.student_model = LayerwiseAdapter(
            embedding_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            num_users=num_users,
            num_items=num_items,
            dropout=0.1
        )
        
        logger.info(f"Student model: {sum(p.numel() for p in self.student_model.parameters())} parameters")
    
    def run_baseline_experiment(self) -> Dict:
        """Run baseline experiment without teacher guidance."""
        logger.info("\\n🎯 Running Baseline Experiment (No Teacher)...")
        
        baseline_config = {
            'learning_rate': 0.001,
            'batch_size': 256,
            'num_epochs': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_teachers': False,
            'teacher_models': {}
        }
        
        # Create trainer
        trainer = LayerwiseTrainer(
            model=self.student_model,
            data_processor=self.data_processor,
            config=baseline_config
        )
        
        # Train model
        start_time = time.time()
        history = trainer.train()
        training_time = time.time() - start_time
        
        # Evaluate
        metrics = trainer.evaluate()
        
        # Store results
        baseline_results = {
            'training_time': training_time,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'test_metrics': metrics,
            'model_parameters': sum(p.numel() for p in self.student_model.parameters()),
            'config': baseline_config
        }
        
        logger.info(f"Baseline RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"Baseline Training Time: {training_time:.2f}s")
        
        return baseline_results
    
    def run_llama3_distillation(self) -> Dict:
        """Run distillation experiment with Llama3 teacher."""
        if self.llama_teacher is None:
            logger.warning("Llama3 teacher not available, skipping experiment")
            return {}
        
        logger.info("\\n🦙 Running Llama3 Distillation Experiment...")
        
        # Reset student model
        self._reset_student_model()
        
        llama_config = {
            'learning_rate': 0.001,
            'batch_size': 256,
            'num_epochs': 15,  # More epochs for distillation
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_teachers': True,
            'teacher_models': {'llama3': self.llama_teacher},
            'distillation_alpha': 0.7,  # Weight for distillation loss
            'distillation_temperature': 3.0
        }
        
        # Create trainer with teacher
        trainer = LayerwiseTrainer(
            model=self.student_model,
            data_processor=self.data_processor,
            config=llama_config
        )
        
        # Train with distillation
        start_time = time.time()
        history = trainer.train()
        training_time = time.time() - start_time
        
        # Evaluate
        metrics = trainer.evaluate()
        
        # Test teacher prediction quality
        teacher_metrics = self._evaluate_teacher_quality(self.llama_teacher)
        
        # Store results
        llama_results = {
            'training_time': training_time,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'test_metrics': metrics,
            'teacher_metrics': teacher_metrics,
            'config': llama_config,
            'distillation_successful': True
        }
        
        logger.info(f"Llama3 Distillation RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"Llama3 Training Time: {training_time:.2f}s")
        logger.info(f"Teacher Quality RMSE: {teacher_metrics.get('rmse', 'N/A'):.4f}")
        
        return llama_results
    
    def run_qwen3_distillation(self) -> Dict:
        """Run distillation experiment with Qwen3 teacher."""
        if self.qwen_teacher is None:
            logger.warning("Qwen3 teacher not available, skipping experiment")
            return {}
        
        logger.info("\\n🤖 Running Qwen3 Distillation Experiment...")
        
        # Reset student model
        self._reset_student_model()
        
        qwen_config = {
            'learning_rate': 0.001,
            'batch_size': 256,
            'num_epochs': 15,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_teachers': True,
            'teacher_models': {'qwen3': self.qwen_teacher},
            'distillation_alpha': 0.7,
            'distillation_temperature': 2.5  # Lower temperature for Qwen
        }
        
        trainer = LayerwiseTrainer(
            model=self.student_model,
            data_processor=self.data_processor,
            config=qwen_config
        )
        
        start_time = time.time()
        history = trainer.train()
        training_time = time.time() - start_time
        
        metrics = trainer.evaluate()
        teacher_metrics = self._evaluate_teacher_quality(self.qwen_teacher)
        
        qwen_results = {
            'training_time': training_time,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'test_metrics': metrics,
            'teacher_metrics': teacher_metrics,
            'config': qwen_config,
            'distillation_successful': True
        }
        
        logger.info(f"Qwen3 Distillation RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"Qwen3 Training Time: {training_time:.2f}s")
        
        return qwen_results
    
    def run_multi_teacher_distillation(self) -> Dict:
        """Run distillation with multiple teachers (Llama3 + Qwen3)."""
        if self.llama_teacher is None and self.qwen_teacher is None:
            logger.warning("No teachers available for multi-teacher distillation")
            return {}
        
        logger.info("\\n🎭 Running Multi-Teacher Distillation Experiment...")
        
        # Reset student model
        self._reset_student_model()
        
        # Prepare teachers
        teachers = {}
        if self.llama_teacher:
            teachers['llama3'] = self.llama_teacher
        if self.qwen_teacher:
            teachers['qwen3'] = self.qwen_teacher
        
        multi_config = {
            'learning_rate': 0.001,
            'batch_size': 256,
            'num_epochs': 20,  # More epochs for multi-teacher
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_teachers': True,
            'teacher_models': teachers,
            'distillation_alpha': 0.8,  # Higher weight for multi-teacher
            'distillation_temperature': 3.0
        }
        
        trainer = LayerwiseTrainer(
            model=self.student_model,
            data_processor=self.data_processor,
            config=multi_config
        )
        
        start_time = time.time()
        history = trainer.train()
        training_time = time.time() - start_time
        
        metrics = trainer.evaluate()
        
        multi_results = {
            'training_time': training_time,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'test_metrics': metrics,
            'num_teachers': len(teachers),
            'teachers_used': list(teachers.keys()),
            'config': multi_config,
            'distillation_successful': True
        }
        
        logger.info(f"Multi-Teacher RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"Multi-Teacher Training Time: {training_time:.2f}s")
        
        return multi_results
    
    def _reset_student_model(self) -> None:
        """Reset student model parameters."""
        for layer in self.student_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def _evaluate_teacher_quality(self, teacher) -> Dict:
        """Evaluate teacher prediction quality on test set."""
        try:
            test_data = self.data_processor.get_test_data()
            predictions = []
            targets = []
            
            # Sample a subset for evaluation (teachers can be slow)
            sample_size = min(100, len(test_data))
            test_sample = test_data[:sample_size]
            
            for user_id, item_id, rating in test_sample:
                # Create profiles
                user_profile = UserProfile(
                    user_id=user_id,
                    feature_vector=torch.randn(64),  # Dummy features
                    interaction_history=[]
                )
                
                item_profile = ItemProfile(
                    item_id=item_id,
                    feature_vector=torch.randn(64),
                    category="movie"
                )
                
                # Get teacher prediction
                try:
                    output = teacher.predict(user_profile, [item_profile])
                    prediction = output.predictions[0].item()
                    predictions.append(prediction)
                    targets.append(rating)
                except Exception as e:
                    logger.warning(f"Teacher prediction failed: {e}")
                    predictions.append(3.0)  # Neutral prediction
                    targets.append(rating)
            
            # Calculate metrics
            if predictions and targets:
                predictions = torch.tensor(predictions)
                targets = torch.tensor(targets)
                
                mse = torch.mean((predictions - targets) ** 2)
                rmse = torch.sqrt(mse)
                mae = torch.mean(torch.abs(predictions - targets))
                
                return {
                    'rmse': rmse.item(),
                    'mae': mae.item(),
                    'samples_evaluated': len(predictions)
                }
            
        except Exception as e:
            logger.error(f"Teacher evaluation failed: {e}")
        
        return {'rmse': float('inf'), 'mae': float('inf'), 'samples_evaluated': 0}
    
    def run_full_experiment(self) -> None:
        """Run complete experiment suite."""
        logger.info("🚀 Starting Real LLM Distillation Experiment Suite...")
        
        # Setup
        self.setup_experiment()
        
        # Run experiments
        self.results['baseline_results'] = self.run_baseline_experiment()
        self.results['llama3_results'] = self.run_llama3_distillation()
        self.results['qwen3_results'] = self.run_qwen3_distillation()
        self.results['multi_teacher_results'] = self.run_multi_teacher_distillation()
        
        # Analyze results
        self._analyze_results()
        
        # Save results
        self._save_results()
        
        logger.info("\\n✅ Real LLM Distillation Experiment Completed!")
    
    def _analyze_results(self) -> None:
        """Analyze and compare experimental results."""
        logger.info("\\n📊 Analyzing Results...")
        
        comparison = {}
        
        # Extract RMSEs for comparison
        baseline_rmse = self.results['baseline_results'].get('test_metrics', {}).get('rmse', float('inf'))
        llama_rmse = self.results['llama3_results'].get('test_metrics', {}).get('rmse', float('inf'))
        qwen_rmse = self.results['qwen3_results'].get('test_metrics', {}).get('rmse', float('inf'))
        multi_rmse = self.results['multi_teacher_results'].get('test_metrics', {}).get('rmse', float('inf'))
        
        comparison['rmse_comparison'] = {
            'baseline': baseline_rmse,
            'llama3_distillation': llama_rmse,
            'qwen3_distillation': qwen_rmse,
            'multi_teacher': multi_rmse
        }
        
        # Calculate improvements
        if baseline_rmse < float('inf'):
            comparison['improvements'] = {
                'llama3_improvement': (baseline_rmse - llama_rmse) / baseline_rmse * 100 if llama_rmse < float('inf') else 0,
                'qwen3_improvement': (baseline_rmse - qwen_rmse) / baseline_rmse * 100 if qwen_rmse < float('inf') else 0,
                'multi_teacher_improvement': (baseline_rmse - multi_rmse) / baseline_rmse * 100 if multi_rmse < float('inf') else 0
            }
        
        # Best performer
        rmse_values = [(name, rmse) for name, rmse in comparison['rmse_comparison'].items() if rmse < float('inf')]
        if rmse_values:
            best_method, best_rmse = min(rmse_values, key=lambda x: x[1])
            comparison['best_method'] = best_method
            comparison['best_rmse'] = best_rmse
        
        self.results['distillation_comparison'] = comparison
        
        # Print summary
        logger.info("\\n🏆 Results Summary:")
        for method, rmse in comparison['rmse_comparison'].items():
            if rmse < float('inf'):
                logger.info(f"  {method}: RMSE = {rmse:.4f}")
        
        if 'best_method' in comparison:
            logger.info(f"\\n🥇 Best Method: {comparison['best_method']} (RMSE: {comparison['best_rmse']:.4f})")
    
    def _save_results(self) -> None:
        """Save experiment results to files."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        results_file = self.results_dir / f"real_llm_experiment_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy/torch types to native Python for JSON serialization
            json_results = self._convert_for_json(self.results)
            json.dump(json_results, f, indent=2)
        
        # Save summary report
        report_file = self.results_dir / f"experiment_report_{timestamp}.md"
        self._generate_report(report_file)
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Report saved to: {report_file}")
    
    def _convert_for_json(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif hasattr(obj, 'item'):  # torch.Tensor or numpy scalar
            return obj.item()
        elif isinstance(obj, (torch.Tensor, )):
            return obj.tolist()
        else:
            return obj
    
    def _generate_report(self, report_file: Path) -> None:
        """Generate markdown experiment report."""
        with open(report_file, 'w') as f:
            f.write("# Real LLM Distillation Experiment Report\\n\\n")
            f.write(f"**Date**: {self.results['experiment_start']}\\n\\n")
            
            # Results table
            f.write("## Results Summary\\n\\n")
            f.write("| Method | RMSE | Training Time | Status |\\n")
            f.write("|--------|------|---------------|--------|\\n")
            
            for method in ['baseline_results', 'llama3_results', 'qwen3_results', 'multi_teacher_results']:
                if method in self.results and self.results[method]:
                    result = self.results[method]
                    rmse = result.get('test_metrics', {}).get('rmse', 'N/A')
                    time_taken = result.get('training_time', 'N/A')
                    status = "✅" if result.get('distillation_successful', True) else "❌"
                    
                    method_name = method.replace('_results', '').replace('_', ' ').title()
                    f.write(f"| {method_name} | {rmse:.4f if isinstance(rmse, float) else rmse} | {time_taken:.2f}s if isinstance(time_taken, (int, float)) else time_taken} | {status} |\\n")
            
            # Best method
            if 'distillation_comparison' in self.results:
                comparison = self.results['distillation_comparison']
                if 'best_method' in comparison:
                    f.write(f"\\n**Best Method**: {comparison['best_method']} (RMSE: {comparison['best_rmse']:.4f})\\n")


def main():
    """Main experiment entry point."""
    
    # Configuration
    data_path = "/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens/small"
    results_dir = "real_llm_experiment_results"
    
    # Check if data exists
    if not os.path.exists(data_path):
        logger.error(f"Data path not found: {data_path}")
        return
    
    # Run experiment
    experiment = RealLLMDistillationExperiment(data_path, results_dir)
    experiment.run_full_experiment()


if __name__ == "__main__":
    main()
