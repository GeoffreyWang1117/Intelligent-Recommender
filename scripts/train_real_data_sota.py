#!/usr/bin/env python3
"""
基于真实MovieLens数据的SOTA算法训练脚本
利用现有的模型架构，使用真实数据训练6个SOTA推荐算法

基于现有的架构:
- 使用现有的BaseRecommender接口
- 复用现有的模型类
- 扩展现有的数据加载功能

作者: GitHub Copilot
日期: 2025-08-27
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from utils.data_loader import split_train_test
from models.svd_model import SVDRecommender

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealMovieLensLoader:
    """真实MovieLens数据加载器 - 基于现有架构"""
    
    def __init__(self, data_dir: str = "data/movielens/small"):
        self.data_dir = Path(data_dir)
        
    def load_movielens_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载真实MovieLens数据并转换为标准格式"""
        logger.info(f"从 {self.data_dir} 加载真实MovieLens数据...")
        
        # 加载评分数据
        ratings_file = self.data_dir / "ratings.csv"
        if not ratings_file.exists():
            raise FileNotFoundError(f"评分文件不存在: {ratings_file}")
        
        ratings_df = pd.read_csv(ratings_file)
        logger.info(f"加载评分数据: {len(ratings_df)} 条记录")
        
        # 标准化列名 (适配现有代码期望的格式)
        if 'userId' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        
        # 加载电影数据
        movies_file = self.data_dir / "movies.csv"
        movies_df = pd.DataFrame()
        if movies_file.exists():
            movies_df = pd.read_csv(movies_file)
            if 'movieId' in movies_df.columns:
                movies_df = movies_df.rename(columns={'movieId': 'item_id'})
            logger.info(f"加载电影数据: {len(movies_df)} 部电影")
        
        # 创建用户数据 (从评分数据中提取)
        users_df = self._create_users_from_ratings(ratings_df)
        
        # 数据清洗
        ratings_df = self._clean_ratings_data(ratings_df)
        
        logger.info(f"数据统计: {ratings_df['user_id'].nunique()} 用户, "
                   f"{ratings_df['item_id'].nunique()} 物品, {len(ratings_df)} 评分")
        
        return ratings_df, movies_df, users_df
    
    def _create_users_from_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """从评分数据中创建用户数据"""
        unique_users = ratings_df['user_id'].unique()
        
        # 生成模拟的用户属性
        np.random.seed(42)
        users_data = []
        
        for user_id in unique_users:
            users_data.append({
                'user_id': user_id,
                'gender': np.random.choice(['M', 'F']),
                'age': np.random.randint(18, 65),
                'occupation': np.random.randint(0, 21),
                'zipcode': f"{np.random.randint(10000, 99999)}"
            })
        
        return pd.DataFrame(users_data)
    
    def _clean_ratings_data(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """清洗评分数据"""
        # 移除空值
        ratings_df = ratings_df.dropna(subset=['user_id', 'item_id', 'rating'])
        
        # 确保数据类型正确
        ratings_df['user_id'] = ratings_df['user_id'].astype(int)
        ratings_df['item_id'] = ratings_df['item_id'].astype(int)
        ratings_df['rating'] = ratings_df['rating'].astype(float)
        
        # 移除重复项
        ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'item_id'])
        
        return ratings_df

class SOTAModelTrainer:
    """SOTA模型训练器 - 基于现有架构"""
    
    def __init__(self, model_save_dir: str = "models/saved"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True, parents=True)
        
        # 使用现有的模型配置，但增加epoch和添加早停
        self.available_models = {
            'SVD': {
                'class': SVDRecommender,
                'params': {'n_components': 50}  # SVD不需要epoch设置
            }
        }
        
        # 检测可用的模型并设置改进的训练参数
        try:
            from models.deepfm import DeepFMRecommender
            self.available_models['DeepFM'] = {
                'class': DeepFMRecommender,
                'params': {'embedding_dim': 64, 'epochs': 50, 'learning_rate': 0.001}  # 增加epoch
            }
        except ImportError:
            logger.warning("DeepFM模型不可用")
        
        try:
            from models.dcnv2 import DCNv2Recommender
            self.available_models['DCNv2'] = {
                'class': DCNv2Recommender,
                'params': {'embedding_dim': 64, 'epochs': 100, 'learning_rate': 0.001}  # 增加epoch
            }
        except ImportError:
            logger.warning("DCNv2模型不可用")
        
        try:
            from models.autoint import AutoIntRecommender
            self.available_models['AutoInt'] = {
                'class': AutoIntRecommender,
                'params': {'embedding_dim': 64, 'epochs': 50, 'learning_rate': 0.001}  # 增加epoch
            }
        except ImportError:
            logger.warning("AutoInt模型不可用")
        
        try:
            from models.xdeepfm import xDeepFMRecommender
            self.available_models['xDeepFM'] = {
                'class': xDeepFMRecommender,
                'params': {'embedding_dim': 64, 'epochs': 100, 'learning_rate': 0.001}  # 增加epoch
            }
        except ImportError:
            logger.warning("xDeepFM模型不可用")
        
        try:
            from models.transformer4rec import Transformer4RecRecommender
            self.available_models['Transformer4Rec'] = {
                'class': Transformer4RecRecommender,
                'params': {'embedding_dim': 64, 'num_heads': 8, 'num_layers': 3, 'epochs': 80}  # 新增
            }
        except ImportError:
            logger.warning("Transformer4Rec模型不可用")
        
        try:
            from models.din import DINRecommender
            self.available_models['DIN'] = {
                'class': DINRecommender,
                'params': {'embedding_dim': 64, 'epochs': 60, 'learning_rate': 0.001}  # 新增
            }
        except ImportError:
            logger.warning("DIN模型不可用")
        
        try:
            from models.lightfm_model import LightFMRecommender
            self.available_models['LightFM'] = {
                'class': LightFMRecommender,
                'params': {'no_components': 64, 'epochs': 50}  # 增加epoch
            }
        except ImportError:
            logger.warning("LightFM模型不可用 - 可能缺少lightfm库")
        
        logger.info(f"检测到可用模型: {list(self.available_models.keys())}")
    
    def train_all_models(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """训练所有可用的模型"""
        results = {}
        
        logger.info("开始训练所有可用的SOTA推荐算法...")
        logger.info("=" * 60)
        
        for model_name, model_config in self.available_models.items():
            logger.info(f"\n>>> 训练 {model_name} 模型 <<<")
            logger.info("-" * 40)
            
            try:
                # 初始化模型
                model_class = model_config['class']
                model_params = model_config['params']
                
                start_time = datetime.now()
                model = model_class(**model_params)
                
                # 训练模型
                logger.info("开始训练...")
                model.fit(train_data)
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                # 保存模型
                model_path = self.model_save_dir / f"{model_name}_real_movielens.pkl"
                model.save_model(str(model_path))
                
                # 评估模型
                evaluation = self._evaluate_model(model, test_data)
                evaluation['training_time'] = training_time
                
                results[model_name] = {
                    'model': model,
                    'model_path': str(model_path),
                    'evaluation': evaluation,
                    'status': 'success'
                }
                
                logger.info(f"✓ {model_name} 训练完成")
                logger.info(f"  训练时间: {training_time:.2f}秒")
                logger.info(f"  模型已保存: {model_path}")
                
                # 测试推荐功能
                self._test_recommendations(model, train_data, model_name)
                
            except Exception as e:
                logger.error(f"✗ {model_name} 训练失败: {str(e)}")
                results[model_name] = {
                    'model': None,
                    'error': str(e),
                    'status': 'failed'
                }
        
        return results
    
    def _evaluate_model(self, model, test_data: pd.DataFrame, sample_size: int = 1000) -> Dict[str, float]:
        """评估模型性能 - 减少错误输出"""
        try:
            logger.info("评估模型性能...")
            
            # 随机采样避免计算量过大
            if len(test_data) > sample_size:
                test_sample = test_data.sample(n=sample_size, random_state=42)
            else:
                test_sample = test_data
            
            predictions = []
            actuals = []
            error_count = 0
            max_errors_to_show = 3  # 最多显示3个错误
            
            for _, row in test_sample.iterrows():
                try:
                    pred = model.predict(int(row['user_id']), int(row['item_id']))
                    if pred is not None and not np.isnan(pred):
                        predictions.append(pred)
                        actuals.append(row['rating'])
                except Exception as e:
                    error_count += 1
                    if error_count <= max_errors_to_show:
                        logger.warning(f"预测失败 (样本 {error_count}): {str(e)[:50]}...")
                    elif error_count == max_errors_to_show + 1:
                        logger.warning(f"更多预测错误已静默处理...")
                    continue
            
            if error_count > 0:
                logger.info(f"总预测错误数: {error_count}/{len(test_sample)}")
            
            if len(predictions) == 0:
                return {'rmse': float('inf'), 'mae': float('inf'), 'coverage': 0.0}
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            mae = np.mean(np.abs(predictions - actuals))
            coverage = len(predictions) / len(test_sample)
            
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  覆盖率: {coverage:.2%}")
            
            return {
                'rmse': float(rmse),
                'mae': float(mae),
                'coverage': float(coverage),
                'sample_size': len(predictions),
                'error_count': error_count
            }
            
        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            return {'rmse': float('inf'), 'mae': float('inf'), 'coverage': 0.0}
    
    def _test_recommendations(self, model, train_data: pd.DataFrame, model_name: str):
        """测试推荐功能 - 优化输出"""
        try:
            # 选择一个测试用户
            test_user_id = train_data['user_id'].iloc[0]
            
            logger.info(f"测试用户 {test_user_id} 的推荐结果:")
            
            # 设置超时避免卡顿
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("推荐生成超时")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30秒超时
            
            try:
                recommendations = model.get_user_recommendations(test_user_id, top_k=5)
                signal.alarm(0)  # 取消超时
                
                if recommendations:
                    for i, rec in enumerate(recommendations[:5]):
                        if isinstance(rec, dict):
                            item_id = rec.get('item_id', 'N/A')
                            score = rec.get('score', 'N/A')
                        elif isinstance(rec, tuple):
                            item_id, score = rec[0], rec[1]
                        else:
                            item_id, score = rec, 'N/A'
                        
                        logger.info(f"  {i+1}. 物品#{item_id}, 评分: {score}")
                else:
                    logger.warning("推荐结果为空")
                    
            except TimeoutError:
                logger.warning("推荐生成超时，跳过推荐测试")
                signal.alarm(0)
                
        except Exception as e:
            logger.warning(f"推荐测试失败: {str(e)[:100]}...")  # 限制错误信息长度
    
    def save_training_results(self, results: Dict[str, Any], filename: str = "real_data_training_results.json"):
        """保存训练结果"""
        import json
        
        # 准备可序列化的结果
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'model_path': result.get('model_path', ''),
                'evaluation': result.get('evaluation', {}),
                'error': result.get('error', ''),
                'status': result.get('status', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
        
        results_file = self.model_save_dir / filename
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练结果已保存: {results_file}")

def main():
    """主函数 - 基于现有架构"""
    logger.info("使用真实MovieLens数据训练SOTA推荐算法")
    logger.info("基于现有的模型架构和接口")
    logger.info("=" * 80)
    
    try:
        # 1. 加载真实数据
        data_loader = RealMovieLensLoader("data/movielens/small")
        ratings_df, movies_df, users_df = data_loader.load_movielens_data()
        
        # 2. 划分训练测试集 (使用现有的split_train_test函数)
        train_data, test_data = split_train_test(ratings_df, test_ratio=0.2)
        logger.info(f"数据划分: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")
        
        # 3. 训练所有可用模型
        trainer = SOTAModelTrainer()
        results = trainer.train_all_models(train_data, test_data)
        
        # 4. 保存结果
        trainer.save_training_results(results)
        
        # 5. 输出总结
        logger.info("\n" + "=" * 80)
        logger.info("训练总结:")
        
        successful_models = [name for name, result in results.items() if result.get('status') == 'success']
        failed_models = [name for name, result in results.items() if result.get('status') == 'failed']
        
        logger.info(f"成功训练: {len(successful_models)} 个模型")
        for name in successful_models:
            eval_result = results[name]['evaluation']
            logger.info(f"  {name}: RMSE={eval_result.get('rmse', 'N/A'):.4f}, "
                       f"MAE={eval_result.get('mae', 'N/A'):.4f}, "
                       f"覆盖率={eval_result.get('coverage', 0):.2%}")
        
        if failed_models:
            logger.info(f"训练失败: {len(failed_models)} 个模型")
            for name in failed_models:
                logger.warning(f"  {name}: {results[name]['error']}")
        
        logger.info(f"所有模型已保存到: {trainer.model_save_dir}")
        logger.info("可以使用这些模型进行后续的Fisher信息分析和pruning-aware蒸馏实验")
        
    except Exception as e:
        logger.error(f"训练过程出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
