#!/usr/bin/env python3
"""
训练推荐模型并构建FAISS索引 (使用示例数据)

本脚本用于训练基于SVD的推荐模型，并构建FAISS向量索引用于快速相似性搜索。
支持生成示例数据进行演示和测试。

Usage:
    python scripts/train_model.py --components 50 --epochs 30
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import pickle
import time
from typing import Tuple, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: FAISS库未安装，将使用简化版本")

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.svd_model import SVDRecommender
from models.faiss_index import FAISSIndex
from models.ensemble_recommender import ExplainableRecommender

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    创建示例数据用于演示
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 评分数据, 电影数据, 用户数据
    """
    logger.info("创建示例数据...")
    
    # 创建示例用户数据
    np.random.seed(42)
    n_users = 500
    n_items = 200
    n_ratings = 5000
    
    # 生成用户数据
    users = []
    for i in range(1, n_users + 1):
        users.append({
            'user_id': i,
            'gender': np.random.choice(['M', 'F']),
            'age': np.random.randint(18, 65),
            'occupation': np.random.choice(['student', 'engineer', 'teacher', 'doctor', 'artist'])
        })
    users_df = pd.DataFrame(users)
    
    # 生成电影数据
    movie_titles = [
        "The Matrix", "Star Wars", "The Godfather", "Pulp Fiction", "Fight Club",
        "Forrest Gump", "The Dark Knight", "Inception", "Goodfellas", "The Shawshank Redemption",
        "Schindler's List", "12 Angry Men", "The Lord of the Rings", "Casablanca", "Gone with the Wind"
    ] * (n_items // 15 + 1)
    
    movies = []
    genres_list = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Sci-Fi', 'Horror']
    
    for i in range(1, n_items + 1):
        movies.append({
            'item_id': i,
            'title': f"{movie_titles[i-1]} ({1990 + i % 30})",
            'genres': '|'.join(np.random.choice(genres_list, size=np.random.randint(1, 4), replace=False))
        })
    movies_df = pd.DataFrame(movies)
    
    # 生成评分数据
    ratings = []
    for _ in range(n_ratings):
        user_id = np.random.randint(1, n_users + 1)
        item_id = np.random.randint(1, n_items + 1)
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])  # 偏向高评分
        ratings.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': 946684800 + np.random.randint(0, 631152000)  # 2000-2020年间
        })
    
    # 去重（用户-物品对唯一）
    ratings_df = pd.DataFrame(ratings)
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'item_id']).reset_index(drop=True)
    
    logger.info(f"生成数据: {len(users_df)} 用户, {len(movies_df)} 电影, {len(ratings_df)} 评分")
    
    return ratings_df, movies_df, users_df


def split_train_test(ratings_df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    分割训练测试集
    
    Args:
        ratings_df: 评分数据
        test_ratio: 测试集比例
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 训练集, 测试集
    """
    np.random.seed(42)  # 确保可重复性
    n_test = int(len(ratings_df) * test_ratio)
    test_indices = np.random.choice(len(ratings_df), size=n_test, replace=False)
    
    test_ratings = ratings_df.iloc[test_indices].reset_index(drop=True)
    train_ratings = ratings_df.drop(test_indices).reset_index(drop=True)
    
    logger.info(f"数据分割: 训练集 {len(train_ratings)} 条, 测试集 {len(test_ratings)} 条")
    
    return train_ratings, test_ratings


def save_data_files(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, users_df: pd.DataFrame, 
                   data_dir: str = 'data/sample') -> None:
    """
    保存数据文件
    
    Args:
        ratings_df: 评分数据
        movies_df: 电影数据  
        users_df: 用户数据
        data_dir: 数据保存目录
    """
    os.makedirs(data_dir, exist_ok=True)
    
    ratings_df.to_csv(os.path.join(data_dir, 'ratings.csv'), index=False)
    movies_df.to_csv(os.path.join(data_dir, 'movies.csv'), index=False)
    users_df.to_csv(os.path.join(data_dir, 'users.csv'), index=False)
    
    logger.info(f"示例数据已保存到: {data_dir}")


def _train_specific_model(model_type: str, train_data, test_data, args):
    """训练特定模型"""
    logger.info(f"===== 训练 {model_type.upper()} 模型 =====")
    
    try:
        model = None
        if model_type == 'svd':
            model = SVDRecommender(n_components=args.components)
        elif model_type == 'deepfm':
            try:
                from models.deep_models import create_deep_recommender
                model = create_deep_recommender('deepfm', embedding_dim=8, hidden_dims=[128, 64])
            except ImportError:
                logger.error("无法导入深度学习模型，请确保PyTorch已安装")
                return
        elif model_type == 'xdeepfm':
            try:
                from models.deep_models import create_deep_recommender
                model = create_deep_recommender('xdeepfm', embedding_dim=8, hidden_dims=[128, 64])
            except ImportError:
                logger.error("无法导入深度学习模型，请确保PyTorch已安装")
                return
        elif model_type == 'din':
            try:
                from models.deep_models import create_deep_recommender
                model = create_deep_recommender('din', embedding_dim=8, hidden_dims=[128, 64])
            except ImportError:
                logger.error("无法导入深度学习模型，请确保PyTorch已安装")
                return
        elif model_type == 'autoint':
            try:
                from models.deep_models import create_deep_recommender
                model = create_deep_recommender('autoint', embedding_dim=8, num_heads=2, num_layers=3)
            except ImportError:
                logger.error("无法导入深度学习模型，请确保PyTorch已安装")
                return
        
        if model is None:
            logger.error(f"未知模型类型: {model_type}")
            return
        
        # 训练模型
        logger.info(f"开始训练{model_type}模型...")
        model.fit(train_data)
        
        # 保存模型
        model_file = os.path.join(args.model_path, f'{model_type}_model.pkl')
        model.save_model(model_file)
        logger.info(f"{model_type}模型已保存: {model_file}")
        
        # 测试推荐
        test_user_id = train_data['user_id'].iloc[0]
        logger.info(f"测试用户{test_user_id}的推荐结果:")
        
        recommendations = model.get_user_recommendations(test_user_id, top_k=5)
        if recommendations:
            for i, (item_id, score) in enumerate(recommendations[:5]):
                logger.info(f"  {i+1}. 电影#{item_id}, 预测评分: {score:.4f}")
        else:
            logger.warning("推荐结果为空")
            
    except Exception as e:
        logger.error(f"训练{model_type}模型失败: {e}")


def _train_individual_models(train_data, test_data, args):
    """单独训练所有模型并比较结果"""
    models_to_train = ['svd', 'deepfm', 'xdeepfm', 'din', 'autoint']
    results = {}
    
    logger.info("===== 开始单独训练所有模型 =====")
    
    for model_type in models_to_train:
        logger.info(f"\n>>> 训练 {model_type.upper()} 模型 <<<")
        
        try:
            model = None
            if model_type == 'svd':
                model = SVDRecommender(n_components=args.components)
            else:
                # 尝试创建深度学习模型
                try:
                    from models.deep_models import create_deep_recommender
                    if model_type == 'deepfm':
                        model = create_deep_recommender('deepfm', embedding_dim=8, hidden_dims=[128, 64])
                    elif model_type == 'xdeepfm':
                        model = create_deep_recommender('xdeepfm', embedding_dim=8, hidden_dims=[128, 64])
                    elif model_type == 'din':
                        model = create_deep_recommender('din', embedding_dim=8, hidden_dims=[128, 64])
                    elif model_type == 'autoint':
                        model = create_deep_recommender('autoint', embedding_dim=8, num_heads=2, num_layers=3)
                except ImportError:
                    logger.warning(f"无法导入{model_type}模型，跳过...")
                    continue
                except Exception as e:
                    logger.error(f"创建{model_type}模型失败: {e}")
                    continue
            
            if model is None:
                logger.warning(f"无法创建{model_type}模型")
                continue
            
            # 训练模型
            start_time = time.time()
            model.fit(train_data)
            training_time = time.time() - start_time
            
            # 保存模型
            model_file = os.path.join(args.model_path, f'{model_type}_individual_model.pkl')
            model.save_model(model_file)
            
            # 测试推荐
            test_user_id = train_data['user_id'].iloc[0]
            recommendations = model.get_user_recommendations(test_user_id, top_k=5)
            
            # 记录结果
            results[model_type] = {
                'model': model,
                'training_time': training_time,
                'recommendations': recommendations,
                'model_file': model_file
            }
            
            logger.info(f"✓ {model_type}模型训练完成，耗时: {training_time:.2f}秒")
            logger.info(f"  模型已保存: {model_file}")
            
            if recommendations:
                logger.info(f"  用户{test_user_id}的推荐结果:")
                for i, (item_id, score) in enumerate(recommendations[:3]):
                    logger.info(f"    {i+1}. 电影#{item_id}, 预测评分: {score:.4f}")
            else:
                logger.warning(f"  {model_type}模型推荐结果为空")
                
        except Exception as e:
            logger.error(f"训练{model_type}模型失败: {e}")
            results[model_type] = {'error': str(e)}
    
    # 总结比较结果
    logger.info("\n===== 模型训练结果总结 =====")
    successful_models = [k for k, v in results.items() if 'error' not in v]
    failed_models = [k for k, v in results.items() if 'error' in v]
    
    logger.info(f"成功训练的模型 ({len(successful_models)}): {', '.join(successful_models)}")
    if failed_models:
        logger.info(f"训练失败的模型 ({len(failed_models)}): {', '.join(failed_models)}")
    
    # 比较训练时间
    if successful_models:
        logger.info("\n训练时间比较:")
        for model_type in successful_models:
            time_cost = results[model_type]['training_time']
            logger.info(f"  {model_type}: {time_cost:.2f}秒")
    
    return results


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='训练推荐模型')
    parser.add_argument('--model-path', default='models/saved', help='模型保存路径')
    parser.add_argument('--components', type=int, default=50, help='SVD组件数')
    parser.add_argument('--use-sample-data', action='store_true', default=True, help='使用示例数据')
    parser.add_argument('--enable-ensemble', action='store_true', default=False, help='启用集成模型')
    parser.add_argument('--algorithms', nargs='+', default=['svd', 'deepfm'], help='集成算法列表')
    parser.add_argument('--train-individual', action='store_true', default=False, help='单独训练所有模型')
    parser.add_argument('--model-type', type=str, choices=['svd', 'deepfm', 'xdeepfm', 'din', 'autoint'], help='单独训练特定模型')
    
    args = parser.parse_args()
    
    try:
        # 使用示例数据
        if args.use_sample_data:
            ratings, movies, users = create_sample_data()
        else:
            logger.error("目前只支持示例数据模式")
            return False
        
        # 分割训练测试集
        logger.info("分割训练测试集...")
        train_ratings, test_ratings = split_train_test(ratings, test_ratio=0.2)
        
        # 创建模型保存目录
        os.makedirs(args.model_path, exist_ok=True)
        
        if args.enable_ensemble:
            # 训练集成模型
            logger.info("训练集成推荐模型...")
            algorithms = args.algorithms if args.algorithms else ['svd', 'deepfm']
            ensemble_model = ExplainableRecommender(
                algorithm_names=algorithms,
                diversity_weight=0.1,
                popularity_penalty=0.05
            )
            ensemble_model.fit(train_ratings)
            
            # 保存集成模型
            ensemble_file = os.path.join(args.model_path, 'ensemble_model.pkl')
            ensemble_model.save_model(ensemble_file)
            logger.info("集成模型已保存: %s", ensemble_file)
            
            # 测试集成推荐
            logger.info("测试集成推荐...")
            test_user_id = train_ratings['user_id'].iloc[0]
            ensemble_recs = ensemble_model.get_user_recommendations(test_user_id, top_k=5, enable_explanation=True)
            
            if ensemble_recs:
                logger.info("用户 %d 的集成推荐:", test_user_id)
                for i, rec in enumerate(ensemble_recs[:3]):
                    logger.info("  %d. %s (评分: %.3f)", i+1, rec.get('title', f"电影#{rec['item_id']}"), rec['score'])
                    if 'explanation' in rec:
                        logger.info("     推荐理由: %s", rec['explanation']['primary_reason'])
            
        elif args.train_individual:
            # 单独训练所有模型并比较结果
            logger.info("开始单独训练所有模型...")
            _train_individual_models(train_ratings, test_ratings, args)
        
        elif args.model_type:
            # 训练特定模型
            logger.info("训练特定模型: %s", args.model_type)
            _train_specific_model(args.model_type, train_ratings, test_ratings, args)
            
        else:
            # 原有的SVD训练逻辑
            logger.info("训练SVD推荐模型...")
            svd_model = SVDRecommender(n_components=args.components)
            svd_model.fit(train_ratings)
            
            # 保存模型
            model_file = os.path.join(args.model_path, 'svd_model.pkl')
            svd_model.save_model(model_file)
            logger.info("SVD模型已保存: %s", model_file)
            
            # 构建FAISS索引
            logger.info("构建FAISS索引...")
            
            # 获取所有物品的嵌入向量
            all_item_ids = list(svd_model.item_encoder.classes_)
            if not all_item_ids:
                logger.error("没有找到物品编码")
                return False
                
            item_embeddings = svd_model.get_item_embeddings(all_item_ids)
            if item_embeddings is None or len(item_embeddings) == 0:
                logger.error("无法获取物品嵌入向量")
                return False
                
            # 创建FAISS索引
            dimension = item_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
            
            # 标准化向量（对于内积相似度）
            item_embeddings = item_embeddings.astype('float32')  # 确保是float32类型
            faiss.normalize_L2(item_embeddings)
            index.add(item_embeddings)
            
            # 保存FAISS索引
            index_file = os.path.join(args.model_path, 'faiss_index.bin')
            faiss.write_index(index, index_file)
            logger.info("FAISS索引已保存: %s", index_file)
            
            # 保存物品编码映射
            item_mapping = {i: item_id for i, item_id in enumerate(all_item_ids)}
            mapping_file = os.path.join(args.model_path, 'item_mapping.pkl')
            with open(mapping_file, 'wb') as f:
                pickle.dump(item_mapping, f)
            logger.info("物品映射已保存: %s", mapping_file)
            
            # 测试索引
            logger.info("测试FAISS索引...")
            test_item_id = all_item_ids[0]
            test_vector = svd_model.get_item_embeddings([test_item_id])
            if test_vector is not None and len(test_vector) > 0:
                test_vector_norm = test_vector.copy().astype('float32')  # 确保是float32类型
                faiss.normalize_L2(test_vector_norm)
                D, I = index.search(test_vector_norm, 5)
                
                logger.info("物品 %s 的相似物品:", test_item_id)
                for i in range(min(3, len(I[0]))):
                    similar_item_id = all_item_ids[I[0][i]]
                    similarity = D[0][i]
                    logger.info("  物品ID: %s, 相似度: %.4f", similar_item_id, similarity)
            else:
                logger.warning("无法获取测试向量")
            
            # 简单的推荐测试
            logger.info("测试用户推荐...")
            test_user_id = train_ratings['user_id'].iloc[0]
            recommendations = svd_model.get_user_recommendations(test_user_id, top_k=5)
            
            if recommendations:
                logger.info("用户 %d 的推荐:", test_user_id)
                for item_id, score in recommendations[:3]:
                    logger.info("  物品ID: %d, 预测评分: %.4f", item_id, score)
            else:
                logger.warning("推荐结果为空")
        
        # 保存示例数据
        save_data_files(ratings, movies, users)
        
        logger.info("模型训练和索引构建完成！")
        return True
        
    except Exception as e:
        logger.error(f"训练过程失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
