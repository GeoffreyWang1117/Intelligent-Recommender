#!/usr/bin/env python3
"""
简化的模型训练脚本
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# 添加项目根目录到Python路径
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')

from models.svd_model import SVDRecommender
from models.faiss_index import FAISSIndex
from utils.data_loader import create_sample_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主训练流程"""
    logger.info("开始模型训练...")
    
    # 1. 创建示例数据
    logger.info("创建示例数据...")
    n_users = 500
    n_items = 200
    n_ratings = 5000
    
    ratings_df = create_sample_data(n_users, n_items, n_ratings)
    logger.info(f"生成数据: {n_users} 用户, {n_items} 电影, {len(ratings_df)} 评分")
    
    # 保存示例数据
    data_dir = 'data/sample'
    os.makedirs(data_dir, exist_ok=True)
    ratings_df.to_csv(f'{data_dir}/ratings.csv', index=False)
    
    # 2. 训练SVD模型
    logger.info("训练SVD推荐模型...")
    model = SVDRecommender(n_components=50)
    model.fit(ratings_df)
    
    # 3. 保存模型
    model_dir = 'models/saved'
    os.makedirs(model_dir, exist_ok=True)
    model_path = f'{model_dir}/lightfm_model.pkl'
    model.save_model(model_path)
    logger.info(f"SVD模型已保存: {model_path}")
    
    # 4. 测试推荐
    logger.info("测试推荐功能...")
    user_id = 1
    recommendations = model.get_user_recommendations(user_id, top_k=5)
    logger.info(f"用户 {user_id} 的推荐: {recommendations}")
    
    # 5. 构建FAISS索引
    logger.info("构建FAISS索引...")
    try:
        faiss_index = FAISSIndex(dimension=model.n_components)
        
        # 使用模型的物品特征构建索引
        if model.is_trained and model.item_features is not None:
            # 创建物品ID列表
            item_ids = list(model.item_encoder.classes_)
            faiss_index.build_index(model.item_features, item_ids)
            
            # 保存索引
            index_path = f'{model_dir}/faiss_index.bin'
            faiss_index.save_index(index_path)
            logger.info(f"FAISS索引已保存: {index_path}")
            
            # 测试相似物品搜索
            item_id = 1
            similar_items = faiss_index.search_similar_items(item_id, k=5)
            logger.info(f"物品 {item_id} 的相似物品: {similar_items}")
    except Exception as e:
        logger.error(f"FAISS索引构建失败: {e}")
    
    logger.info("模型训练和索引构建完成！")

if __name__ == "__main__":
    main()
