import numpy as np
import pickle
import os
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

class LightFMRecommender:
    """LightFM推荐模型封装类"""
    
    def __init__(self, components=100, learning_rate=0.05, loss='warp'):
        self.components = components
        self.learning_rate = learning_rate
        self.loss = loss
        self.model = None
        self.dataset = None
        self.user_features = None
        self.item_features = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def prepare_data(self, ratings_df, user_features_df=None, item_features_df=None):
        """准备训练数据"""
        try:
            # 创建dataset对象
            self.dataset = Dataset()
            
            # 获取所有用户和物品
            users = ratings_df['user_id'].unique()
            items = ratings_df['item_id'].unique()
            
            # 准备用户和物品特征
            user_feature_list = []
            item_feature_list = []
            
            if user_features_df is not None:
                user_feature_list = list(user_features_df.columns)
            if item_features_df is not None:
                item_feature_list = list(item_features_df.columns)
            
            # 构建dataset
            self.dataset.fit(
                users=users,
                items=items,
                user_features=user_feature_list,
                item_features=item_feature_list
            )
            
            # 创建映射
            self.user_mapping = self.dataset.mapping()[0]
            self.item_mapping = self.dataset.mapping()[2]
            self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
            self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
            
            # 构建交互矩阵
            interactions, weights = self.dataset.build_interactions(
                [(row['user_id'], row['item_id'], row['rating']) 
                 for _, row in ratings_df.iterrows()]
            )
            
            # 构建特征矩阵
            if user_features_df is not None:
                user_features_data = []
                for user_id in users:
                    features = user_features_df[user_features_df['user_id'] == user_id]
                    if not features.empty:
                        feature_list = [f"{col}:{features[col].iloc[0]}" 
                                      for col in user_feature_list]
                        user_features_data.append((user_id, feature_list))
                
                self.user_features = self.dataset.build_user_features(user_features_data)
            
            if item_features_df is not None:
                item_features_data = []
                for item_id in items:
                    features = item_features_df[item_features_df['item_id'] == item_id]
                    if not features.empty:
                        feature_list = [f"{col}:{features[col].iloc[0]}" 
                                      for col in item_feature_list]
                        item_features_data.append((item_id, feature_list))
                
                self.item_features = self.dataset.build_item_features(item_features_data)
            
            return interactions, weights
            
        except Exception as e:
            logger.error(f"数据准备失败: {e}")
            raise
    
    def train(self, interactions, weights=None, epochs=50, num_threads=4):
        """训练模型"""
        try:
            self.model = LightFM(
                no_components=self.components,
                learning_rate=self.learning_rate,
                loss=self.loss,
                random_state=42
            )
            
            logger.info(f"开始训练LightFM模型，epochs={epochs}")
            
            self.model.fit(
                interactions=interactions,
                sample_weight=weights,
                user_features=self.user_features,
                item_features=self.item_features,
                epochs=epochs,
                num_threads=num_threads,
                verbose=True
            )
            
            logger.info("模型训练完成")
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise
    
    def predict(self, user_id, item_ids=None, top_k=10):
        """为用户生成推荐"""
        try:
            if self.model is None:
                raise ValueError("模型未训练")
            
            # 检查用户是否存在
            if user_id not in self.user_mapping:
                logger.warning(f"用户 {user_id} 不在训练数据中")
                return []
            
            user_idx = self.user_mapping[user_id]
            
            if item_ids is None:
                # 推荐所有物品
                item_ids = list(self.item_mapping.keys())
            
            # 获取物品索引
            item_indices = []
            valid_item_ids = []
            for item_id in item_ids:
                if item_id in self.item_mapping:
                    item_indices.append(self.item_mapping[item_id])
                    valid_item_ids.append(item_id)
            
            if not item_indices:
                return []
            
            # 预测评分
            scores = self.model.predict(
                user_ids=user_idx,
                item_ids=item_indices,
                user_features=self.user_features,
                item_features=self.item_features,
                num_threads=4
            )
            
            # 排序并返回top_k
            item_scores = list(zip(valid_item_ids, scores))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            return item_scores[:top_k]
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return []
    
    def get_user_embeddings(self, user_ids=None):
        """获取用户嵌入向量"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        if user_ids is None:
            user_indices = list(range(len(self.user_mapping)))
        else:
            user_indices = [self.user_mapping[uid] for uid in user_ids 
                          if uid in self.user_mapping]
        
        embeddings = self.model.get_user_representations(
            features=self.user_features
        )[0]
        
        return embeddings[user_indices]
    
    def get_item_embeddings(self, item_ids=None):
        """获取物品嵌入向量"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        if item_ids is None:
            item_indices = list(range(len(self.item_mapping)))
        else:
            item_indices = [self.item_mapping[iid] for iid in item_ids 
                          if iid in self.item_mapping]
        
        embeddings = self.model.get_item_representations(
            features=self.item_features
        )[0]
        
        return embeddings[item_indices]
    
    def evaluate(self, test_interactions, k=10):
        """评估模型性能"""
        try:
            precision = precision_at_k(
                self.model, 
                test_interactions,
                user_features=self.user_features,
                item_features=self.item_features,
                k=k
            ).mean()
            
            auc = auc_score(
                self.model,
                test_interactions,
                user_features=self.user_features,
                item_features=self.item_features
            ).mean()
            
            return {
                'precision_at_k': precision,
                'auc_score': auc
            }
            
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return {}
    
    def save_model(self, model_path):
        """保存模型"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'dataset': self.dataset,
                'user_features': self.user_features,
                'item_features': self.item_features,
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'reverse_user_mapping': self.reverse_user_mapping,
                'reverse_item_mapping': self.reverse_item_mapping,
                'components': self.components,
                'learning_rate': self.learning_rate,
                'loss': self.loss
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"模型已保存到: {model_path}")
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            raise
    
    def load_model(self, model_path):
        """加载模型"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.dataset = model_data['dataset']
            self.user_features = model_data['user_features']
            self.item_features = model_data['item_features']
            self.user_mapping = model_data['user_mapping']
            self.item_mapping = model_data['item_mapping']
            self.reverse_user_mapping = model_data['reverse_user_mapping']
            self.reverse_item_mapping = model_data['reverse_item_mapping']
            self.components = model_data['components']
            self.learning_rate = model_data['learning_rate']
            self.loss = model_data['loss']
            
            logger.info(f"模型已从 {model_path} 加载")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
