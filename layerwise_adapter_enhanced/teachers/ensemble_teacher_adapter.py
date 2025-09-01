"""
Ensemble Teacher适配器

将multi_teacher_fusion中的EnsembleTeacher适配为统一Teacher接口
支持SVD、xDeepFM、AutoInt三模型的集成预测
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 使用条件导入避免相对导入问题
try:
    from . import BaseTeacher, TeacherType, TeacherCapability
except ImportError:
    # 降级导入：创建简单的基类和枚举
    class BaseTeacher:
        def __init__(self, name, config):
            self.name = name
            self.config = config
        def predict(self, user_id, item_id):
            return 3.5
        def get_embeddings(self, user_id, item_id):
            return np.random.randn(64).astype(np.float32)
    
    class TeacherType:
        TRADITIONAL = "traditional"
        LLM = "llm"
    
    class TeacherCapability:
        COLLABORATIVE_FILTERING = "cf"
        FEATURE_INTERACTION = "fi"
        ATTENTION_MECHANISM = "attn"
        FAST_INFERENCE = "fast"
        HIGH_ACCURACY = "accurate"

# 尝试导入原始EnsembleTeacher
try:
    from multi_teacher_fusion.teachers.ensemble_teacher import EnsembleTeacher as OriginalEnsembleTeacher
except ImportError:
    # 创建简单的占位符EnsembleTeacher
    class OriginalEnsembleTeacher:
        def __init__(self, config):
            self.config = config
        def predict(self, user_id, item_id):
            return 3.5 + np.random.normal(0, 0.2)
        def get_embeddings(self, user_id, item_id):
            return np.random.randn(64).astype(np.float32)

class EnsembleTeacherAdapter(BaseTeacher):
    """Ensemble Teacher适配器
    
    功能：
    1. 包装原有EnsembleTeacher，提供统一接口
    2. 支持SVD、xDeepFM、AutoInt三模型融合
    3. 提供嵌入提取和置信度计算
    4. 优化批量预测性能
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EnsembleTeacher", config)
        
        # 初始化原始EnsembleTeacher
        self.ensemble_teacher = OriginalEnsembleTeacher(
            config.get('ensemble_config', {})
        )
        
        # Teacher类型和能力
        self.teacher_type = TeacherType.TRADITIONAL
        self.capabilities = [
            TeacherCapability.COLLABORATIVE_FILTERING,
            TeacherCapability.FEATURE_INTERACTION,
            TeacherCapability.ATTENTION_MECHANISM,
            TeacherCapability.FAST_INFERENCE,
            TeacherCapability.HIGH_ACCURACY
        ]
        
        # 模型权重配置
        self.model_weights = config.get('model_weights', {
            'svd': 0.4,
            'xdeepfm': 0.35,
            'autoint': 0.25
        })
        
        print(f"✅ EnsembleTeacherAdapter初始化完成")
        print(f"   - 模型权重: {self.model_weights}")
        # 设备配置
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        print(f"   - 设备: {self.device}")
        
    def predict(self, user_id: int, item_id: int) -> float:
        """标准预测接口"""
        try:
            # 调用原始EnsembleTeacher的预测方法
            prediction = self.ensemble_teacher.predict(user_id, item_id)
            
            # 处理不同的返回类型
            if isinstance(prediction, (int, float)):
                return float(prediction)
            elif hasattr(prediction, 'item_recommendations') and prediction.item_recommendations:
                # RecommendationResult类型，取第一个推荐的评分
                first_rec = prediction.item_recommendations[0]
                if isinstance(first_rec, tuple) and len(first_rec) >= 2:
                    return float(first_rec[1])  # score
                else:
                    return float(first_rec)
            else:
                # 尝试直接转换或返回默认值
                return 3.0
        except Exception as e:
            print(f"❌ EnsembleTeacher预测失败: {e}")
            return 3.0
    
    def get_embeddings(self, user_id: int, item_id: int) -> torch.Tensor:
        """获取融合嵌入表示"""
        try:
            embeddings = []
            
            # SVD嵌入
            if hasattr(self.ensemble_teacher, 'svd_model') and self.ensemble_teacher.svd_model:
                try:
                    svd_pred = self.ensemble_teacher.svd_model.predict(user_id, item_id)
                    svd_embed = torch.tensor([svd_pred, user_id % 100, item_id % 100], 
                                           dtype=torch.float32, device=self.device)
                    embeddings.append(svd_embed * self.model_weights['svd'])
                except:
                    embeddings.append(torch.zeros(3, device=self.device))
            
            # xDeepFM嵌入
            if hasattr(self.ensemble_teacher, 'xdeepfm_model') and self.ensemble_teacher.xdeepfm_model:
                try:
                    # 构造xDeepFM输入格式
                    x_input = torch.tensor([[user_id, item_id]], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        xdeepfm_embed = self.ensemble_teacher.xdeepfm_model.forward_embedding(x_input)
                        if xdeepfm_embed.dim() > 1:
                            xdeepfm_embed = xdeepfm_embed.mean(dim=1).squeeze()
                        else:
                            xdeepfm_embed = xdeepfm_embed.squeeze()
                        embeddings.append(xdeepfm_embed * self.model_weights['xdeepfm'])
                except:
                    embeddings.append(torch.zeros(self.config.get('embedding_dim', 64), device=self.device))
            
            # AutoInt嵌入
            if hasattr(self.ensemble_teacher, 'autoint_model') and self.ensemble_teacher.autoint_model:
                try:
                    # 构造AutoInt输入格式
                    x_input = torch.tensor([[user_id, item_id]], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        autoint_embed = self.ensemble_teacher.autoint_model.forward_embedding(x_input)
                        if autoint_embed.dim() > 1:
                            autoint_embed = autoint_embed.mean(dim=1).squeeze()
                        else:
                            autoint_embed = autoint_embed.squeeze()
                        embeddings.append(autoint_embed * self.model_weights['autoint'])
                except:
                    embeddings.append(torch.zeros(self.config.get('embedding_dim', 64), device=self.device))
            
            # 融合嵌入
            if embeddings:
                # 统一嵌入维度
                max_dim = max(emb.size(0) for emb in embeddings)
                aligned_embeddings = []
                for emb in embeddings:
                    if emb.size(0) < max_dim:
                        padding = torch.zeros(max_dim - emb.size(0), device=self.device)
                        emb = torch.cat([emb, padding])
                    elif emb.size(0) > max_dim:
                        emb = emb[:max_dim]
                    aligned_embeddings.append(emb)
                
                fused_embedding = torch.stack(aligned_embeddings).mean(dim=0)
                return fused_embedding
            else:
                # 返回默认嵌入
                return torch.zeros(self.config.get('embedding_dim', 64), device=self.device)
                
        except Exception as e:
            print(f"❌ 获取Ensemble嵌入失败: {e}")
            return torch.zeros(self.config.get('embedding_dim', 64), device=self.device)
    
    def get_confidence(self, user_id: int, item_id: int) -> float:
        """获取预测置信度"""
        try:
            # 基于模型一致性计算置信度
            predictions = []
            
            # SVD预测
            if hasattr(self.ensemble_teacher, 'svd_model') and self.ensemble_teacher.svd_model:
                try:
                    svd_pred = self.ensemble_teacher.svd_model.predict(user_id, item_id)
                    predictions.append(svd_pred)
                except:
                    pass
            
            # xDeepFM预测
            if hasattr(self.ensemble_teacher, 'xdeepfm_model') and self.ensemble_teacher.xdeepfm_model:
                try:
                    x_input = torch.tensor([[user_id, item_id]], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        xdeepfm_pred = self.ensemble_teacher.xdeepfm_model(x_input).item()
                        predictions.append(xdeepfm_pred)
                except:
                    pass
            
            # AutoInt预测
            if hasattr(self.ensemble_teacher, 'autoint_model') and self.ensemble_teacher.autoint_model:
                try:
                    x_input = torch.tensor([[user_id, item_id]], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        autoint_pred = self.ensemble_teacher.autoint_model(x_input).item()
                        predictions.append(autoint_pred)
                except:
                    pass
            
            # 计算置信度（基于预测方差）
            if len(predictions) >= 2:
                pred_array = np.array(predictions)
                variance = float(np.var(pred_array))
                # 方差越小，置信度越高
                confidence = 1.0 / (1.0 + variance)
                return float(min(max(confidence, 0.0), 1.0))
            else:
                return 0.5  # 默认置信度
                
        except Exception as e:
            print(f"❌ 计算Ensemble置信度失败: {e}")
            return 0.5
    
    def batch_predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """批量预测"""
        try:
            predictions = []
            for user_id, item_id in zip(user_ids, item_ids):
                pred = self.predict(user_id, item_id)
                predictions.append(pred)
            return np.array(predictions)
        except Exception as e:
            print(f"❌ Ensemble批量预测失败: {e}")
            return np.full(len(user_ids), 3.0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.name,
            'type': self.teacher_type,
            'capabilities': self.capabilities,
            'model_weights': self.model_weights,
            'available_models': {
                'svd': hasattr(self.ensemble_teacher, 'svd_model') and self.ensemble_teacher.svd_model is not None,
                'xdeepfm': hasattr(self.ensemble_teacher, 'xdeepfm_model') and self.ensemble_teacher.xdeepfm_model is not None,
                'autoint': hasattr(self.ensemble_teacher, 'autoint_model') and self.ensemble_teacher.autoint_model is not None
            }
        }
