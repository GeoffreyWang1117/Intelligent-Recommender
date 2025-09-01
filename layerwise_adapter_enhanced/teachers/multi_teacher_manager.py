"""
多Teacher管理器

统一管理和协调多个Teacher的融合系统
支持Ensemble + LLM Teacher的智能协作
"""

import torch
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any, Union, Optional
import time

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 使用条件导入避免相对导入问题
try:
    from . import BaseTeacher, TeacherType, TeacherCapability
    from .ensemble_teacher_adapter import EnsembleTeacherAdapter
    from .llama_teacher_adapter import LlamaTeacherAdapter
    from .fisher_guided_selector import FisherGuidedSelector
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
    
    class EnsembleTeacherAdapter(BaseTeacher):
        def __init__(self, config):
            super().__init__("ensemble", config)
    
    class LlamaTeacherAdapter(BaseTeacher):
        def __init__(self, config):
            super().__init__("llama", config)
    
    class FisherGuidedSelector:
        def __init__(self, config=None):
            pass
        def select_teachers(self, *args, **kwargs):
            return ["ensemble", "llama"]

class MultiTeacherManager:
    """多Teacher管理器
    
    功能：
    1. 统一管理多个Teacher实例
    2. 协调Teacher之间的协作
    3. 提供统一的预测接口
    4. 支持Fisher引导的智能选择
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Teachers容器
        self.teachers: Dict[str, BaseTeacher] = {}
        self.teacher_types: Dict[str, str] = {}
        
        # Fisher引导选择器
        self.fisher_selector = FisherGuidedSelector(
            config.get('fisher_config', {})
        )
        
        # 融合配置
        self.fusion_mode = config.get('fusion_mode', 'weighted')  # weighted, voting, adaptive
        self.default_weights = config.get('default_weights', {
            'ensemble': 0.6,
            'llama': 0.4
        })
        
        # 性能监控
        self.prediction_count = 0
        self.total_inference_time = 0.0
        self.teacher_performance = {}
        
        print(f"✅ MultiTeacherManager初始化完成")
        print(f"   - 融合模式: {self.fusion_mode}")
        print(f"   - 默认权重: {self.default_weights}")
        print(f"   - 设备: {self.device}")
        
    def add_teacher(self, teacher: BaseTeacher, teacher_id: Optional[str] = None) -> str:
        """添加Teacher"""
        try:
            if teacher_id is None:
                teacher_id = teacher.name
            
            self.teachers[teacher_id] = teacher
            self.teacher_types[teacher_id] = getattr(teacher, 'teacher_type', 'unknown')
            
            # 初始化性能监控
            self.teacher_performance[teacher_id] = {
                'prediction_count': 0,
                'total_time': 0.0,
                'accuracy_samples': [],
                'confidence_samples': []
            }
            
            print(f"✅ Teacher已添加: {teacher_id} (类型: {self.teacher_types[teacher_id]})")
            return teacher_id
            
        except Exception as e:
            print(f"❌ 添加Teacher失败: {e}")
            raise
    
    def initialize_default_teachers(self) -> bool:
        """初始化默认的Teachers (Ensemble + LLM)"""
        try:
            # 初始化Ensemble Teacher
            ensemble_config = self.config.get('ensemble_config', {})
            ensemble_teacher = EnsembleTeacherAdapter(ensemble_config)
            self.add_teacher(ensemble_teacher, 'ensemble')
            
            # 初始化LLM Teacher
            llm_config = self.config.get('llm_config', {})
            llm_teacher = LlamaTeacherAdapter(llm_config)
            self.add_teacher(llm_teacher, 'llama')
            
            print(f"✅ 默认Teachers初始化完成: {list(self.teachers.keys())}")
            return True
            
        except Exception as e:
            print(f"❌ 初始化默认Teachers失败: {e}")
            return False
    
    def predict(self, user_id: int, item_id: int, mode: str = 'fusion') -> float:
        """统一预测接口"""
        try:
            start_time = time.time()
            
            if not self.teachers:
                print("⚠️ 没有可用的Teachers")
                return 3.0
            
            # 构造查询上下文
            query_context = {
                'user_id': user_id,
                'item_id': item_id,
                'query_type': 'rating',
                'timestamp': time.time()
            }
            
            if mode == 'fusion':
                # 融合模式预测
                result = self._fusion_predict(user_id, item_id, query_context)
            elif mode == 'ensemble_only':
                # 仅使用Ensemble Teacher
                if 'ensemble' in self.teachers:
                    result = self.teachers['ensemble'].predict(user_id, item_id)
                else:
                    result = 3.0
            elif mode == 'llm_only':
                # 仅使用LLM Teacher
                if 'llama' in self.teachers:
                    result = self.teachers['llama'].predict(user_id, item_id)
                else:
                    result = 3.0
            else:
                # 默认融合模式
                result = self._fusion_predict(user_id, item_id, query_context)
            
            # 更新性能统计
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time)
            
            return result
            
        except Exception as e:
            print(f"❌ 多Teacher预测失败: {e}")
            return 3.0
    
    def _fusion_predict(self, user_id: int, item_id: int, query_context: Dict[str, Any]) -> float:
        """融合预测逻辑"""
        try:
            if self.fusion_mode == 'weighted':
                return self._weighted_fusion_predict(user_id, item_id, query_context)
            elif self.fusion_mode == 'voting':
                return self._voting_fusion_predict(user_id, item_id)
            elif self.fusion_mode == 'adaptive':
                return self._adaptive_fusion_predict(user_id, item_id, query_context)
            else:
                return self._weighted_fusion_predict(user_id, item_id, query_context)
                
        except Exception as e:
            print(f"❌ 融合预测失败: {e}")
            return 3.0
    
    def _weighted_fusion_predict(self, user_id: int, item_id: int, query_context: Dict[str, Any]) -> float:
        """加权融合预测"""
        try:
            # 获取Fisher引导的权重
            weights = self.fisher_selector.calculate_teacher_weights(self.teachers, query_context)
            
            # 收集各Teacher预测
            predictions = {}
            for teacher_id, teacher in self.teachers.items():
                try:
                    pred = teacher.predict(user_id, item_id)
                    predictions[teacher_id] = pred
                except Exception as e:
                    print(f"⚠️ Teacher {teacher_id} 预测失败: {e}")
                    predictions[teacher_id] = 3.0
            
            # 加权融合
            weighted_sum = 0.0
            total_weight = 0.0
            
            for teacher_id, pred in predictions.items():
                weight = weights.get(teacher_id, 0.0)
                weighted_sum += pred * weight
                total_weight += weight
            
            if total_weight > 0:
                final_prediction = weighted_sum / total_weight
            else:
                # 简单平均
                final_prediction = np.mean(list(predictions.values()))
            
            return float(final_prediction)
            
        except Exception as e:
            print(f"❌ 加权融合失败: {e}")
            return 3.0
    
    def _voting_fusion_predict(self, user_id: int, item_id: int) -> float:
        """投票融合预测"""
        try:
            predictions = []
            
            for teacher_id, teacher in self.teachers.items():
                try:
                    pred = teacher.predict(user_id, item_id)
                    predictions.append(pred)
                except Exception as e:
                    print(f"⚠️ Teacher {teacher_id} 预测失败: {e}")
                    predictions.append(3.0)
            
            if predictions:
                # 中位数投票
                return float(np.median(predictions))
            else:
                return 3.0
                
        except Exception as e:
            print(f"❌ 投票融合失败: {e}")
            return 3.0
    
    def _adaptive_fusion_predict(self, user_id: int, item_id: int, query_context: Dict[str, Any]) -> float:
        """自适应融合预测"""
        try:
            # 获取选择策略
            strategy = self.fisher_selector.get_selection_strategy(self.teachers, query_context)
            
            # 基于策略的自适应融合
            primary_teacher = strategy['primary_teacher']
            primary_weight = strategy['primary_weight']
            
            if primary_weight > 0.7:
                # 主Teacher权重很高，主要使用主Teacher
                if primary_teacher in self.teachers:
                    primary_pred = self.teachers[primary_teacher].predict(user_id, item_id)
                    
                    # 轻微融合其他Teacher
                    other_preds = []
                    for teacher_id, teacher in self.teachers.items():
                        if teacher_id != primary_teacher:
                            try:
                                pred = teacher.predict(user_id, item_id)
                                other_preds.append(pred)
                            except:
                                other_preds.append(3.0)
                    
                    if other_preds:
                        other_avg = np.mean(other_preds)
                        final_pred = primary_pred * 0.8 + other_avg * 0.2
                    else:
                        final_pred = primary_pred
                    
                    return float(final_pred)
            
            # 使用加权融合
            return self._weighted_fusion_predict(user_id, item_id, query_context)
            
        except Exception as e:
            print(f"❌ 自适应融合失败: {e}")
            return 3.0
    
    def get_teacher_embeddings(self, user_id: int, item_id: int) -> Dict[str, torch.Tensor]:
        """获取所有Teacher的嵌入"""
        try:
            embeddings = {}
            
            for teacher_id, teacher in self.teachers.items():
                try:
                    embedding = teacher.get_embeddings(user_id, item_id)
                    # 确保嵌入在正确的设备上
                    embedding = embedding.to(self.device)
                    embeddings[teacher_id] = embedding
                except Exception as e:
                    print(f"⚠️ 获取Teacher {teacher_id} 嵌入失败: {e}")
                    # 使用默认嵌入
                    default_dim = 64
                    embeddings[teacher_id] = torch.zeros(default_dim, device=self.device)
            
            return embeddings
            
        except Exception as e:
            print(f"❌ 获取Teacher嵌入失败: {e}")
            return {}
    
    def get_fusion_embeddings(self, user_id: int, item_id: int) -> torch.Tensor:
        """获取融合嵌入"""
        try:
            teacher_embeddings = self.get_teacher_embeddings(user_id, item_id)
            
            if not teacher_embeddings:
                return torch.zeros(64, device=self.device)
            
            # 获取融合权重
            query_context = {'user_id': user_id, 'item_id': item_id, 'query_type': 'embedding'}
            weights = self.fisher_selector.calculate_teacher_weights(self.teachers, query_context)
            
            # 统一嵌入维度
            embedding_dim = max(emb.size(0) for emb in teacher_embeddings.values())
            aligned_embeddings = []
            aligned_weights = []
            
            for teacher_id, embedding in teacher_embeddings.items():
                # 对齐维度
                if embedding.size(0) < embedding_dim:
                    padding = torch.zeros(embedding_dim - embedding.size(0), device=self.device)
                    embedding = torch.cat([embedding, padding])
                elif embedding.size(0) > embedding_dim:
                    embedding = embedding[:embedding_dim]
                
                aligned_embeddings.append(embedding)
                aligned_weights.append(weights.get(teacher_id, 0.0))
            
            # 加权融合
            if aligned_embeddings and sum(aligned_weights) > 0:
                weighted_embeddings = []
                for emb, weight in zip(aligned_embeddings, aligned_weights):
                    weighted_embeddings.append(emb * weight)
                
                fusion_embedding = torch.stack(weighted_embeddings).sum(dim=0)
                return fusion_embedding
            else:
                return torch.zeros(embedding_dim, device=self.device)
                
        except Exception as e:
            print(f"❌ 获取融合嵌入失败: {e}")
            return torch.zeros(64, device=self.device)
    
    def _update_performance_stats(self, inference_time: float):
        """更新性能统计"""
        try:
            self.prediction_count += 1
            self.total_inference_time += inference_time
            
        except Exception as e:
            print(f"❌ 更新性能统计失败: {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        try:
            avg_inference_time = (self.total_inference_time / self.prediction_count 
                                if self.prediction_count > 0 else 0.0)
            
            return {
                'total_teachers': len(self.teachers),
                'teacher_types': dict(self.teacher_types),
                'prediction_count': self.prediction_count,
                'avg_inference_time': avg_inference_time,
                'fusion_mode': self.fusion_mode,
                'teacher_performance': dict(self.teacher_performance),
                'fisher_stats': self.fisher_selector.get_fisher_statistics()
            }
            
        except Exception as e:
            print(f"❌ 获取管理器统计失败: {e}")
            return {}
