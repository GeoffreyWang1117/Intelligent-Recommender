"""
Fisher引导的Teacher选择器

基于Fisher信息矩阵的智能Teacher权重分配和选择策略
结合已完成的Fisher/贝叶斯分析结果
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import json
import os

from . import BaseTeacher, TeacherType, TeacherCapability

class FisherGuidedSelector:
    """Fisher引导的Teacher选择器
    
    功能：
    1. 基于Fisher信息矩阵计算Teacher权重
    2. 动态调整Teacher选择策略
    3. 查询特性感知的权重分配
    4. 集成已完成的Fisher/贝叶斯分析
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Fisher信息矩阵相关配置
        self.fisher_weight_decay = config.get('fisher_weight_decay', 0.95)
        self.min_weight_threshold = config.get('min_weight_threshold', 0.1)
        self.max_weight_threshold = config.get('max_weight_threshold', 0.9)
        
        # 动态权重历史
        self.weight_history = {}
        self.performance_history = {}
        
        # 预加载的Fisher分析结果
        self.fisher_analysis_path = config.get('fisher_analysis_path', 
                                             'multi_teacher_fusion/analysis/fisher_analysis_results.json')
        self.fisher_results = self._load_fisher_analysis()
        
        # Teacher能力映射
        self.teacher_capability_weights = {
            TeacherCapability.SEMANTIC_UNDERSTANDING: {
                'user_complexity': 0.8,  # 复杂用户偏好语义理解权重高
                'item_novelty': 0.7,     # 新颖物品需要语义理解
                'context_richness': 0.9  # 丰富上下文语义权重高
            },
            TeacherCapability.COLLABORATIVE_FILTERING: {
                'user_popularity': 0.8,  # 流行用户协同过滤权重高
                'item_popularity': 0.9,  # 流行物品协同过滤效果好
                'historical_richness': 0.8  # 丰富历史记录权重高
            },
            TeacherCapability.FEATURE_INTERACTION: {
                'feature_complexity': 0.9,  # 复杂特征交互权重高
                'cross_domain': 0.7,        # 跨域推荐特征交互重要
                'precision_demand': 0.8     # 精确性要求高时权重高
            },
            TeacherCapability.ATTENTION_MECHANISM: {
                'sequence_length': 0.8,     # 长序列注意力权重高
                'multi_aspect': 0.9,        # 多方面考虑注意力重要
                'personalization': 0.7      # 个性化需求注意力权重高
            }
        }
        
        print(f"✅ FisherGuidedSelector初始化完成")
        print(f"   - Fisher权重衰减: {self.fisher_weight_decay}")
        print(f"   - 权重阈值: [{self.min_weight_threshold}, {self.max_weight_threshold}]")
        print(f"   - Fisher分析结果: {'已加载' if self.fisher_results else '未找到'}")
        
    def _load_fisher_analysis(self) -> Dict[str, Any]:
        """加载预计算的Fisher分析结果"""
        try:
            if os.path.exists(self.fisher_analysis_path):
                with open(self.fisher_analysis_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"✅ Fisher分析结果已加载: {len(results.get('teacher_importance', {}))}个Teacher")
                return results
            else:
                print(f"⚠️ Fisher分析文件未找到: {self.fisher_analysis_path}")
                return {}
        except Exception as e:
            print(f"❌ 加载Fisher分析失败: {e}")
            return {}
    
    def calculate_teacher_weights(self, 
                                teachers: Dict[str, BaseTeacher],
                                query_context: Dict[str, Any]) -> Dict[str, float]:
        """计算基于Fisher信息的Teacher权重"""
        try:
            weights = {}
            
            # 1. 基础Fisher权重
            fisher_weights = self._get_fisher_weights(teachers.keys())
            
            # 2. 查询上下文权重
            context_weights = self._get_context_weights(teachers, query_context)
            
            # 3. 性能历史权重
            performance_weights = self._get_performance_weights(teachers.keys())
            
            # 4. 融合权重
            for teacher_name in teachers.keys():
                fisher_w = fisher_weights.get(teacher_name, 0.33)
                context_w = context_weights.get(teacher_name, 0.33)
                perf_w = performance_weights.get(teacher_name, 0.33)
                
                # 加权融合
                combined_weight = (
                    fisher_w * 0.4 +      # Fisher信息权重40%
                    context_w * 0.4 +     # 上下文权重40%
                    perf_w * 0.2          # 历史性能权重20%
                )
                
                # 应用权重阈值
                weights[teacher_name] = max(
                    self.min_weight_threshold,
                    min(self.max_weight_threshold, combined_weight)
                )
            
            # 5. 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            else:
                # 默认均等权重
                equal_weight = 1.0 / len(teachers)
                weights = {k: equal_weight for k in teachers.keys()}
            
            return weights
            
        except Exception as e:
            print(f"❌ 计算Teacher权重失败: {e}")
            # 返回默认均等权重
            equal_weight = 1.0 / len(teachers)
            return {k: equal_weight for k in teachers.keys()}
    
    def _get_fisher_weights(self, teacher_names) -> Dict[str, float]:
        """获取基于Fisher信息的权重"""
        try:
            # 确保teacher_names是字符串列表
            if not isinstance(teacher_names, list):
                teacher_names = list(teacher_names)
            
            # 处理Teacher对象，提取名称
            names = []
            for name in teacher_names:
                if hasattr(name, '__class__'):
                    # 如果是Teacher对象，提取类名
                    class_name = name.__class__.__name__.lower()
                    if 'ensemble' in class_name:
                        names.append('ensemble')
                    elif 'llama' in class_name:
                        names.append('llama')
                    else:
                        names.append(str(name))
                else:
                    names.append(str(name))
            
            if not self.fisher_results or 'teacher_importance' not in self.fisher_results:
                # 默认权重配置
                return self._get_default_fisher_weights(names)
            
            fisher_importance = self.fisher_results['teacher_importance']
            weights = {}
            
            for i, name in enumerate(names):
                # 查找匹配的Teacher重要性
                importance = 0.33  # 默认重要性
                
                for teacher_key, importance_value in fisher_importance.items():
                    if any(keyword in name.lower() for keyword in 
                          [teacher_key.lower(), teacher_key.replace('_', '').lower()]):
                        importance = float(importance_value)
                        break
                
                # 使用原始键（可能是Teacher对象）作为权重键
                original_key = teacher_names[i] if i < len(teacher_names) else name
                weights[original_key] = importance
            
            return weights
            
        except Exception as e:
            print(f"❌ 获取Fisher权重失败: {e}")
            return self._get_default_fisher_weights(teacher_names)
    
    def _get_default_fisher_weights(self, teacher_names: List[str]) -> Dict[str, float]:
        """获取默认Fisher权重"""
        default_weights = {
            'ensemble': 0.45,    # Ensemble Teacher基础权重高
            'llama': 0.35,       # LLM Teacher语义权重中等
            'traditional': 0.20  # 传统方法权重较低
        }
        
        weights = {}
        for name in teacher_names:
            name_lower = name.lower()
            if 'ensemble' in name_lower:
                weights[name] = default_weights['ensemble']
            elif 'llama' in name_lower or 'llm' in name_lower:
                weights[name] = default_weights['llama']
            else:
                weights[name] = default_weights['traditional']
        
        return weights
    
    def _get_context_weights(self, 
                           teachers: Dict[str, BaseTeacher],
                           context: Dict[str, Any]) -> Dict[str, float]:
        """基于查询上下文计算权重"""
        try:
            weights = {}
            
            # 提取上下文特征
            user_id = context.get('user_id', 0)
            item_id = context.get('item_id', 0)
            query_type = context.get('query_type', 'rating')
            
            # 计算上下文特征
            user_complexity = min(1.0, (user_id % 100) / 100.0)  # 用户复杂度
            item_novelty = min(1.0, (item_id % 50) / 50.0)       # 物品新颖度
            
            for name, teacher in teachers.items():
                context_weight = 0.33  # 默认权重
                
                # 根据Teacher类型设置默认权重  
                name_lower = str(name).lower()
                class_name = teacher.__class__.__name__.lower()
                
                if 'ensemble' in name_lower or 'ensemble' in class_name:
                    context_weight = 0.45 + user_complexity * 0.1  # 复杂用户偏好Ensemble
                elif 'llama' in name_lower or 'llama' in class_name:
                    context_weight = 0.35 + item_novelty * 0.15   # 新颖物品偏好LLM语义理解
                else:
                    context_weight = 0.2 + (1.0 - user_complexity) * 0.1  # 简单用户偏好传统方法
                
                weights[name] = context_weight
            
            return weights
            
        except Exception as e:
            print(f"❌ 计算上下文权重失败: {e}")
            return {name: 0.33 for name in teachers.keys()}
    
    def _get_performance_weights(self, teacher_names: List[str]) -> Dict[str, float]:
        """基于历史性能计算权重"""
        try:
            weights = {}
            
            for name in teacher_names:
                if name in self.performance_history:
                    # 基于历史表现计算权重
                    history = self.performance_history[name]
                    avg_performance = np.mean(history.get('accuracy', [0.5]))
                    weight = min(1.0, max(0.1, avg_performance))
                else:
                    # 新Teacher默认权重
                    weight = 0.33
                
                weights[name] = weight
            
            return weights
            
        except Exception as e:
            print(f"❌ 计算性能权重失败: {e}")
            return {name: 0.33 for name in teacher_names}
    
    def update_performance(self, teacher_name: str, performance_metrics: Dict[str, float]):
        """更新Teacher性能记录"""
        try:
            if teacher_name not in self.performance_history:
                self.performance_history[teacher_name] = {
                    'accuracy': [],
                    'confidence': [],
                    'response_time': []
                }
            
            history = self.performance_history[teacher_name]
            
            # 更新各项指标
            if 'accuracy' in performance_metrics:
                history['accuracy'].append(performance_metrics['accuracy'])
                # 保持历史记录长度
                if len(history['accuracy']) > 100:
                    history['accuracy'] = history['accuracy'][-100:]
            
            if 'confidence' in performance_metrics:
                history['confidence'].append(performance_metrics['confidence'])
                if len(history['confidence']) > 100:
                    history['confidence'] = history['confidence'][-100:]
            
            if 'response_time' in performance_metrics:
                history['response_time'].append(performance_metrics['response_time'])
                if len(history['response_time']) > 100:
                    history['response_time'] = history['response_time'][-100:]
            
        except Exception as e:
            print(f"❌ 更新Teacher性能失败: {e}")
    
    def get_selection_strategy(self, 
                             teachers: Dict[str, BaseTeacher],
                             query_context: Dict[str, Any]) -> Dict[str, Any]:
        """获取Teacher选择策略"""
        try:
            # 计算权重
            weights = self.calculate_teacher_weights(teachers, query_context)
            
            # 选择策略
            max_weight_teacher = max(weights.items(), key=lambda x: x[1])
            
            strategy = {
                'weights': weights,
                'primary_teacher': max_weight_teacher[0],
                'primary_weight': max_weight_teacher[1],
                'selection_mode': 'fisher_guided',
                'context_factors': {
                    'user_complexity': min(1.0, (query_context.get('user_id', 0) % 100) / 100.0),
                    'item_novelty': min(1.0, (query_context.get('item_id', 0) % 50) / 50.0),
                    'query_type': query_context.get('query_type', 'rating')
                }
            }
            
            return strategy
            
        except Exception as e:
            print(f"❌ 获取选择策略失败: {e}")
            return {
                'weights': {name: 1.0/len(teachers) for name in teachers.keys()},
                'primary_teacher': list(teachers.keys())[0] if teachers else 'default',
                'primary_weight': 1.0/len(teachers) if teachers else 1.0,
                'selection_mode': 'equal',
                'context_factors': {}
            }
    
    def get_fisher_statistics(self) -> Dict[str, Any]:
        """获取Fisher选择器统计信息"""
        return {
            'fisher_analysis_loaded': bool(self.fisher_results),
            'performance_history_size': len(self.performance_history),
            'weight_history_size': len(self.weight_history),
            'teacher_capabilities': len(self.teacher_capability_weights),
            'config': {
                'weight_decay': self.fisher_weight_decay,
                'weight_thresholds': [self.min_weight_threshold, self.max_weight_threshold]
            }
        }
