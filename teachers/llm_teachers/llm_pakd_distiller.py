#!/usr/bin/env python3
"""
LLM PAKD (Pruning-Aware Knowledge Distillation) Implementation
LLM专用剪枝感知知识蒸馏实现

功能:
1. LLM Teacher-Student知识蒸馏
2. 剪枝感知的推荐质量保持
3. Llama3-Qwen3 跨模型知识传递
4. 推荐性能优化和压缩

作者: GitHub Copilot
日期: 2025-08-29
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class LLMPAKDDistiller:
    """LLM专用PAKD蒸馏器"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7, 
                 pruning_ratio: float = 0.1, device: str = None):
        """
        初始化PAKD蒸馏器
        
        Args:
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            pruning_ratio: 剪枝比例
            device: 计算设备
        """
        self.temperature = temperature
        self.alpha = alpha
        self.pruning_ratio = pruning_ratio
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        print(f"🧠 LLM PAKD Distiller 初始化")
        print(f"   Temperature: {self.temperature}")
        print(f"   Alpha: {self.alpha}")
        print(f"   Pruning Ratio: {self.pruning_ratio}")
        print(f"   Device: {self.device}")
        
        # 蒸馏统计
        self.distillation_stats = {
            'teacher_data_count': 0,
            'student_data_count': 0,
            'distillation_loss_history': [],
            'pruning_mask_history': [],
            'performance_metrics': {}
        }
        
    def run_llm_pakd(self, teacher_data: List[Dict], student_data: List[Dict],
                    distillation_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        运行LLM PAKD实验
        
        Args:
            teacher_data: Teacher模型数据
            student_data: Student模型数据
            distillation_config: 蒸馏配置
            
        Returns:
            Dict: PAKD实验结果
        """
        print(f"🧠 开始LLM PAKD实验...")
        print(f"   Teacher数据: {len(teacher_data)} 样本")
        print(f"   Student数据: {len(student_data)} 样本")
        
        if not teacher_data or not student_data:
            print("❌ Teacher或Student数据为空")
            return {}
        
        # 更新配置
        if distillation_config:
            self.temperature = distillation_config.get('temperature', self.temperature)
            self.alpha = distillation_config.get('alpha', self.alpha)
            self.pruning_ratio = distillation_config.get('pruning_ratio', self.pruning_ratio)
        
        # 1. 数据预处理和对齐
        aligned_data = self._align_teacher_student_data(teacher_data, student_data)
        
        # 2. 特征提取和向量化
        teacher_features, student_features = self._extract_features(aligned_data)
        
        # 3. 知识蒸馏
        distillation_results = self._perform_knowledge_distillation(teacher_features, student_features)
        
        # 4. 剪枝感知优化
        pruning_results = self._apply_pruning_aware_optimization(
            teacher_features, student_features, distillation_results
        )
        
        # 5. 性能评估
        evaluation_results = self._evaluate_pakd_performance(
            aligned_data, teacher_features, student_features, pruning_results
        )
        
        # 6. 汇总结果
        final_results = {
            'config': {
                'temperature': self.temperature,
                'alpha': self.alpha,
                'pruning_ratio': self.pruning_ratio,
                'device': str(self.device)
            },
            'data_info': {
                'teacher_samples': len(teacher_data),
                'student_samples': len(student_data),
                'aligned_pairs': len(aligned_data)
            },
            'distillation_results': distillation_results,
            'pruning_results': pruning_results,
            'evaluation_results': evaluation_results,
            'distillation_stats': self.distillation_stats
        }
        
        print("✅ LLM PAKD实验完成")
        return final_results
    
    def _align_teacher_student_data(self, teacher_data: List[Dict], 
                                   student_data: List[Dict]) -> List[Dict]:
        """对齐Teacher和Student数据"""
        print("🔄 对齐Teacher-Student数据...")
        
        aligned_pairs = []
        
        # 按用户ID对齐数据
        teacher_by_user = {item.get('user_id'): item for item in teacher_data}
        student_by_user = {item.get('user_id'): item for item in student_data}
        
        common_users = set(teacher_by_user.keys()) & set(student_by_user.keys())
        
        for user_id in common_users:
            if user_id is not None:
                aligned_pairs.append({
                    'user_id': user_id,
                    'teacher': teacher_by_user[user_id],
                    'student': student_by_user[user_id]
                })
        
        print(f"✅ 对齐完成: {len(aligned_pairs)} 个Teacher-Student对")
        return aligned_pairs
    
    def _extract_features(self, aligned_data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取Teacher和Student特征"""
        print("🔧 提取特征向量...")
        
        teacher_features = []
        student_features = []
        
        for pair in aligned_data:
            try:
                # 提取Teacher特征
                teacher_feat = self._extract_single_features(pair['teacher'])
                student_feat = self._extract_single_features(pair['student'])
                
                teacher_features.append(teacher_feat)
                student_features.append(student_feat)
                
            except Exception as e:
                print(f"⚠️  特征提取失败: {e}")
                continue
        
        if not teacher_features or not student_features:
            print("❌ 没有有效的特征向量")
            return torch.empty(0, 10), torch.empty(0, 10)
        
        # 转换为tensor
        teacher_tensor = torch.stack(teacher_features).to(self.device)
        student_tensor = torch.stack(student_features).to(self.device)
        
        print(f"✅ 特征提取完成: Teacher {teacher_tensor.shape}, Student {student_tensor.shape}")
        return teacher_tensor, student_tensor
    
    def _extract_single_features(self, recommendation_data: Dict) -> torch.Tensor:
        """提取单个推荐的特征向量"""
        features = {}
        
        # 基本信息特征
        features['model_type'] = 1.0 if 'llama' in str(recommendation_data.get('model', '')).lower() else 0.0
        features['timestamp'] = float(recommendation_data.get('timestamp', 0)) / 1e9  # 归一化时间戳
        
        # 推荐特征
        recommendations = recommendation_data.get('recommendations', [])
        features['rec_count'] = min(len(recommendations) / 10.0, 1.0)  # 推荐数量归一化
        
        # 推荐质量特征
        if recommendations:
            features['has_reasons'] = sum(1 for rec in recommendations if rec.get('reason')) / len(recommendations)
            features['avg_reason_length'] = np.mean([len(rec.get('reason', '')) for rec in recommendations]) / 100.0
            features['title_diversity'] = len(set(rec.get('title', '') for rec in recommendations)) / len(recommendations)
        else:
            features['has_reasons'] = 0.0
            features['avg_reason_length'] = 0.0
            features['title_diversity'] = 0.0
        
        # LLM响应特征
        llm_response = recommendation_data.get('llm_response', '')
        features['response_length'] = min(len(llm_response) / 1000.0, 1.0)  # 响应长度归一化
        features['json_structure'] = 1.0 if '{' in llm_response and '}' in llm_response else 0.0
        features['chinese_ratio'] = sum(1 for char in llm_response if '\u4e00' <= char <= '\u9fff') / max(len(llm_response), 1)
        
        # 填充到固定维度（10维）
        feature_values = list(features.values())
        while len(feature_values) < 10:
            feature_values.append(0.0)
        
        # 归一化和处理NaN
        feature_values = np.array(feature_values[:10], dtype=np.float32)
        feature_values = np.nan_to_num(feature_values)
        
        return torch.tensor(feature_values, dtype=torch.float32)
    
    def _perform_knowledge_distillation(self, teacher_features: torch.Tensor, 
                                      student_features: torch.Tensor) -> Dict[str, Any]:
        """执行知识蒸馏"""
        print("🎓 执行知识蒸馏...")
        
        if teacher_features.size(0) == 0 or student_features.size(0) == 0:
            return {'error': 'Empty feature tensors'}
        
        # 创建简单的蒸馏网络
        teacher_net = nn.Linear(teacher_features.size(1), teacher_features.size(1)).to(self.device)
        student_net = nn.Linear(student_features.size(1), student_features.size(1)).to(self.device)
        
        # 初始化网络权重
        with torch.no_grad():
            teacher_net.weight.copy_(torch.eye(teacher_features.size(1)))
            student_net.weight.copy_(torch.eye(student_features.size(1)))
        
        # 计算Teacher和Student的输出
        teacher_outputs = teacher_net(teacher_features)
        student_outputs = student_net(student_features)
        
        # 计算蒸馏损失
        distillation_loss = self._compute_distillation_loss(teacher_outputs, student_outputs)
        
        # 计算特征对齐损失
        alignment_loss = self._compute_alignment_loss(teacher_features, student_features)
        
        # 总损失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * alignment_loss
        
        # 计算信息传递效率
        information_transfer = self._compute_information_transfer(teacher_outputs, student_outputs)
        
        results = {
            'distillation_loss': float(distillation_loss),
            'alignment_loss': float(alignment_loss),
            'total_loss': float(total_loss),
            'information_transfer_efficiency': information_transfer,
            'teacher_output_stats': self._compute_tensor_stats(teacher_outputs),
            'student_output_stats': self._compute_tensor_stats(student_outputs)
        }
        
        # 记录统计信息
        self.distillation_stats['distillation_loss_history'].append(float(total_loss))
        
        return results
    
    def _compute_distillation_loss(self, teacher_outputs: torch.Tensor, 
                                 student_outputs: torch.Tensor) -> torch.Tensor:
        """计算蒸馏损失"""
        # 使用温度软化的KL散度
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        return kl_loss * (self.temperature ** 2)
    
    def _compute_alignment_loss(self, teacher_features: torch.Tensor, 
                              student_features: torch.Tensor) -> torch.Tensor:
        """计算特征对齐损失"""
        # 使用MSE损失进行特征对齐
        min_size = min(teacher_features.size(0), student_features.size(0))
        teacher_aligned = teacher_features[:min_size]
        student_aligned = student_features[:min_size]
        
        return F.mse_loss(teacher_aligned, student_aligned)
    
    def _compute_information_transfer(self, teacher_outputs: torch.Tensor, 
                                    student_outputs: torch.Tensor) -> Dict[str, float]:
        """计算信息传递效率"""
        min_size = min(teacher_outputs.size(0), student_outputs.size(0))
        teacher_aligned = teacher_outputs[:min_size]
        student_aligned = student_outputs[:min_size]
        
        # 计算相关性
        teacher_flat = teacher_aligned.flatten()
        student_flat = student_aligned.flatten()
        
        if len(teacher_flat) > 1 and len(student_flat) > 1:
            correlation = torch.corrcoef(torch.stack([teacher_flat, student_flat]))[0, 1]
            correlation = float(correlation) if not torch.isnan(correlation) else 0.0
        else:
            correlation = 0.0
        
        # 计算信息保持率
        teacher_entropy = self._compute_entropy(teacher_aligned)
        student_entropy = self._compute_entropy(student_aligned)
        
        information_retention = float(student_entropy / (teacher_entropy + 1e-8))
        
        return {
            'correlation': correlation,
            'information_retention': information_retention,
            'teacher_entropy': float(teacher_entropy),
            'student_entropy': float(student_entropy)
        }
    
    def _compute_entropy(self, tensor: torch.Tensor) -> torch.Tensor:
        """计算张量的熵"""
        # 归一化到概率分布
        probs = F.softmax(tensor.flatten(), dim=0)
        
        # 计算熵
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy
    
    def _apply_pruning_aware_optimization(self, teacher_features: torch.Tensor,
                                        student_features: torch.Tensor,
                                        distillation_results: Dict) -> Dict[str, Any]:
        """应用剪枝感知优化"""
        print("✂️ 应用剪枝感知优化...")
        
        if teacher_features.size(0) == 0:
            return {'error': 'Empty teacher features'}
        
        # 计算特征重要性
        feature_importance = self._compute_feature_importance(teacher_features, student_features)
        
        # 生成剪枝掩码
        pruning_mask = self._generate_pruning_mask(feature_importance)
        
        # 应用剪枝
        pruned_teacher_features = self._apply_pruning(teacher_features, pruning_mask)
        pruned_student_features = self._apply_pruning(student_features, pruning_mask)
        
        # 评估剪枝后的性能
        pruning_performance = self._evaluate_pruning_performance(
            teacher_features, student_features, 
            pruned_teacher_features, pruned_student_features
        )
        
        results = {
            'feature_importance': feature_importance.tolist() if isinstance(feature_importance, torch.Tensor) else feature_importance,
            'pruning_mask': pruning_mask.tolist() if isinstance(pruning_mask, torch.Tensor) else pruning_mask,
            'pruning_ratio_actual': float(1 - pruning_mask.sum() / len(pruning_mask)),
            'pruning_performance': pruning_performance,
            'original_feature_count': teacher_features.size(1),
            'remaining_feature_count': int(pruning_mask.sum())
        }
        
        # 记录剪枝掩码
        self.distillation_stats['pruning_mask_history'].append(pruning_mask.tolist() if isinstance(pruning_mask, torch.Tensor) else pruning_mask)
        
        return results
    
    def _compute_feature_importance(self, teacher_features: torch.Tensor, 
                                  student_features: torch.Tensor) -> torch.Tensor:
        """计算特征重要性"""
        # 使用方差作为重要性指标
        teacher_var = torch.var(teacher_features, dim=0)
        student_var = torch.var(student_features, dim=0)
        
        # 综合Teacher和Student的方差
        combined_importance = (teacher_var + student_var) / 2
        
        # 归一化
        importance = combined_importance / (torch.sum(combined_importance) + 1e-8)
        
        return importance
    
    def _generate_pruning_mask(self, feature_importance: torch.Tensor) -> torch.Tensor:
        """生成剪枝掩码"""
        # 保留重要性最高的特征
        num_features_to_keep = int(len(feature_importance) * (1 - self.pruning_ratio))
        
        # 获取topk重要特征的索引
        _, top_indices = torch.topk(feature_importance, num_features_to_keep)
        
        # 创建掩码
        mask = torch.zeros_like(feature_importance, dtype=torch.bool)
        mask[top_indices] = True
        
        return mask
    
    def _apply_pruning(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """应用剪枝掩码"""
        return features[:, mask]
    
    def _evaluate_pruning_performance(self, original_teacher: torch.Tensor, 
                                    original_student: torch.Tensor,
                                    pruned_teacher: torch.Tensor,
                                    pruned_student: torch.Tensor) -> Dict[str, float]:
        """评估剪枝性能"""
        
        performance = {}
        
        # 计算信息保持率
        if original_teacher.numel() > 0 and pruned_teacher.numel() > 0:
            teacher_info_retention = float(torch.norm(pruned_teacher) / torch.norm(original_teacher))
            performance['teacher_info_retention'] = teacher_info_retention
        
        if original_student.numel() > 0 and pruned_student.numel() > 0:
            student_info_retention = float(torch.norm(pruned_student) / torch.norm(original_student))
            performance['student_info_retention'] = student_info_retention
        
        # 计算压缩比
        original_params = original_teacher.numel()
        pruned_params = pruned_teacher.numel()
        compression_ratio = float(pruned_params / original_params) if original_params > 0 else 0.0
        performance['compression_ratio'] = compression_ratio
        
        # 计算效率增益
        efficiency_gain = float(1 - compression_ratio) if compression_ratio < 1 else 0.0
        performance['efficiency_gain'] = efficiency_gain
        
        return performance
    
    def _evaluate_pakd_performance(self, aligned_data: List[Dict],
                                 teacher_features: torch.Tensor,
                                 student_features: torch.Tensor,
                                 pruning_results: Dict) -> Dict[str, Any]:
        """评估PAKD整体性能"""
        print("📊 评估PAKD性能...")
        
        evaluation = {
            'recommendation_quality': {},
            'model_efficiency': {},
            'knowledge_transfer': {}
        }
        
        # 推荐质量评估
        evaluation['recommendation_quality'] = self._evaluate_recommendation_quality(aligned_data)
        
        # 模型效率评估
        evaluation['model_efficiency'] = {
            'parameter_reduction': pruning_results.get('pruning_ratio_actual', 0.0),
            'information_retention': {
                'teacher': pruning_results.get('pruning_performance', {}).get('teacher_info_retention', 0.0),
                'student': pruning_results.get('pruning_performance', {}).get('student_info_retention', 0.0)
            },
            'compression_ratio': pruning_results.get('pruning_performance', {}).get('compression_ratio', 0.0)
        }
        
        # 知识传递评估
        if teacher_features.size(0) > 0 and student_features.size(0) > 0:
            evaluation['knowledge_transfer'] = self._evaluate_knowledge_transfer(teacher_features, student_features)
        
        return evaluation
    
    def _evaluate_recommendation_quality(self, aligned_data: List[Dict]) -> Dict[str, float]:
        """评估推荐质量"""
        if not aligned_data:
            return {}
        
        teacher_scores = []
        student_scores = []
        
        for pair in aligned_data:
            # Teacher推荐质量
            teacher_recs = pair['teacher'].get('recommendations', [])
            if teacher_recs:
                teacher_score = len(teacher_recs) / 10.0  # 推荐数量得分
                teacher_score += sum(1 for rec in teacher_recs if rec.get('reason')) / len(teacher_recs)  # 解释得分
                teacher_scores.append(min(teacher_score, 2.0))
            
            # Student推荐质量
            student_recs = pair['student'].get('recommendations', [])
            if student_recs:
                student_score = len(student_recs) / 10.0
                student_score += sum(1 for rec in student_recs if rec.get('reason')) / len(student_recs)
                student_scores.append(min(student_score, 2.0))
        
        quality_metrics = {}
        
        if teacher_scores:
            quality_metrics['teacher_avg_quality'] = float(np.mean(teacher_scores))
            quality_metrics['teacher_quality_std'] = float(np.std(teacher_scores))
        
        if student_scores:
            quality_metrics['student_avg_quality'] = float(np.mean(student_scores))
            quality_metrics['student_quality_std'] = float(np.std(student_scores))
        
        if teacher_scores and student_scores:
            # 计算质量保持率
            min_len = min(len(teacher_scores), len(student_scores))
            quality_retention = np.mean(student_scores[:min_len]) / (np.mean(teacher_scores[:min_len]) + 1e-8)
            quality_metrics['quality_retention_ratio'] = float(quality_retention)
        
        return quality_metrics
    
    def _evaluate_knowledge_transfer(self, teacher_features: torch.Tensor, 
                                   student_features: torch.Tensor) -> Dict[str, float]:
        """评估知识传递效果"""
        min_size = min(teacher_features.size(0), student_features.size(0))
        
        if min_size == 0:
            return {}
        
        teacher_sample = teacher_features[:min_size]
        student_sample = student_features[:min_size]
        
        # 计算特征相似性
        similarity = F.cosine_similarity(teacher_sample, student_sample, dim=1)
        avg_similarity = float(torch.mean(similarity))
        
        # 计算知识蒸馏效率
        teacher_norm = torch.norm(teacher_sample, dim=1)
        student_norm = torch.norm(student_sample, dim=1)
        
        norm_ratio = float(torch.mean(student_norm / (teacher_norm + 1e-8)))
        
        return {
            'feature_similarity': avg_similarity,
            'knowledge_compression_ratio': norm_ratio,
            'transfer_efficiency': avg_similarity * norm_ratio
        }
    
    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """计算张量统计信息"""
        if tensor.numel() == 0:
            return {'error': 'Empty tensor'}
        
        return {
            'mean': float(torch.mean(tensor)),
            'std': float(torch.std(tensor)),
            'min': float(torch.min(tensor)),
            'max': float(torch.max(tensor)),
            'norm': float(torch.norm(tensor))
        }
    
    def save_pakd_results(self, pakd_results: Dict[str, Any], output_path: str):
        """保存PAKD结果"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pakd_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 PAKD结果已保存至: {output_file}")
    
    def generate_pakd_summary(self, pakd_results: Dict[str, Any]) -> str:
        """生成PAKD摘要报告"""
        summary = "# LLM PAKD (Pruning-Aware Knowledge Distillation) Summary\n\n"
        
        # 配置信息
        config = pakd_results.get('config', {})
        summary += "## ⚙️ 配置参数\n"
        summary += f"- 蒸馏温度: {config.get('temperature', 'N/A')}\n"
        summary += f"- Alpha权重: {config.get('alpha', 'N/A')}\n"
        summary += f"- 剪枝比例: {config.get('pruning_ratio', 'N/A')}\n"
        summary += f"- 计算设备: {config.get('device', 'N/A')}\n\n"
        
        # 数据信息
        data_info = pakd_results.get('data_info', {})
        summary += "## 📊 数据信息\n"
        summary += f"- Teacher样本数: {data_info.get('teacher_samples', 0)}\n"
        summary += f"- Student样本数: {data_info.get('student_samples', 0)}\n"
        summary += f"- 对齐数据对: {data_info.get('aligned_pairs', 0)}\n\n"
        
        # 蒸馏结果
        dist_results = pakd_results.get('distillation_results', {})
        if 'error' not in dist_results:
            summary += "## 🎓 知识蒸馏结果\n"
            summary += f"- 蒸馏损失: {dist_results.get('distillation_loss', 0):.6f}\n"
            summary += f"- 对齐损失: {dist_results.get('alignment_loss', 0):.6f}\n"
            summary += f"- 总损失: {dist_results.get('total_loss', 0):.6f}\n"
            
            info_transfer = dist_results.get('information_transfer_efficiency', {})
            if info_transfer:
                summary += f"- 信息传递相关性: {info_transfer.get('correlation', 0):.4f}\n"
                summary += f"- 信息保持率: {info_transfer.get('information_retention', 0):.4f}\n\n"
        
        # 剪枝结果
        pruning_results = pakd_results.get('pruning_results', {})
        if 'error' not in pruning_results:
            summary += "## ✂️ 剪枝优化结果\n"
            summary += f"- 实际剪枝比例: {pruning_results.get('pruning_ratio_actual', 0):.2%}\n"
            summary += f"- 原始特征数: {pruning_results.get('original_feature_count', 0)}\n"
            summary += f"- 保留特征数: {pruning_results.get('remaining_feature_count', 0)}\n"
            
            pruning_perf = pruning_results.get('pruning_performance', {})
            if pruning_perf:
                summary += f"- 压缩比: {pruning_perf.get('compression_ratio', 0):.4f}\n"
                summary += f"- 效率增益: {pruning_perf.get('efficiency_gain', 0):.2%}\n\n"
        
        # 评估结果
        eval_results = pakd_results.get('evaluation_results', {})
        if eval_results:
            summary += "## 📈 性能评估\n"
            
            # 推荐质量
            rec_quality = eval_results.get('recommendation_quality', {})
            if rec_quality:
                summary += "### 推荐质量\n"
                summary += f"- Teacher平均质量: {rec_quality.get('teacher_avg_quality', 0):.4f}\n"
                summary += f"- Student平均质量: {rec_quality.get('student_avg_quality', 0):.4f}\n"
                summary += f"- 质量保持率: {rec_quality.get('quality_retention_ratio', 0):.4f}\n\n"
            
            # 知识传递
            knowledge_transfer = eval_results.get('knowledge_transfer', {})
            if knowledge_transfer:
                summary += "### 知识传递效果\n"
                summary += f"- 特征相似性: {knowledge_transfer.get('feature_similarity', 0):.4f}\n"
                summary += f"- 传递效率: {knowledge_transfer.get('transfer_efficiency', 0):.4f}\n\n"
        
        return summary


def main():
    """主函数 - 演示LLM PAKD"""
    print("🧠 LLM PAKD Distiller 演示")
    
    distiller = LLMPAKDDistiller()
    
    # 创建示例数据
    teacher_data = [
        {
            'user_id': 1,
            'model': 'llama3',
            'recommendations': [{'title': '星际穿越', 'reason': '科幻佳作'}],
            'llm_response': '{"recommendations": [{"title": "星际穿越"}]}',
            'timestamp': time.time()
        }
    ]
    
    student_data = [
        {
            'user_id': 1,
            'model': 'qwen3',
            'recommendations': [{'title': '星际穿越', 'reason': '推荐'}],
            'llm_response': '{"recommendations": [{"title": "星际穿越"}]}',
            'timestamp': time.time()
        }
    ]
    
    # 运行PAKD
    results = distiller.run_llm_pakd(teacher_data, student_data)
    
    # 生成摘要
    summary = distiller.generate_pakd_summary(results)
    print(summary)


if __name__ == "__main__":
    import time
    main()
