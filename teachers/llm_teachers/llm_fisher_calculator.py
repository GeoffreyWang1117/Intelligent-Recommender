#!/usr/bin/env python3
"""
LLM Fisher Information Calculator - LLM模型的Fisher信息计算
LLM-specific Fisher Information Analysis for Recommendation Systems

功能:
1. LLM推荐交互的Fisher信息计算
2. 提示词和响应的信息价值分析
3. 多模型(Llama3+Qwen3)的信息对比
4. 支持推荐质量的Fisher信息度量

作者: GitHub Copilot
日期: 2025-08-29
"""

import numpy as np
import torch
import torch.nn as nn
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class LLMFisherCalculator:
    """LLM专用Fisher Information计算器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 LLM Fisher Calculator 初始化 (设备: {self.device})")
        
        # Fisher信息存储
        self.fisher_matrices = {}
        self.interaction_embeddings = {}
        self.model_statistics = {}
        
    def compute_llm_fisher_information(self, llm_interactions: List[Dict[str, Any]], 
                                     embedding_dim: int = 512) -> Dict[str, Any]:
        """
        计算LLM交互的Fisher Information
        
        Args:
            llm_interactions: LLM交互数据列表
            embedding_dim: 嵌入维度
            
        Returns:
            Dict: Fisher信息分析结果
        """
        print(f"🔍 计算LLM Fisher Information ({len(llm_interactions)} 个交互)...")
        
        if not llm_interactions:
            print("❌ 没有LLM交互数据")
            return {}
        
        # 1. 将交互转换为向量表示
        embeddings = self._convert_interactions_to_embeddings(llm_interactions, embedding_dim)
        
        # 2. 计算Fisher信息矩阵
        fisher_matrices = self._compute_fisher_matrices(embeddings)
        
        # 3. 分析信息价值
        information_analysis = self._analyze_information_value(embeddings, llm_interactions)
        
        # 4. 模型对比分析
        model_comparison = self._compare_model_fisher_info(embeddings, llm_interactions)
        
        # 5. 推荐质量与Fisher信息关联
        quality_correlation = self._analyze_quality_fisher_correlation(embeddings, llm_interactions)
        
        results = {
            'fisher_matrices': fisher_matrices,
            'information_analysis': information_analysis,
            'model_comparison': model_comparison,
            'quality_correlation': quality_correlation,
            'interaction_count': len(llm_interactions),
            'embedding_dimension': embedding_dim,
            'computation_device': str(self.device)
        }
        
        print("✅ LLM Fisher Information计算完成")
        return results
    
    def _convert_interactions_to_embeddings(self, interactions: List[Dict], embedding_dim: int) -> Dict[str, torch.Tensor]:
        """将LLM交互转换为向量表示"""
        print("🔄 转换LLM交互为向量表示...")
        
        embeddings = {
            'prompt_embeddings': [],
            'response_embeddings': [],
            'recommendation_embeddings': [],
            'metadata': []
        }
        
        for interaction in interactions:
            try:
                # 提取交互特征
                prompt_features = self._extract_prompt_features(interaction.get('prompt', ''))
                response_features = self._extract_response_features(interaction.get('response', ''))
                rec_features = self._extract_recommendation_features(interaction.get('recommendations', []))
                
                # 转换为固定维度向量
                prompt_emb = self._features_to_embedding(prompt_features, embedding_dim)
                response_emb = self._features_to_embedding(response_features, embedding_dim)
                rec_emb = self._features_to_embedding(rec_features, embedding_dim)
                
                embeddings['prompt_embeddings'].append(prompt_emb)
                embeddings['response_embeddings'].append(response_emb)
                embeddings['recommendation_embeddings'].append(rec_emb)
                
                # 保存元数据
                metadata = {
                    'user_id': interaction.get('user_id'),
                    'timestamp': interaction.get('timestamp'),
                    'prompt_length': interaction.get('prompt_length', 0),
                    'response_length': interaction.get('response_length', 0),
                    'num_recommendations': interaction.get('num_recommendations', 0)
                }
                embeddings['metadata'].append(metadata)
                
            except Exception as e:
                print(f"⚠️  交互向量化失败: {e}")
                continue
        
        # 转换为tensor
        for key in ['prompt_embeddings', 'response_embeddings', 'recommendation_embeddings']:
            if embeddings[key]:
                embeddings[key] = torch.stack(embeddings[key]).to(self.device)
            else:
                embeddings[key] = torch.empty(0, embedding_dim).to(self.device)
        
        print(f"✅ 转换完成: {len(embeddings['metadata'])} 个交互向量")
        return embeddings
    
    def _extract_prompt_features(self, prompt: str) -> Dict[str, float]:
        """提取提示词特征"""
        features = {
            'length': len(prompt),
            'word_count': len(prompt.split()),
            'question_marks': prompt.count('?'),
            'exclamation_marks': prompt.count('!'),
            'periods': prompt.count('.'),
            'commas': prompt.count(','),
            'chinese_chars': sum(1 for char in prompt if '\u4e00' <= char <= '\u9fff'),
            'english_words': len([word for word in prompt.split() if word.isalpha() and word.isascii()]),
            'numbers': sum(1 for char in prompt if char.isdigit()),
            'recommendation_keywords': sum(1 for keyword in ['推荐', 'recommend', '电影', 'movie'] if keyword in prompt.lower()),
            'user_context_keywords': sum(1 for keyword in ['用户', 'user', '喜欢', 'like'] if keyword in prompt.lower())
        }
        
        # 归一化
        max_length = 2000  # 假设最大提示词长度
        features['length_normalized'] = min(features['length'] / max_length, 1.0)
        features['word_density'] = features['word_count'] / max(features['length'], 1)
        features['punctuation_density'] = (features['question_marks'] + features['exclamation_marks'] + features['periods']) / max(features['length'], 1)
        
        return features
    
    def _extract_response_features(self, response: str) -> Dict[str, float]:
        """提取响应特征"""
        features = {
            'length': len(response),
            'word_count': len(response.split()),
            'json_structure': 1.0 if '{' in response and '}' in response else 0.0,
            'recommendation_structure': 1.0 if 'recommendations' in response.lower() else 0.0,
            'chinese_chars': sum(1 for char in response if '\u4e00' <= char <= '\u9fff'),
            'english_words': len([word for word in response.split() if word.isalpha() and word.isascii()]),
            'numbers': sum(1 for char in response if char.isdigit()),
            'movie_mentions': response.lower().count('电影') + response.lower().count('movie'),
            'reason_keywords': sum(1 for keyword in ['因为', 'because', '适合', 'suitable'] if keyword in response.lower())
        }
        
        # 归一化
        max_length = 1500  # 假设最大响应长度
        features['length_normalized'] = min(features['length'] / max_length, 1.0)
        features['word_density'] = features['word_count'] / max(features['length'], 1)
        features['content_richness'] = (features['movie_mentions'] + features['reason_keywords']) / max(features['word_count'], 1)
        
        return features
    
    def _extract_recommendation_features(self, recommendations: List[Dict]) -> Dict[str, float]:
        """提取推荐特征"""
        if not recommendations:
            return {f'rec_feature_{i}': 0.0 for i in range(10)}
        
        features = {
            'count': len(recommendations),
            'has_reasons': sum(1 for rec in recommendations if rec.get('reason')) / len(recommendations),
            'avg_reason_length': np.mean([len(rec.get('reason', '')) for rec in recommendations]),
            'title_diversity': len(set(rec.get('title', '') for rec in recommendations)) / len(recommendations),
            'ranking_completeness': sum(1 for rec in recommendations if rec.get('rank')) / len(recommendations)
        }
        
        # 归一化
        features['count_normalized'] = min(features['count'] / 10, 1.0)
        features['reason_quality'] = min(features['avg_reason_length'] / 100, 1.0)
        
        # 补充特征以达到固定维度
        for i in range(len(features), 10):
            features[f'rec_feature_{i}'] = 0.0
        
        return features
    
    def _features_to_embedding(self, features: Dict[str, float], target_dim: int) -> torch.Tensor:
        """将特征转换为固定维度的嵌入向量"""
        # 提取特征值
        values = list(features.values())
        
        # 填充或截断到目标维度
        if len(values) < target_dim:
            values.extend([0.0] * (target_dim - len(values)))
        elif len(values) > target_dim:
            values = values[:target_dim]
        
        # 归一化
        values = np.array(values, dtype=np.float32)
        values = np.nan_to_num(values)  # 处理NaN值
        
        # 简单的归一化
        if np.std(values) > 0:
            values = (values - np.mean(values)) / np.std(values)
        
        return torch.tensor(values, dtype=torch.float32)
    
    def _compute_fisher_matrices(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """计算Fisher信息矩阵"""
        print("🧮 计算Fisher信息矩阵...")
        
        fisher_matrices = {}
        
        for emb_type in ['prompt_embeddings', 'response_embeddings', 'recommendation_embeddings']:
            emb_tensor = embeddings[emb_type]
            
            if emb_tensor.size(0) == 0:
                continue
            
            try:
                # 计算协方差矩阵
                mean_emb = torch.mean(emb_tensor, dim=0)
                centered = emb_tensor - mean_emb
                
                # Fisher信息矩阵 = 协方差矩阵的逆
                cov_matrix = torch.mm(centered.T, centered) / (emb_tensor.size(0) - 1)
                
                # 添加正则化项避免奇异矩阵
                reg_term = torch.eye(cov_matrix.size(0), device=self.device) * 1e-6
                cov_matrix += reg_term
                
                # 计算Fisher信息矩阵
                try:
                    fisher_matrix = torch.inverse(cov_matrix)
                except:
                    # 如果逆矩阵计算失败，使用伪逆
                    fisher_matrix = torch.pinverse(cov_matrix)
                
                # 计算关键统计量
                eigenvalues = torch.linalg.eigvals(fisher_matrix).real
                fisher_trace = torch.trace(fisher_matrix)
                fisher_determinant = torch.det(fisher_matrix)
                condition_number = torch.max(eigenvalues) / (torch.min(eigenvalues) + 1e-8)
                
                fisher_matrices[emb_type] = {
                    'matrix_shape': list(fisher_matrix.shape),
                    'trace': float(fisher_trace),
                    'determinant': float(fisher_determinant),
                    'condition_number': float(condition_number),
                    'eigenvalue_stats': {
                        'mean': float(torch.mean(eigenvalues)),
                        'std': float(torch.std(eigenvalues)),
                        'max': float(torch.max(eigenvalues)),
                        'min': float(torch.min(eigenvalues))
                    },
                    'information_content': float(torch.sum(torch.log(eigenvalues + 1e-8)))
                }
                
            except Exception as e:
                print(f"⚠️  计算Fisher矩阵失败 ({emb_type}): {e}")
                fisher_matrices[emb_type] = {'error': str(e)}
        
        return fisher_matrices
    
    def _analyze_information_value(self, embeddings: Dict[str, torch.Tensor], 
                                 interactions: List[Dict]) -> Dict[str, Any]:
        """分析信息价值"""
        print("📊 分析信息价值...")
        
        analysis = {
            'interaction_statistics': {},
            'information_efficiency': {},
            'content_complexity': {}
        }
        
        # 基本统计
        prompt_lengths = [meta['prompt_length'] for meta in embeddings['metadata']]
        response_lengths = [meta['response_length'] for meta in embeddings['metadata']]
        rec_counts = [meta['num_recommendations'] for meta in embeddings['metadata']]
        
        analysis['interaction_statistics'] = {
            'prompt_length_stats': {
                'mean': float(np.mean(prompt_lengths)),
                'std': float(np.std(prompt_lengths)),
                'min': int(np.min(prompt_lengths)),
                'max': int(np.max(prompt_lengths))
            },
            'response_length_stats': {
                'mean': float(np.mean(response_lengths)),
                'std': float(np.std(response_lengths)),
                'min': int(np.min(response_lengths)),
                'max': int(np.max(response_lengths))
            },
            'recommendation_count_stats': {
                'mean': float(np.mean(rec_counts)),
                'std': float(np.std(rec_counts)),
                'min': int(np.min(rec_counts)),
                'max': int(np.max(rec_counts))
            }
        }
        
        # 信息效率分析
        if embeddings['prompt_embeddings'].size(0) > 0 and embeddings['response_embeddings'].size(0) > 0:
            # 计算提示词-响应相关性
            prompt_norm = torch.norm(embeddings['prompt_embeddings'], dim=1)
            response_norm = torch.norm(embeddings['response_embeddings'], dim=1)
            
            cosine_similarities = []
            for i in range(min(len(prompt_norm), len(response_norm))):
                if prompt_norm[i] > 0 and response_norm[i] > 0:
                    cos_sim = torch.dot(
                        embeddings['prompt_embeddings'][i] / prompt_norm[i],
                        embeddings['response_embeddings'][i] / response_norm[i]
                    )
                    cosine_similarities.append(float(cos_sim))
            
            if cosine_similarities:
                analysis['information_efficiency'] = {
                    'prompt_response_similarity': {
                        'mean': float(np.mean(cosine_similarities)),
                        'std': float(np.std(cosine_similarities)),
                        'correlation_strength': 'strong' if np.mean(cosine_similarities) > 0.7 else 'moderate' if np.mean(cosine_similarities) > 0.3 else 'weak'
                    }
                }
        
        return analysis
    
    def _compare_model_fisher_info(self, embeddings: Dict[str, torch.Tensor], 
                                 interactions: List[Dict]) -> Dict[str, Any]:
        """对比不同模型的Fisher信息"""
        print("🔄 对比模型Fisher信息...")
        
        model_data = defaultdict(list)
        
        # 按模型分组数据
        for i, interaction in enumerate(interactions):
            model = 'unknown'
            if 'llm_response' in interaction:
                if 'llama' in str(interaction.get('model', '')).lower():
                    model = 'llama3'
                elif 'qwen' in str(interaction.get('model', '')).lower():
                    model = 'qwen3'
            
            if i < embeddings['response_embeddings'].size(0):
                model_data[model].append(i)
        
        comparison = {}
        
        for model, indices in model_data.items():
            if len(indices) < 2:
                continue
            
            try:
                # 提取该模型的嵌入
                model_embeddings = embeddings['response_embeddings'][indices]
                
                # 计算模型特定的Fisher信息
                mean_emb = torch.mean(model_embeddings, dim=0)
                centered = model_embeddings - mean_emb
                cov_matrix = torch.mm(centered.T, centered) / (len(indices) - 1)
                
                # 添加正则化
                reg_term = torch.eye(cov_matrix.size(0), device=self.device) * 1e-6
                cov_matrix += reg_term
                
                # 计算Fisher矩阵
                fisher_matrix = torch.pinverse(cov_matrix)
                eigenvalues = torch.linalg.eigvals(fisher_matrix).real
                
                comparison[model] = {
                    'sample_count': len(indices),
                    'fisher_trace': float(torch.trace(fisher_matrix)),
                    'information_content': float(torch.sum(torch.log(eigenvalues + 1e-8))),
                    'eigenvalue_spread': float(torch.std(eigenvalues)),
                    'dominant_eigenvalue': float(torch.max(eigenvalues))
                }
                
            except Exception as e:
                print(f"⚠️  模型 {model} Fisher分析失败: {e}")
                comparison[model] = {'error': str(e)}
        
        return comparison
    
    def _analyze_quality_fisher_correlation(self, embeddings: Dict[str, torch.Tensor], 
                                          interactions: List[Dict]) -> Dict[str, Any]:
        """分析推荐质量与Fisher信息的关联"""
        print("🎯 分析质量-Fisher关联...")
        
        quality_scores = []
        fisher_scores = []
        
        for i, interaction in enumerate(interactions):
            if i >= embeddings['recommendation_embeddings'].size(0):
                continue
            
            # 计算推荐质量分数（简化版）
            recommendations = interaction.get('recommendations', [])
            if recommendations:
                quality = len(recommendations) / 10.0  # 推荐数量归一化
                quality += sum(1 for rec in recommendations if rec.get('reason', '')) / len(recommendations)  # 解释比例
                quality_scores.append(min(quality, 2.0))  # 限制在0-2范围
                
                # 计算对应的Fisher分数（基于嵌入的信息量）
                rec_emb = embeddings['recommendation_embeddings'][i]
                fisher_score = float(torch.norm(rec_emb))  # 使用向量范数作为信息量代理
                fisher_scores.append(fisher_score)
        
        correlation_analysis = {}
        
        if len(quality_scores) >= 5:
            correlation = np.corrcoef(quality_scores, fisher_scores)[0, 1]
            
            correlation_analysis = {
                'sample_count': len(quality_scores),
                'quality_fisher_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'quality_stats': {
                    'mean': float(np.mean(quality_scores)),
                    'std': float(np.std(quality_scores)),
                    'range': [float(np.min(quality_scores)), float(np.max(quality_scores))]
                },
                'fisher_stats': {
                    'mean': float(np.mean(fisher_scores)),
                    'std': float(np.std(fisher_scores)),
                    'range': [float(np.min(fisher_scores)), float(np.max(fisher_scores))]
                },
                'correlation_interpretation': self._interpret_correlation(correlation if not np.isnan(correlation) else 0.0)
            }
        
        return correlation_analysis
    
    def _interpret_correlation(self, correlation: float) -> str:
        """解释相关性强度"""
        abs_corr = abs(correlation)
        if abs_corr > 0.8:
            return 'very_strong'
        elif abs_corr > 0.6:
            return 'strong' 
        elif abs_corr > 0.4:
            return 'moderate'
        elif abs_corr > 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def save_fisher_analysis(self, fisher_results: Dict[str, Any], output_path: str):
        """保存Fisher分析结果"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fisher_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Fisher分析结果已保存至: {output_file}")
    
    def generate_fisher_summary(self, fisher_results: Dict[str, Any]) -> str:
        """生成Fisher分析摘要"""
        summary = "# LLM Fisher Information Analysis Summary\n\n"
        
        # 基本信息
        summary += f"## 📊 基本信息\n"
        summary += f"- 交互样本数: {fisher_results.get('interaction_count', 0)}\n"
        summary += f"- 嵌入维度: {fisher_results.get('embedding_dimension', 0)}\n"
        summary += f"- 计算设备: {fisher_results.get('computation_device', 'unknown')}\n\n"
        
        # Fisher矩阵分析
        if 'fisher_matrices' in fisher_results:
            summary += "## 🧮 Fisher矩阵分析\n"
            for emb_type, matrix_info in fisher_results['fisher_matrices'].items():
                if 'error' not in matrix_info:
                    summary += f"### {emb_type}\n"
                    summary += f"- 矩阵维度: {matrix_info.get('matrix_shape', [])}\n"
                    summary += f"- 迹 (Trace): {matrix_info.get('trace', 0):.4f}\n"
                    summary += f"- 行列式: {matrix_info.get('determinant', 0):.6f}\n"
                    summary += f"- 条件数: {matrix_info.get('condition_number', 0):.4f}\n"
                    summary += f"- 信息内容: {matrix_info.get('information_content', 0):.4f}\n\n"
        
        # 模型对比
        if 'model_comparison' in fisher_results:
            summary += "## 🔄 模型对比\n"
            for model, stats in fisher_results['model_comparison'].items():
                if 'error' not in stats:
                    summary += f"### {model}\n"
                    summary += f"- 样本数: {stats.get('sample_count', 0)}\n"
                    summary += f"- Fisher迹: {stats.get('fisher_trace', 0):.4f}\n"
                    summary += f"- 信息内容: {stats.get('information_content', 0):.4f}\n\n"
        
        # 质量关联分析
        if 'quality_correlation' in fisher_results and fisher_results['quality_correlation']:
            qc = fisher_results['quality_correlation']
            summary += "## 🎯 质量-Fisher关联\n"
            summary += f"- 相关系数: {qc.get('quality_fisher_correlation', 0):.4f}\n"
            summary += f"- 相关性强度: {qc.get('correlation_interpretation', 'unknown')}\n"
            summary += f"- 分析样本数: {qc.get('sample_count', 0)}\n\n"
        
        return summary


def main():
    """主函数 - 演示LLM Fisher分析"""
    print("🔍 LLM Fisher Information Calculator 演示")
    
    calculator = LLMFisherCalculator()
    
    # 创建示例数据
    sample_interactions = [
        {
            'user_id': 1,
            'prompt': '推荐一些科幻电影',
            'response': '{"recommendations": [{"title": "星际穿越", "reason": "优秀的科幻片"}]}',
            'recommendations': [{'title': '星际穿越', 'reason': '优秀的科幻片'}],
            'model': 'llama3',
            'prompt_length': 10,
            'response_length': 50,
            'num_recommendations': 1
        }
    ]
    
    # 计算Fisher信息
    results = calculator.compute_llm_fisher_information(sample_interactions)
    
    # 生成摘要
    summary = calculator.generate_fisher_summary(results)
    print(summary)


if __name__ == "__main__":
    main()
