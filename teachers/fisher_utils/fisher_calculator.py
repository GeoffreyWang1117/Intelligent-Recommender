#!/usr/bin/env python3
"""
Fisher Information Per-Layer计算模块
用于分析神经网络各层参数重要性，指导pruning-aware蒸馏

Fisher Information 衡量模型参数对输出的敏感度:
F_ij = E[∂log p(y|x,θ)/∂θ_i * ∂log p(y|x,θ)/∂θ_j]

作者: GitHub Copilot  
日期: 2025-08-27
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from collections import defaultdict
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FisherInformationCalculator:
    """Fisher Information per-layer计算器"""
    
    def __init__(self, model: nn.Module, device: str = 'auto'):
        """
        初始化Fisher计算器
        
        Args:
            model: 要分析的PyTorch模型
            device: 计算设备
        """
        self.model = model
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        # Fisher信息存储
        self.fisher_info = {}
        self.layer_names = []
        self.gradient_hooks = {}
        self.accumulated_gradients = {}
        
        # 计算配置
        self.batch_count = 0
        self.fisher_samples = 0
        
        logger.info(f"初始化Fisher Information计算器，设备: {self.device}")
        
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def register_gradient_hooks(self):
        """注册梯度钩子函数"""
        logger.info("🔗 注册梯度钩子...")
        
        def create_hook(name: str):
            def hook_fn(grad):
                if name not in self.accumulated_gradients:
                    self.accumulated_gradients[name] = []
                # 存储梯度的平方（Fisher Information的核心）
                self.accumulated_gradients[name].append(grad.detach().clone() ** 2)
                return grad
            return hook_fn
        
        # 为所有可训练参数注册钩子
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.layer_names.append(name)
                hook = param.register_hook(create_hook(name))
                self.gradient_hooks[name] = hook
                logger.debug(f"   注册钩子: {name}, 形状: {param.shape}")
        
        logger.info(f"✅ 成功注册 {len(self.layer_names)} 个参数的梯度钩子")
    
    def compute_fisher_information(self, 
                                 dataloader: torch.utils.data.DataLoader,
                                 criterion: nn.Module,
                                 num_batches: Optional[int] = None,
                                 accumulate: bool = True) -> Dict[str, torch.Tensor]:
        """
        计算Fisher Information
        
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            num_batches: 限制批次数量（None表示使用全部数据）
            accumulate: 是否累积之前的Fisher信息
            
        Returns:
            每层的Fisher Information字典
        """
        logger.info("🧮 开始计算Fisher Information...")
        
        if not accumulate:
            self.accumulated_gradients.clear()
            self.batch_count = 0
            self.fisher_samples = 0
        
        # 确保钩子已注册
        if not self.gradient_hooks:
            self.register_gradient_hooks()
        
        self.model.train()  # 训练模式以计算梯度
        
        batch_processed = 0
        total_samples = 0
        
        with torch.enable_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                if num_batches and batch_idx >= num_batches:
                    break
                
                try:
                    # 数据预处理
                    if isinstance(batch_data, (tuple, list)):
                        if len(batch_data) == 2:
                            inputs, targets = batch_data
                        else:
                            inputs = batch_data[0]
                            targets = batch_data[1] if len(batch_data) > 1 else None
                    else:
                        inputs = batch_data
                        targets = None
                    
                    # 移动到设备
                    inputs = self._move_to_device(inputs)
                    if targets is not None:
                        targets = self._move_to_device(targets)
                    
                    # 前向传播
                    self.model.zero_grad()
                    outputs = self.model(inputs)
                    
                    # 计算损失
                    if targets is not None:
                        loss = criterion(outputs, targets)
                    else:
                        # 如果没有目标，使用输出的均值作为损失（无监督场景）
                        loss = outputs.mean()
                    
                    # 反向传播计算梯度
                    loss.backward()
                    
                    batch_processed += 1
                    # 安全地计算样本数量
                    if torch.is_tensor(inputs):
                        total_samples += inputs.shape[0]
                    elif isinstance(inputs, (tuple, list)) and len(inputs) > 0:
                        first_tensor = inputs[0]
                        if torch.is_tensor(first_tensor):
                            total_samples += first_tensor.shape[0]
                        else:
                            total_samples += 1
                    else:
                        total_samples += 1
                    
                    if batch_idx % 50 == 0:
                        logger.debug(f"   处理批次 {batch_idx}, 损失: {loss.item():.4f}")
                
                except Exception as e:
                    logger.warning(f"   批次 {batch_idx} 处理失败: {e}")
                    continue
        
        self.batch_count += batch_processed
        self.fisher_samples += total_samples
        
        logger.info(f"✅ 完成 {batch_processed} 个批次，{total_samples} 个样本的Fisher计算")
        
        # 计算平均Fisher Information
        return self._compute_average_fisher()
    
    def _move_to_device(self, data):
        """将数据移动到指定设备"""
        if torch.is_tensor(data):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(item) for item in data)
        else:
            return data
    
    def _compute_average_fisher(self) -> Dict[str, torch.Tensor]:
        """计算平均Fisher Information"""
        logger.info("📊 计算平均Fisher Information...")
        
        fisher_info = {}
        
        for name in self.layer_names:
            if name in self.accumulated_gradients and self.accumulated_gradients[name]:
                # 计算梯度平方的平均值
                grad_squares = torch.stack(self.accumulated_gradients[name])
                fisher_info[name] = grad_squares.mean(dim=0)
                
                logger.debug(f"   {name}: Fisher形状 {fisher_info[name].shape}, "
                           f"平均值 {fisher_info[name].mean().item():.6f}")
            else:
                logger.warning(f"   {name}: 没有收集到梯度信息")
        
        self.fisher_info = fisher_info
        logger.info(f"✅ 计算完成，获得 {len(fisher_info)} 层的Fisher信息")
        
        return fisher_info
    
    def analyze_importance(self, fisher_info: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        分析Fisher Information重要性统计
        
        Args:
            fisher_info: Fisher Information字典
            
        Returns:
            重要性统计信息
        """
        # 全局统计
        all_values = []
        layer_stats = {}
        
        for name, tensor in fisher_info.items():
            values = tensor.flatten().cpu().numpy()
            all_values.extend(values)
            
            layer_stats[name] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(np.median(values))
            }
        
        all_values = np.array(all_values)
        global_stats = {
            'mean': float(all_values.mean()),
            'std': float(all_values.std()),
            'min': float(all_values.min()),
            'max': float(all_values.max()),
            'median': float(np.median(all_values))
        }
        
        return {
            'global_stats': global_stats,
            'layer_stats': layer_stats
        }
    
    def get_sparsity_stats(self, masks: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        获取稀疏性统计信息
        
        Args:
            masks: 剪枝掩码字典
            
        Returns:
            稀疏性统计信息
        """
        total_params = 0
        zero_params = 0
        layer_sparsity = {}
        
        for name, mask in masks.items():
            layer_total = mask.numel()
            layer_zeros = (mask == 0).sum().item()
            layer_sparsity[name] = layer_zeros / layer_total
            
            total_params += layer_total
            zero_params += layer_zeros
        
        overall_sparsity = zero_params / total_params
        non_zero_ratio = 1.0 - overall_sparsity
        
        sparsity_values = list(layer_sparsity.values())
        sparsity_range = (min(sparsity_values), max(sparsity_values))
        
        return {
            'overall_sparsity': overall_sparsity,
            'non_zero_ratio': non_zero_ratio,
            'layer_sparsity': layer_sparsity,
            'layer_sparsity_range': sparsity_range,
            'total_params': total_params,
            'zero_params': zero_params
        }
    
    def analyze_layer_importance(self, 
                                normalize: bool = True, 
                                top_k: int = 10) -> Dict[str, Any]:
        """
        分析各层重要性
        
        Args:
            normalize: 是否归一化重要性分数
            top_k: 返回前K个重要层
            
        Returns:
            层重要性分析结果
        """
        logger.info("📈 分析层重要性...")
        
        if not self.fisher_info:
            raise ValueError("请先计算Fisher Information")
        
        # 计算每层的重要性分数
        layer_importance = {}
        
        for name, fisher_tensor in self.fisher_info.items():
            # 使用Fisher信息的总和作为重要性指标
            importance = fisher_tensor.sum().item()
            layer_importance[name] = importance
        
        # 归一化
        if normalize and layer_importance:
            total_importance = sum(layer_importance.values())
            layer_importance = {k: v/total_importance for k, v in layer_importance.items()}
        
        # 排序
        sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 分析结果
        analysis = {
            'layer_importance': dict(sorted_layers),
            'top_layers': sorted_layers[:top_k],
            'bottom_layers': sorted_layers[-top_k:],
            'importance_distribution': self._analyze_importance_distribution(layer_importance),
            'pruning_candidates': self._identify_pruning_candidates(layer_importance)
        }
        
        logger.info("📋 重要性分析完成:")
        logger.info(f"   最重要层: {sorted_layers[0][0]} (重要性: {sorted_layers[0][1]:.6f})")
        logger.info(f"   最不重要层: {sorted_layers[-1][0]} (重要性: {sorted_layers[-1][1]:.6f})")
        
        return analysis
    
    def _analyze_importance_distribution(self, layer_importance: Dict[str, float]) -> Dict[str, float]:
        """分析重要性分布"""
        values = list(layer_importance.values())
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75))
        }
    
    def _identify_pruning_candidates(self, 
                                   layer_importance: Dict[str, float],
                                   threshold_percentile: float = 20) -> List[str]:
        """识别剪枝候选层"""
        values = list(layer_importance.values())
        threshold = np.percentile(values, threshold_percentile)
        
        candidates = [name for name, importance in layer_importance.items() 
                     if importance <= threshold]
        
        return candidates
    
    def compute_parameter_sensitivity(self, 
                                    sensitivity_type: str = 'absolute') -> Dict[str, torch.Tensor]:
        """
        计算参数敏感度
        
        Args:
            sensitivity_type: 'absolute', 'relative', 'normalized'
            
        Returns:
            参数敏感度张量
        """
        logger.info(f"🎯 计算参数敏感度 ({sensitivity_type})...")
        
        if not self.fisher_info:
            raise ValueError("请先计算Fisher Information")
        
        sensitivity = {}
        
        for name, fisher_tensor in self.fisher_info.items():
            param = dict(self.model.named_parameters())[name]
            
            if sensitivity_type == 'absolute':
                # 绝对敏感度：直接使用Fisher信息
                sensitivity[name] = fisher_tensor
                
            elif sensitivity_type == 'relative':
                # 相对敏感度：Fisher信息除以参数值的平方
                param_squared = param.detach() ** 2
                sensitivity[name] = fisher_tensor / (param_squared + 1e-8)
                
            elif sensitivity_type == 'normalized':
                # 归一化敏感度：按层内最大值归一化
                max_fisher = fisher_tensor.max()
                sensitivity[name] = fisher_tensor / (max_fisher + 1e-8)
                
            else:
                raise ValueError(f"未知的敏感度类型: {sensitivity_type}")
        
        logger.info(f"✅ 完成 {len(sensitivity)} 层的敏感度计算")
        return sensitivity
    
    def generate_pruning_mask(self, 
                            fisher_info: Dict[str, torch.Tensor],
                            pruning_ratio: float = 0.2,
                            strategy: str = 'global') -> Dict[str, torch.Tensor]:
        """
        生成剪枝掩码
        
        Args:
            pruning_ratio: 剪枝比例 (0-1)
            strategy: 'global' (全局阈值) 或 'layer_wise' (逐层阈值)
            
        Returns:
            剪枝掩码字典
        """
        logger.info(f"✂️ 生成剪枝掩码 (比例: {pruning_ratio:.1%}, 策略: {strategy})...")
        
        sensitivity = fisher_info  # 直接使用传入的Fisher信息
        masks = {}
        
        if strategy == 'global':
            # 全局阈值：收集所有参数的Fisher信息
            all_fisher_values = []
            for fisher_tensor in sensitivity.values():
                all_fisher_values.extend(fisher_tensor.flatten().tolist())
            
            threshold = np.percentile(all_fisher_values, pruning_ratio * 100)
            
            for name, fisher_tensor in sensitivity.items():
                # 确保是torch tensor
                if not torch.is_tensor(fisher_tensor):
                    fisher_tensor = torch.tensor(fisher_tensor, device=self.device)
                mask = fisher_tensor > threshold
                if torch.is_tensor(mask):
                    masks[name] = mask.to(torch.float32)
                else:
                    masks[name] = torch.tensor(mask, dtype=torch.float32, device=self.device)
                
        elif strategy == 'layer_wise':
            # 逐层阈值：每层单独计算阈值
            for name, fisher_tensor in sensitivity.items():
                # 确保是torch tensor
                if not torch.is_tensor(fisher_tensor):
                    fisher_tensor = torch.tensor(fisher_tensor, device=self.device)
                layer_threshold = torch.quantile(fisher_tensor.flatten(), pruning_ratio)
                mask = fisher_tensor > layer_threshold
                if torch.is_tensor(mask):
                    masks[name] = mask.to(torch.float32)
                else:
                    masks[name] = torch.tensor(mask, dtype=torch.float32, device=self.device)
        
        else:
            raise ValueError(f"未知的剪枝策略: {strategy}")
        
        # 统计剪枝信息
        total_params = sum(mask.numel() for mask in masks.values())
        kept_params = sum(mask.sum().item() for mask in masks.values())
        actual_pruning_ratio = 1 - (kept_params / total_params)
        
        logger.info(f"✅ 剪枝掩码生成完成:")
        logger.info(f"   总参数: {total_params}")
        logger.info(f"   保留参数: {int(kept_params)}")
        logger.info(f"   实际剪枝比例: {actual_pruning_ratio:.1%}")
        
        return masks
    
    def save_fisher_info(self, save_path: str):
        """保存Fisher Information"""
        logger.info(f"💾 保存Fisher Information到: {save_path}")
        
        # 准备保存数据
        save_data = {
            'fisher_info': {name: tensor.cpu().numpy() for name, tensor in self.fisher_info.items()},
            'layer_names': self.layer_names,
            'batch_count': self.batch_count,
            'fisher_samples': self.fisher_samples,
            'model_info': {
                'model_class': self.model.__class__.__name__,
                'device': str(self.device)
            }
        }
        
        # 保存为numpy格式（更高效）
        np.savez_compressed(save_path, **save_data)
        logger.info("✅ Fisher Information保存完成")
    
    def load_fisher_info(self, load_path: str):
        """加载Fisher Information"""
        logger.info(f"📂 从 {load_path} 加载Fisher Information...")
        
        data = np.load(load_path, allow_pickle=True)
        
        # 恢复Fisher信息
        self.fisher_info = {}
        fisher_data = data['fisher_info'].item()
        for name, array in fisher_data.items():
            self.fisher_info[name] = torch.from_numpy(array).to(self.device)
        
        self.layer_names = data['layer_names'].tolist()
        self.batch_count = int(data['batch_count'])
        self.fisher_samples = int(data['fisher_samples'])
        
        logger.info(f"✅ 成功加载 {len(self.fisher_info)} 层的Fisher信息")
        logger.info(f"   基于 {self.batch_count} 批次，{self.fisher_samples} 样本")
    
    def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理Fisher计算器资源...")
        
        # 移除梯度钩子
        for hook in self.gradient_hooks.values():
            hook.remove()
        
        # 清理累积的梯度
        self.accumulated_gradients.clear()
        self.gradient_hooks.clear()
        
        logger.info("✅ 资源清理完成")


def create_simple_model_for_testing():
    """创建简单模型用于测试"""
    class SimpleRecommenderNet(nn.Module):
        def __init__(self, num_users=500, num_items=200, embedding_dim=32):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            
            self.fc_layers = nn.Sequential(
                nn.Linear(embedding_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            )
            
        def forward(self, user_ids, item_ids):
            user_emb = self.user_embedding(user_ids)
            item_emb = self.item_embedding(item_ids)
            
            combined = torch.cat([user_emb, item_emb], dim=-1)
            output = self.fc_layers(combined)
            
            return output.squeeze()
    
    return SimpleRecommenderNet()


def demo_fisher_information():
    """演示Fisher Information计算"""
    print("🧮 Fisher Information per-layer 计算演示")
    print("=" * 60)
    
    try:
        # 1. 创建简单模型
        print("🔧 创建测试模型...")
        model = create_simple_model_for_testing()
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 2. 创建模拟数据
        print("\\n📊 生成模拟数据...")
        batch_size = 64
        num_batches = 20
        
        # 模拟用户-物品交互数据
        dataset = []
        for _ in range(num_batches):
            user_ids = torch.randint(0, 500, (batch_size,))
            item_ids = torch.randint(0, 200, (batch_size,))
            # 模拟评分（1-5分）
            ratings = torch.randint(1, 6, (batch_size,)).float()
            dataset.append(((user_ids, item_ids), ratings))
        
        dataloader = dataset  # 简化的数据加载器
        
        # 3. 创建Fisher计算器
        print("\\n🎯 初始化Fisher Information计算器...")
        fisher_calc = FisherInformationCalculator(model)
        
        # 4. 计算Fisher Information
        print("\\n🧮 计算Fisher Information...")
        criterion = nn.MSELoss()
        
        fisher_calc.register_gradient_hooks()
        
        # 手动模拟数据加载过程
        model.train()
        for batch_idx, (inputs, targets) in enumerate(dataset):
            user_ids, item_ids = inputs
            
            model.zero_grad()
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if batch_idx % 5 == 0:
                print(f"   批次 {batch_idx}, 损失: {loss.item():.4f}")
        
        # 手动触发Fisher计算
        fisher_info = fisher_calc._compute_average_fisher()
        
        # 5. 分析层重要性
        print("\\n📈 分析层重要性...")
        importance_analysis = fisher_calc.analyze_layer_importance()
        
        print("\\n🏆 Top-5 重要层:")
        for i, (layer_name, importance) in enumerate(importance_analysis['top_layers'][:5]):
            print(f"   {i+1}. {layer_name}: {importance:.6f}")
        
        print("\\n⚠️ Bottom-5 重要层:")
        for i, (layer_name, importance) in enumerate(importance_analysis['bottom_layers'][:5]):
            print(f"   {i+1}. {layer_name}: {importance:.6f}")
        
        # 6. 生成剪枝掩码
        print("\\n✂️ 生成剪枝掩码...")
        pruning_masks = fisher_calc.generate_pruning_mask(fisher_info, pruning_ratio=0.3)
        
        print(f"   生成了 {len(pruning_masks)} 层的剪枝掩码")
        
        # 7. 参数敏感度分析
        print("\\n🎯 参数敏感度分析...")
        sensitivity = fisher_calc.compute_parameter_sensitivity('relative')
        
        for name in list(sensitivity.keys())[:3]:  # 显示前3层
            sens_tensor = sensitivity[name]
            print(f"   {name}: 敏感度范围 [{sens_tensor.min():.6f}, {sens_tensor.max():.6f}]")
        
        # 8. 保存结果
        print("\\n💾 保存Fisher Information...")
        fisher_calc.save_fisher_info("fisher_info_demo.npz")
        
        print("\\n🎉 Fisher Information计算演示完成！")
        print("\\n📋 关键发现:")
        dist = importance_analysis['importance_distribution']
        print(f"   重要性分布: 均值={dist['mean']:.6f}, 标准差={dist['std']:.6f}")
        print(f"   剪枝候选层数: {len(importance_analysis['pruning_candidates'])}")
        print("\\n🚀 准备进行pruning-aware蒸馏!")
        
        # 清理资源
        fisher_calc.cleanup()
        
        return fisher_info, importance_analysis
        
    except Exception as e:
        print(f"\\n❌ 演示失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    demo_fisher_information()
