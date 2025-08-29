#!/usr/bin/env python3
"""
Pruning-Aware Knowledge Distillation 主流程
集成Fisher Information指导的剪枝与知识蒸馏

作者: GitHub Copilot
日期: 2025-08-27
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import json
import time
from collections import defaultdict

from fisher_calculator import FisherInformationCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PruningAwareDistillation:
    """Pruning-Aware Knowledge Distillation主类"""
    
    def __init__(self, 
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 device: str = 'auto',
                 distillation_config: Optional[Dict] = None):
        """
        初始化Pruning-Aware蒸馏
        """
        self.teacher = teacher_model
        self.student = student_model
        self.device = self._setup_device(device)
        
        # 移动模型到设备
        self.teacher.to(self.device)
        self.student.to(self.device)
        
        # 配置
        self.config = self._setup_config(distillation_config)
        
        # Fisher Information计算器
        self.fisher_calc = FisherInformationCalculator(self.student, device=str(self.device))
        
        # 状态跟踪
        self.current_epoch = 0
        self.pruning_masks = {}
        self.fisher_info = {}
        self.distillation_losses = []
        
        logger.info(f"初始化Pruning-Aware蒸馏，设备: {self.device}")
        logger.info(f"教师模型参数: {sum(p.numel() for p in self.teacher.parameters()):,}")
        logger.info(f"学生模型参数: {sum(p.numel() for p in self.student.parameters()):,}")
        
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _setup_config(self, config: Optional[Dict]) -> Dict:
        """设置默认配置"""
        default_config = {
            'temperature': 4.0,
            'alpha': 0.7,  # KD损失权重
            'beta': 0.3,   # 任务损失权重
            'pruning_schedule': {
                'start_epoch': 5,
                'frequency': 10,
                'initial_ratio': 0.1,
                'final_ratio': 0.5,
                'strategy': 'gradual'
            },
            'fisher_config': {
                'update_frequency': 5,
                'num_batches': 10,
                'accumulate': True
            }
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def compute_distillation_loss(self, 
                                student_outputs: torch.Tensor,
                                teacher_outputs: torch.Tensor,
                                true_labels: torch.Tensor,
                                task_criterion: nn.Module) -> Dict[str, torch.Tensor]:
        """计算蒸馏损失"""
        temperature = self.config['temperature']
        alpha = self.config['alpha']
        beta = self.config['beta']
        
        # 任务损失
        task_loss = task_criterion(student_outputs, true_labels)
        
        # 知识蒸馏损失
        if student_outputs.dim() > 1:
            student_soft = F.softmax(student_outputs / temperature, dim=-1)
            teacher_soft = F.softmax(teacher_outputs / temperature, dim=-1)
            kd_loss = F.kl_div(F.log_softmax(student_outputs / temperature, dim=-1), 
                              teacher_soft, reduction='batchmean') * (temperature ** 2)
        else:
            student_soft = torch.sigmoid(student_outputs / temperature)
            teacher_soft = torch.sigmoid(teacher_outputs / temperature)
            kd_loss = F.mse_loss(student_soft, teacher_soft)
        
        # 总损失
        total_loss = alpha * kd_loss + beta * task_loss
        
        return {
            'total_loss': total_loss,
            'kd_loss': kd_loss,
            'task_loss': task_loss
        }
    
    def should_update_fisher(self, epoch: int) -> bool:
        """判断是否应该更新Fisher Information"""
        frequency = self.config['fisher_config']['update_frequency']
        return epoch % frequency == 0
    
    def should_prune(self, epoch: int) -> bool:
        """判断是否应该执行剪枝"""
        schedule = self.config['pruning_schedule']
        start_epoch = schedule['start_epoch']
        frequency = schedule['frequency']
        
        return epoch >= start_epoch and (epoch - start_epoch) % frequency == 0
    
    def compute_pruning_ratio(self, epoch: int) -> float:
        """计算当前epoch的剪枝比例"""
        schedule = self.config['pruning_schedule']
        start_epoch = schedule['start_epoch']
        initial_ratio = schedule['initial_ratio']
        final_ratio = schedule['final_ratio']
        strategy = schedule['strategy']
        
        if epoch < start_epoch:
            return 0.0
        
        if strategy == 'gradual':
            progress = min(1.0, (epoch - start_epoch) / 50)
            return initial_ratio + (final_ratio - initial_ratio) * progress
        else:
            return final_ratio
    
    def apply_pruning_masks(self, masks: Dict[str, torch.Tensor]):
        """应用剪枝掩码到学生模型"""
        masked_params = 0
        total_params = 0
        
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if name in masks:
                    mask = masks[name]
                    param.data *= mask
                    
                    masked_params += (mask == 0).sum().item()
                    total_params += mask.numel()
        
        sparsity = masked_params / total_params if total_params > 0 else 0
        logger.info(f"应用剪枝掩码: {masked_params:,}/{total_params:,} 参数被剪枝")
        
        return sparsity
    
    def train_epoch(self, 
                   train_dataloader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   task_criterion: nn.Module,
                   epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.current_epoch = epoch
        
        # 设置模型模式
        self.teacher.eval()
        self.student.train()
        
        # 统计信息
        epoch_stats = {
            'total_loss': 0.0,
            'kd_loss': 0.0,
            'task_loss': 0.0,
            'batch_count': 0
        }
        
        # 是否需要更新Fisher Information
        update_fisher = self.should_update_fisher(epoch)
        if update_fisher:
            logger.info(f"📊 Epoch {epoch}: 更新Fisher Information...")
            
        for batch_idx, batch_data in enumerate(train_dataloader):
            # 数据预处理
            if isinstance(batch_data, (tuple, list)):
                inputs, targets = batch_data
            else:
                inputs = batch_data
                targets = None
            
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 学生模型输出
            student_outputs = self.student(inputs)
            
            # 教师模型输出
            with torch.no_grad():
                teacher_outputs = self.teacher(inputs)
            
            # 计算损失
            if targets is not None:
                loss_dict = self.compute_distillation_loss(
                    student_outputs, teacher_outputs, targets, task_criterion
                )
            else:
                # 如果没有标签，只使用KD损失
                kd_loss = F.mse_loss(student_outputs, teacher_outputs)
                loss_dict = {
                    'total_loss': kd_loss,
                    'kd_loss': kd_loss,
                    'task_loss': torch.tensor(0.0)
                }
            
            # 反向传播
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            # 应用剪枝掩码（如果存在）
            if self.pruning_masks:
                self.apply_pruning_masks(self.pruning_masks)
            
            # 更新统计
            epoch_stats['total_loss'] += loss_dict['total_loss'].item()
            epoch_stats['kd_loss'] += loss_dict['kd_loss'].item()
            epoch_stats['task_loss'] += loss_dict['task_loss'].item()
            epoch_stats['batch_count'] += 1
        
        # 计算平均损失
        for key in ['total_loss', 'kd_loss', 'task_loss']:
            epoch_stats[key] /= epoch_stats['batch_count']
        
        # 更新Fisher Information
        if update_fisher:
            try:
                self.fisher_info = self.fisher_calc.compute_fisher_information(
                    train_dataloader, 
                    task_criterion,
                    num_batches=self.config['fisher_config']['num_batches']
                )
                logger.info(f"✅ Fisher Information更新完成，覆盖 {len(self.fisher_info)} 层")
            except Exception as e:
                logger.warning(f"Fisher Information更新失败: {e}")
        
        # 执行剪枝
        if self.should_prune(epoch) and self.fisher_info:
            pruning_ratio = self.compute_pruning_ratio(epoch)
            logger.info(f"✂️ Epoch {epoch}: 执行剪枝，比例 {pruning_ratio:.1%}")
            
            try:
                self.pruning_masks = self.fisher_calc.generate_pruning_mask(
                    self.fisher_info, pruning_ratio, 'global'
                )
                sparsity = self.apply_pruning_masks(self.pruning_masks)
                epoch_stats['sparsity'] = sparsity
                logger.info(f"✅ 剪枝完成，当前稀疏度: {sparsity:.1%}")
            except Exception as e:
                logger.warning(f"剪枝执行失败: {e}")
        
        return epoch_stats
    
    def evaluate(self, 
                val_dataloader: DataLoader,
                task_criterion: nn.Module) -> Dict[str, float]:
        """评估模型性能"""
        self.student.eval()
        self.teacher.eval()
        
        total_loss = 0.0
        kd_loss = 0.0
        task_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch_data in val_dataloader:
                if isinstance(batch_data, (tuple, list)):
                    inputs, targets = batch_data
                else:
                    inputs = batch_data
                    targets = None
                
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # 前向传播
                student_outputs = self.student(inputs)
                teacher_outputs = self.teacher(inputs)
                
                # 计算损失
                if targets is not None:
                    loss_dict = self.compute_distillation_loss(
                        student_outputs, teacher_outputs, targets, task_criterion
                    )
                else:
                    kd_loss_val = F.mse_loss(student_outputs, teacher_outputs)
                    loss_dict = {
                        'total_loss': kd_loss_val,
                        'kd_loss': kd_loss_val,
                        'task_loss': torch.tensor(0.0)
                    }
                
                total_loss += loss_dict['total_loss'].item()
                kd_loss += loss_dict['kd_loss'].item()
                task_loss += loss_dict['task_loss'].item()
                batch_count += 1
        
        # 计算稀疏度
        sparsity = 0.0
        if self.pruning_masks:
            total_params = sum(mask.numel() for mask in self.pruning_masks.values())
            masked_params = sum((mask == 0).sum().item() for mask in self.pruning_masks.values())
            sparsity = masked_params / total_params if total_params > 0 else 0.0
        
        return {
            'val_total_loss': total_loss / batch_count,
            'val_kd_loss': kd_loss / batch_count,
            'val_task_loss': task_loss / batch_count,
            'sparsity': sparsity
        }
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        student_params = sum(p.numel() for p in self.student.parameters())
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        
        # 有效参数（未被剪枝）
        effective_params = student_params
        if self.pruning_masks:
            effective_params = sum(mask.sum().item() for mask in self.pruning_masks.values())
        
        compression_ratio = effective_params / teacher_params
        
        return {
            'student_params': student_params,
            'teacher_params': teacher_params,
            'effective_params': int(effective_params),
            'compression_ratio': compression_ratio,
            'sparsity': 1 - (effective_params / student_params),
            'fisher_layers': len(self.fisher_info),
            'pruning_masks': len(self.pruning_masks)
        }
    
    def save_checkpoint(self, save_path: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'pruning_masks': self.pruning_masks,
            'fisher_info': {k: v.cpu() for k, v in self.fisher_info.items()},
            'config': self.config,
            'model_stats': self.get_model_statistics()
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, save_path)
        logger.info(f"💾 保存检查点到: {save_path}")
