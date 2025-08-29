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
        
        Args:
            teacher_model: 教师模型
            student_model: 学生模型  
            device: 计算设备
            distillation_config: 蒸馏配置
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
                'strategy': 'gradual'  # 'gradual' or 'aggressive'
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
        """
        计算蒸馏损失
        
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出  
            true_labels: 真实标签
            task_criterion: 任务损失函数
            
        Returns:
            各种损失组件
        """
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
        }\n    \n    def should_update_fisher(self, epoch: int) -> bool:\n        \"\"\"判断是否应该更新Fisher Information\"\"\"\n        frequency = self.config['fisher_config']['update_frequency']\n        return epoch % frequency == 0\n    \n    def should_prune(self, epoch: int) -> bool:\n        \"\"\"判断是否应该执行剪枝\"\"\"\n        schedule = self.config['pruning_schedule']\n        start_epoch = schedule['start_epoch']\n        frequency = schedule['frequency']\n        \n        return epoch >= start_epoch and (epoch - start_epoch) % frequency == 0\n    \n    def compute_pruning_ratio(self, epoch: int) -> float:\n        \"\"\"计算当前epoch的剪枝比例\"\"\"\n        schedule = self.config['pruning_schedule']\n        start_epoch = schedule['start_epoch']\n        initial_ratio = schedule['initial_ratio']\n        final_ratio = schedule['final_ratio']\n        strategy = schedule['strategy']\n        \n        if epoch < start_epoch:\n            return 0.0\n        \n        if strategy == 'gradual':\n            # 渐进式剪枝\n            progress = min(1.0, (epoch - start_epoch) / 50)  # 50个epoch达到最终比例\n            return initial_ratio + (final_ratio - initial_ratio) * progress\n        else:\n            # 激进式剪枝\n            return final_ratio\n    \n    def apply_pruning_masks(self, masks: Dict[str, torch.Tensor]):\n        \"\"\"应用剪枝掩码到学生模型\"\"\"\n        masked_params = 0\n        total_params = 0\n        \n        with torch.no_grad():\n            for name, param in self.student.named_parameters():\n                if name in masks:\n                    mask = masks[name]\n                    param.data *= mask\n                    \n                    masked_params += (mask == 0).sum().item()\n                    total_params += mask.numel()\n        \n        sparsity = masked_params / total_params if total_params > 0 else 0\n        logger.info(f\"应用剪枝掩码: {masked_params:,}/{total_params:,} 参数被剪枝 (稀疏度: {sparsity:.1%})\")\n        \n        return sparsity\n    \n    def train_epoch(self, \n                   train_dataloader: DataLoader,\n                   optimizer: torch.optim.Optimizer,\n                   task_criterion: nn.Module,\n                   epoch: int) -> Dict[str, float]:\n        \"\"\"\n        训练一个epoch\n        \n        Args:\n            train_dataloader: 训练数据加载器\n            optimizer: 优化器\n            task_criterion: 任务损失函数\n            epoch: 当前epoch\n            \n        Returns:\n            训练统计信息\n        \"\"\"\n        self.current_epoch = epoch\n        \n        # 设置模型模式\n        self.teacher.eval()  # 教师模型始终评估模式\n        self.student.train()\n        \n        # 统计信息\n        epoch_stats = {\n            'total_loss': 0.0,\n            'kd_loss': 0.0,\n            'task_loss': 0.0,\n            'batch_count': 0\n        }\n        \n        # 是否需要更新Fisher Information\n        update_fisher = self.should_update_fisher(epoch)\n        if update_fisher:\n            logger.info(f\"📊 Epoch {epoch}: 更新Fisher Information...\")\n            \n        for batch_idx, batch_data in enumerate(train_dataloader):\n            # 数据预处理\n            if isinstance(batch_data, (tuple, list)):\n                inputs, targets = batch_data\n            else:\n                inputs = batch_data\n                targets = None\n            \n            inputs = inputs.to(self.device)\n            if targets is not None:\n                targets = targets.to(self.device)\n            \n            # 前向传播\n            optimizer.zero_grad()\n            \n            # 学生模型输出\n            student_outputs = self.student(inputs)\n            \n            # 教师模型输出\n            with torch.no_grad():\n                teacher_outputs = self.teacher(inputs)\n            \n            # 计算损失\n            if targets is not None:\n                loss_dict = self.compute_distillation_loss(\n                    student_outputs, teacher_outputs, targets, task_criterion\n                )\n            else:\n                # 如果没有标签，只使用KD损失\n                kd_loss = F.mse_loss(student_outputs, teacher_outputs)\n                loss_dict = {\n                    'total_loss': kd_loss,\n                    'kd_loss': kd_loss,\n                    'task_loss': torch.tensor(0.0)\n                }\n            \n            # 反向传播\n            loss_dict['total_loss'].backward()\n            optimizer.step()\n            \n            # 应用剪枝掩码（如果存在）\n            if self.pruning_masks:\n                self.apply_pruning_masks(self.pruning_masks)\n            \n            # 更新统计\n            epoch_stats['total_loss'] += loss_dict['total_loss'].item()\n            epoch_stats['kd_loss'] += loss_dict['kd_loss'].item()\n            epoch_stats['task_loss'] += loss_dict['task_loss'].item()\n            epoch_stats['batch_count'] += 1\n            \n            # 打印进度\n            if batch_idx % 10 == 0:\n                logger.debug(f\"Batch {batch_idx}: Loss={loss_dict['total_loss'].item():.4f}\")\n        \n        # 计算平均损失\n        for key in ['total_loss', 'kd_loss', 'task_loss']:\n            epoch_stats[key] /= epoch_stats['batch_count']\n        \n        # 更新Fisher Information\n        if update_fisher:\n            try:\n                self.fisher_info = self.fisher_calc.compute_fisher_information(\n                    train_dataloader, \n                    task_criterion,\n                    num_batches=self.config['fisher_config']['num_batches']\n                )\n                logger.info(f\"✅ Fisher Information更新完成，覆盖 {len(self.fisher_info)} 层\")\n            except Exception as e:\n                logger.warning(f\"Fisher Information更新失败: {e}\")\n        \n        # 执行剪枝\n        if self.should_prune(epoch) and self.fisher_info:\n            pruning_ratio = self.compute_pruning_ratio(epoch)\n            logger.info(f\"✂️ Epoch {epoch}: 执行剪枝，比例 {pruning_ratio:.1%}\")\n            \n            try:\n                self.pruning_masks = self.fisher_calc.generate_pruning_mask(\n                    self.fisher_info, pruning_ratio, 'global'\n                )\n                sparsity = self.apply_pruning_masks(self.pruning_masks)\n                epoch_stats['sparsity'] = sparsity\n                logger.info(f\"✅ 剪枝完成，当前稀疏度: {sparsity:.1%}\")\n            except Exception as e:\n                logger.warning(f\"剪枝执行失败: {e}\")\n        \n        return epoch_stats\n    \n    def evaluate(self, \n                val_dataloader: DataLoader,\n                task_criterion: nn.Module) -> Dict[str, float]:\n        \"\"\"\n        评估模型性能\n        \n        Args:\n            val_dataloader: 验证数据加载器\n            task_criterion: 任务损失函数\n            \n        Returns:\n            评估结果\n        \"\"\"\n        self.student.eval()\n        self.teacher.eval()\n        \n        total_loss = 0.0\n        kd_loss = 0.0\n        task_loss = 0.0\n        batch_count = 0\n        \n        with torch.no_grad():\n            for batch_data in val_dataloader:\n                if isinstance(batch_data, (tuple, list)):\n                    inputs, targets = batch_data\n                else:\n                    inputs = batch_data\n                    targets = None\n                \n                inputs = inputs.to(self.device)\n                if targets is not None:\n                    targets = targets.to(self.device)\n                \n                # 前向传播\n                student_outputs = self.student(inputs)\n                teacher_outputs = self.teacher(inputs)\n                \n                # 计算损失\n                if targets is not None:\n                    loss_dict = self.compute_distillation_loss(\n                        student_outputs, teacher_outputs, targets, task_criterion\n                    )\n                else:\n                    kd_loss_val = F.mse_loss(student_outputs, teacher_outputs)\n                    loss_dict = {\n                        'total_loss': kd_loss_val,\n                        'kd_loss': kd_loss_val,\n                        'task_loss': torch.tensor(0.0)\n                    }\n                \n                total_loss += loss_dict['total_loss'].item()\n                kd_loss += loss_dict['kd_loss'].item()\n                task_loss += loss_dict['task_loss'].item()\n                batch_count += 1\n        \n        # 计算稀疏度\n        sparsity = 0.0\n        if self.pruning_masks:\n            total_params = sum(mask.numel() for mask in self.pruning_masks.values())\n            masked_params = sum((mask == 0).sum().item() for mask in self.pruning_masks.values())\n            sparsity = masked_params / total_params if total_params > 0 else 0.0\n        \n        return {\n            'val_total_loss': total_loss / batch_count,\n            'val_kd_loss': kd_loss / batch_count,\n            'val_task_loss': task_loss / batch_count,\n            'sparsity': sparsity\n        }\n    \n    def get_model_statistics(self) -> Dict[str, Any]:\n        \"\"\"获取模型统计信息\"\"\"\n        student_params = sum(p.numel() for p in self.student.parameters())\n        teacher_params = sum(p.numel() for p in self.teacher.parameters())\n        \n        # 有效参数（未被剪枝）\n        effective_params = student_params\n        if self.pruning_masks:\n            effective_params = sum(mask.sum().item() for mask in self.pruning_masks.values())\n        \n        compression_ratio = effective_params / teacher_params\n        \n        return {\n            'student_params': student_params,\n            'teacher_params': teacher_params,\n            'effective_params': int(effective_params),\n            'compression_ratio': compression_ratio,\n            'sparsity': 1 - (effective_params / student_params),\n            'fisher_layers': len(self.fisher_info),\n            'pruning_masks': len(self.pruning_masks)\n        }\n    \n    def save_checkpoint(self, save_path: str, epoch: int, optimizer_state: Dict = None):\n        \"\"\"保存检查点\"\"\"\n        checkpoint = {\n            'epoch': epoch,\n            'student_state_dict': self.student.state_dict(),\n            'pruning_masks': self.pruning_masks,\n            'fisher_info': {k: v.cpu() for k, v in self.fisher_info.items()},\n            'config': self.config,\n            'model_stats': self.get_model_statistics()\n        }\n        \n        if optimizer_state:\n            checkpoint['optimizer_state_dict'] = optimizer_state\n        \n        torch.save(checkpoint, save_path)\n        logger.info(f\"💾 保存检查点到: {save_path}\")\n    \n    def load_checkpoint(self, load_path: str) -> Dict:\n        \"\"\"加载检查点\"\"\"\n        checkpoint = torch.load(load_path, map_location=self.device)\n        \n        self.student.load_state_dict(checkpoint['student_state_dict'])\n        self.pruning_masks = checkpoint.get('pruning_masks', {})\n        self.fisher_info = {k: v.to(self.device) for k, v in checkpoint.get('fisher_info', {}).items()}\n        self.config.update(checkpoint.get('config', {}))\n        \n        logger.info(f\"📂 从 {load_path} 加载检查点\")\n        return checkpoint
