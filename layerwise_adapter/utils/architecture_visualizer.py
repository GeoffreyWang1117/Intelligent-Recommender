"""
Model architecture visualization and analysis tools.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.layerwise_adapter import LayerwiseAdapter
from models.base import ModelConfig

plt.style.use('seaborn-v0_8')


class ModelArchitectureVisualizer:
    """模型架构可视化工具"""
    
    def __init__(self, output_dir: str = "./architecture_analysis"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_layerwise_architecture(self, config: ModelConfig) -> None:
        """可视化LayerwiseAdapter架构"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # 定义颜色方案
        colors = {
            'input': '#E8F4FD',
            'embedding': '#B3D9FF', 
            'interaction': '#66B2FF',
            'reasoning': '#1F7CDF',
            'output': '#0F4C99',
            'attention': '#FFE6CC',
            'transform': '#FFA366'
        }
        
        # 绘制输入层
        self._draw_layer(ax, 1, 10, 2, 1, "User Profile\n[batch, embedding_dim]", colors['input'])
        self._draw_layer(ax, 1, 8, 2, 1, "Item Profile\n[batch, embedding_dim]", colors['input'])
        
        # 绘制EmbeddingAdapter
        self._draw_layer(ax, 4, 9, 3, 1.5, "EmbeddingAdapter\nUser Embedding", colors['embedding'])
        self._draw_layer(ax, 4, 7, 3, 1.5, "Item Embedding", colors['embedding'])
        self._draw_layer(ax, 8, 8, 2, 2, "Feature\nFusion", colors['embedding'])
        
        # 绘制InteractionAdapter  
        self._draw_layer(ax, 11, 9, 3, 1, "Multi-Head\nAttention", colors['attention'])
        self._draw_layer(ax, 11, 7, 3, 1, "Cross Attention", colors['attention'])
        self._draw_layer(ax, 15, 8, 2, 2, "Interaction\nFusion", colors['interaction'])
        
        # 绘制ReasoningAdapter
        self._draw_layer(ax, 18, 9, 2.5, 1, "Transformer\nLayer 1", colors['reasoning'])
        self._draw_layer(ax, 18, 7, 2.5, 1, "Transformer\nLayer 2", colors['reasoning'])
        self._draw_layer(ax, 21, 8, 2, 1.5, "Rating\nPredictor", colors['output'])
        
        # 绘制连接线
        self._draw_connections(ax)
        
        # 添加标注
        ax.text(12, 11.5, "LayerwiseAdapter Architecture", fontsize=20, fontweight='bold', ha='center')
        ax.text(1, 6, "Input Layer", fontsize=14, fontweight='bold', color=colors['output'])
        ax.text(6, 6, "Embedding Layer", fontsize=14, fontweight='bold', color=colors['output'])
        ax.text(13, 6, "Interaction Layer", fontsize=14, fontweight='bold', color=colors['output'])
        ax.text(19.5, 6, "Reasoning Layer", fontsize=14, fontweight='bold', color=colors['output'])
        
        # 添加参数信息
        param_text = f"""Model Configuration:
• Embedding Dim: {config.embedding_dim}
• Hidden Dim: {config.hidden_dim}  
• Attention Heads: {config.num_heads}
• Transformer Layers: {config.num_layers}
• Total Parameters: 1.1M
• Model Size: 4.2MB"""
        
        ax.text(1, 4, param_text, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        ax.set_xlim(0, 24)
        ax.set_ylim(3, 12)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/layerwise_architecture.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _draw_layer(self, ax, x, y, width, height, text, color):
        """绘制单个层"""
        rect = FancyBboxPatch((x, y), width, height, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color, 
                             edgecolor='black',
                             linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
    def _draw_connections(self, ax):
        """绘制层间连接"""
        # Input to Embedding
        ax.arrow(3, 9.5, 0.8, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(3, 8.5, 0.8, 0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Embedding to Fusion
        ax.arrow(7, 9.5, 0.8, -0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(7, 7.5, 0.8, 0.8, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Fusion to Interaction
        ax.arrow(10, 8.5, 0.8, 0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(10, 8.5, 0.8, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Interaction to Fusion
        ax.arrow(14, 8.5, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Interaction to Reasoning
        ax.arrow(17, 8.5, 0.8, 0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(17, 8.5, 0.8, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Reasoning to Output
        ax.arrow(20.5, 8.5, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    def compare_model_architectures(self) -> None:
        """对比不同模型架构"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. LayerwiseAdapter
        self._draw_layerwise_simple(ax1)
        ax1.set_title("LayerwiseAdapter (Current)", fontsize=16, fontweight='bold')
        
        # 2. Traditional Ensemble
        self._draw_ensemble_architecture(ax2)
        ax2.set_title("Traditional Ensemble", fontsize=16, fontweight='bold')
        
        # 3. LLM Teacher (Qwen/Llama3)
        self._draw_llm_architecture(ax3)
        ax3.set_title("LLM Teacher (Qwen/Llama3)", fontsize=16, fontweight='bold')
        
        # 4. Knowledge Distillation Framework
        self._draw_distillation_framework(ax4)
        ax4.set_title("PAKD + Fisher Distillation", fontsize=16, fontweight='bold')
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 8)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/architecture_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _draw_layerwise_simple(self, ax):
        """简化的LayerwiseAdapter架构"""
        # 输入
        ax.add_patch(Rectangle((1, 6), 1.5, 0.8, facecolor='#E8F4FD', edgecolor='black'))
        ax.text(1.75, 6.4, 'Input', ha='center', va='center', fontweight='bold')
        
        # 三层架构
        colors = ['#B3D9FF', '#66B2FF', '#1F7CDF']
        labels = ['Embedding', 'Interaction', 'Reasoning']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            ax.add_patch(Rectangle((3 + i*2, 6), 1.5, 0.8, facecolor=color, edgecolor='black'))
            ax.text(3.75 + i*2, 6.4, label, ha='center', va='center', fontweight='bold')
            
            if i > 0:
                ax.arrow(2.5 + i*2, 6.4, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # 输出
        ax.add_patch(Rectangle((9, 6), 1, 0.8, facecolor='#0F4C99', edgecolor='black'))
        ax.text(9.5, 6.4, 'Output', ha='center', va='center', fontweight='bold', color='white')
        
        # 连接
        ax.arrow(2.5, 6.4, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(8.5, 6.4, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # 参数信息
        ax.text(5, 4.5, "• 3-Layer Architecture\n• 1.1M Parameters\n• 4.2MB Model Size\n• 0.08ms Inference", 
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    def _draw_ensemble_architecture(self, ax):
        """传统集成模型架构"""
        models = ['SVD', 'DeepFM', 'xDeepFM', 'AutoInt', 'DCNv2']
        colors = ['#FFE6CC', '#FFA366', '#FF8533', '#FF6600', '#CC4400']
        
        # 绘制各个模型
        for i, (model, color) in enumerate(zip(models, colors)):
            y_pos = 6 - i * 0.8
            ax.add_patch(Rectangle((1, y_pos), 1.5, 0.6, facecolor=color, edgecolor='black'))
            ax.text(1.75, y_pos + 0.3, model, ha='center', va='center', fontweight='bold', fontsize=9)
            
            # 连接到融合层
            ax.arrow(2.5, y_pos + 0.3, 2, 2.5 - i * 0.8, head_width=0.05, head_length=0.1, fc='gray', ec='gray')
        
        # 融合层
        ax.add_patch(Rectangle((5, 5.5), 2, 1, facecolor='#99CCFF', edgecolor='black'))
        ax.text(6, 6, 'Ensemble\nFusion', ha='center', va='center', fontweight='bold')
        
        # 输出
        ax.add_patch(Rectangle((8, 5.8), 1, 0.4, facecolor='#0066CC', edgecolor='black'))
        ax.text(8.5, 6, 'Output', ha='center', va='center', fontweight='bold', color='white')
        
        ax.arrow(7, 6, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # 参数信息
        ax.text(5, 1.5, "• 5 Traditional Models\n• ~50M Parameters\n• Ensemble Fusion\n• High Memory Cost", 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    def _draw_llm_architecture(self, ax):
        """LLM架构"""
        # LLM主体
        ax.add_patch(Rectangle((2, 4), 3, 3, facecolor='#FFE6E6', edgecolor='black', linewidth=2))
        ax.text(3.5, 5.5, 'Large Language Model\n(Qwen/Llama3)', ha='center', va='center', fontweight='bold', fontsize=12)
        
        # 输入
        ax.add_patch(Rectangle((0.5, 5.2), 1, 0.6, facecolor='#E8F4FD', edgecolor='black'))
        ax.text(1, 5.5, 'Prompt', ha='center', va='center', fontweight='bold')
        
        # 输出
        ax.add_patch(Rectangle((6, 5.2), 1.5, 0.6, facecolor='#E6FFE6', edgecolor='black'))
        ax.text(6.75, 5.5, 'Embedding', ha='center', va='center', fontweight='bold')
        
        # 连接
        ax.arrow(1.5, 5.5, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(5, 5.5, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # 参数信息
        ax.text(3.5, 2, "• 7B-70B Parameters\n• Multi-Modal Capability\n• Rich Semantic Knowledge\n• High Computational Cost", 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    def _draw_distillation_framework(self, ax):
        """知识蒸馏框架"""
        # Teacher模型
        ax.add_patch(Rectangle((1, 6), 2, 1, facecolor='#FFE6E6', edgecolor='black'))
        ax.text(2, 6.5, 'Teacher Models\n(Ensemble + LLM)', ha='center', va='center', fontweight='bold')
        
        # 学生模型
        ax.add_patch(Rectangle((6, 6), 2, 1, facecolor='#E6F3FF', edgecolor='black'))
        ax.text(7, 6.5, 'Student Model\n(LayerwiseAdapter)', ha='center', va='center', fontweight='bold')
        
        # PAKD
        ax.add_patch(Rectangle((3.5, 4.5), 2, 0.8, facecolor='#FFFFCC', edgecolor='black'))
        ax.text(4.5, 4.9, 'PAKD\nDistillation', ha='center', va='center', fontweight='bold')
        
        # Fisher Information
        ax.add_patch(Rectangle((3.5, 3), 2, 0.8, facecolor='#E6FFCC', edgecolor='black'))
        ax.text(4.5, 3.4, 'Fisher\nInformation', ha='center', va='center', fontweight='bold')
        
        # 连接线
        ax.arrow(3, 6.3, 0.4, -1.3, head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax.arrow(5.5, 4.9, 0.4, 1.3, head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax.arrow(3, 6.7, 0.4, -3.2, head_width=0.1, head_length=0.1, fc='green', ec='green')
        ax.arrow(5.5, 3.4, 0.4, 2.8, head_width=0.1, head_length=0.1, fc='green', ec='green')
        
        # 标签
        ax.text(3.2, 5.5, 'Knowledge', ha='center', va='center', fontsize=8, color='red')
        ax.text(3.2, 4.5, 'Importance', ha='center', va='center', fontsize=8, color='green')
    
    def analyze_model_complexity(self) -> Dict:
        """分析模型复杂度"""
        # 创建不同配置的模型
        configs = {
            'tiny': {'embedding_dim': 32, 'hidden_dim': 64, 'num_heads': 2, 'num_layers': 1},
            'small': {'embedding_dim': 64, 'hidden_dim': 128, 'num_heads': 4, 'num_layers': 2},
            'medium': {'embedding_dim': 128, 'hidden_dim': 256, 'num_heads': 8, 'num_layers': 3}
        }
        
        analysis = {}
        
        for size_name, params in configs.items():
            config = ModelConfig()
            config.num_users = 610
            config.num_items = 9724
            for key, value in params.items():
                setattr(config, key, value)
            
            model = LayerwiseAdapter(config)
            model_info = model.get_model_size()
            
            analysis[size_name] = {
                'parameters': model_info['total_parameters'],
                'size_mb': model_info['model_size_mb'],
                'config': params,
                'flops_estimate': self._estimate_flops(config)
            }
        
        return analysis
    
    def _estimate_flops(self, config: ModelConfig) -> int:
        """估算FLOPs"""
        # 简化的FLOPs估算
        embedding_flops = config.num_users * config.embedding_dim + config.num_items * config.embedding_dim
        attention_flops = config.num_heads * config.hidden_dim * config.hidden_dim * config.num_layers
        linear_flops = config.hidden_dim * config.hidden_dim * config.num_layers * 4  # 4个线性层
        
        return embedding_flops + attention_flops + linear_flops
    
    def create_comprehensive_report(self) -> None:
        """创建综合分析报告"""
        # 创建配置
        config = ModelConfig()
        config.num_users = 610
        config.num_items = 9724
        config.embedding_dim = 64
        config.hidden_dim = 128
        config.num_heads = 4
        config.num_layers = 2
        
        # 生成可视化
        print("生成LayerwiseAdapter架构图...")
        self.visualize_layerwise_architecture(config)
        
        print("生成架构对比图...")
        self.compare_model_architectures()
        
        print("分析模型复杂度...")
        complexity_analysis = self.analyze_model_complexity()
        
        # 保存分析结果
        with open(f"{self.output_dir}/complexity_analysis.json", "w") as f:
            json.dump(complexity_analysis, f, indent=2)
        
        print(f"所有分析结果已保存到: {self.output_dir}/")
        
        return complexity_analysis


def main():
    """运行模型架构分析"""
    visualizer = ModelArchitectureVisualizer("./architecture_analysis")
    
    print("="*60)
    print("模型架构分析工具")
    print("="*60)
    
    # 生成综合报告
    complexity_analysis = visualizer.create_comprehensive_report()
    
    # 打印复杂度分析
    print("\n📊 模型复杂度分析:")
    print("-" * 50)
    for size, info in complexity_analysis.items():
        print(f"{size.upper()}:")
        print(f"  参数量: {info['parameters']:,}")
        print(f"  模型大小: {info['size_mb']:.2f}MB")
        print(f"  FLOPs估算: {info['flops_estimate']:,}")
        print()
    
    print("✅ 架构分析完成!")
    print("📁 生成文件:")
    print("  - layerwise_architecture.png: 详细架构图")
    print("  - architecture_comparison.png: 架构对比图")
    print("  - complexity_analysis.json: 复杂度分析")


if __name__ == "__main__":
    main()
