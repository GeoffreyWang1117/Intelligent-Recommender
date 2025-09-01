#!/usr/bin/env python3
"""
LayerwiseAdapter架构可视化和模型导出
支持Netron可视化和架构分析
"""

import sys
import os
import torch
import torch.nn as nn
import json
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

# 添加路径
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system')
sys.path.append('/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced')

def export_model_for_netron():
    """导出模型为ONNX格式供Netron可视化"""
    try:
        # 导入模型组件
        import importlib.util
        
        # 导入LayerwiseAdapterV2
        spec = importlib.util.spec_from_file_location(
            "layerwise_adapter_v2", 
            "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/models/layerwise_adapter_v2.py"
        )
        layerwise_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(layerwise_module)
        
        # 创建模型配置
        model_config = {
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_users': 610,
            'num_items': 9724,
            'teacher_config': {
                'fusion_mode': 'weighted',
                'default_weights': {'ensemble': 0.6, 'llama': 0.4},
                'fisher_config': {}
            },
            'fusion_config': {
                'embedding_dim': 64,
                'teacher_count': 2,
                'fusion_hidden_dim': 128
            }
        }
        
        # 创建模型
        model = layerwise_module.LayerwiseAdapterV2(model_config)
        model.eval()
        
        # 创建示例输入
        dummy_user = torch.LongTensor([1])
        dummy_item = torch.LongTensor([1])
        
        # 导出为ONNX
        save_dir = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/analysis"
        os.makedirs(save_dir, exist_ok=True)
        
        onnx_path = os.path.join(save_dir, "layerwise_adapter_v2.onnx")
        
        torch.onnx.export(
            model,
            (dummy_user, dummy_item),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['user_id', 'item_id'],
            output_names=['prediction', 'final_embedding', 'teacher_knowledge', 'layer_outputs'],
            dynamic_axes={
                'user_id': {0: 'batch_size'},
                'item_id': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX模型已导出到: {onnx_path}")
        print(f"🌐 可以使用Netron查看: https://netron.app/ 或 netron {onnx_path}")
        
        return model, onnx_path
        
    except Exception as e:
        print(f"❌ 模型导出失败: {e}")
        return None, None

def analyze_model_architecture(model):
    """分析模型架构"""
    if model is None:
        return {}
    
    architecture_info = {
        'total_parameters': 0,
        'trainable_parameters': 0,
        'layers': {},
        'memory_usage': 0
    }
    
    # 统计参数
    for name, param in model.named_parameters():
        num_params = param.numel()
        architecture_info['total_parameters'] += num_params
        if param.requires_grad:
            architecture_info['trainable_parameters'] += num_params
        
        # 按层分组
        layer_name = name.split('.')[0]
        if layer_name not in architecture_info['layers']:
            architecture_info['layers'][layer_name] = {
                'parameters': 0,
                'shape_info': []
            }
        
        architecture_info['layers'][layer_name]['parameters'] += num_params
        architecture_info['layers'][layer_name]['shape_info'].append({
            'name': name,
            'shape': list(param.shape),
            'parameters': num_params
        })
    
    # 估算内存使用（假设FP32）
    architecture_info['memory_usage'] = architecture_info['total_parameters'] * 4 / (1024 * 1024)  # MB
    
    return architecture_info

def create_architecture_diagram():
    """创建架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # 定义层的位置和大小
    layers = [
        {'name': 'Input Layer', 'pos': (2, 10), 'size': (3, 1), 'color': '#E8F4FD'},
        {'name': 'User Embedding\n(610 users)', 'pos': (0.5, 8.5), 'size': (2.5, 1), 'color': '#B3E5FC'},
        {'name': 'Item Embedding\n(9724 items)', 'pos': (3.5, 8.5), 'size': (2.5, 1), 'color': '#B3E5FC'},
        
        {'name': 'Teacher Knowledge\nExtraction', 'pos': (7, 9), 'size': (3, 1.5), 'color': '#C8E6C9'},
        {'name': 'Ensemble Teacher\n(SVD+AutoInt+xDeepFM)', 'pos': (11, 10), 'size': (3, 1), 'color': '#DCEDC8'},
        {'name': 'LLM Teacher\n(Llama3)', 'pos': (11, 8), 'size': (3, 1), 'color': '#DCEDC8'},
        
        {'name': 'Layer 1: Embedding Adapter\n(Multi-source Fusion)', 'pos': (2, 7), 'size': (4, 1), 'color': '#FFE0B2'},
        {'name': 'Layer 2: Interaction Adapter\n(Cross-Teacher Attention)', 'pos': (2, 5.5), 'size': (4, 1), 'color': '#FFCC80'},
        {'name': 'Layer 3: Reasoning Adapter\n(Integrated Reasoning)', 'pos': (2, 4), 'size': (4, 1), 'color': '#FFB74D'},
        
        {'name': 'Multi-Teacher Fusion\nModule', 'pos': (7, 6), 'size': (3, 2), 'color': '#F8BBD9'},
        {'name': 'Fisher-Guided\nSelector', 'pos': (11, 6), 'size': (2.5, 1), 'color': '#E1BEE7'},
        
        {'name': 'Final Prediction\nLayer', 'pos': (2, 2.5), 'size': (4, 1), 'color': '#FFCDD2'},
        {'name': 'Output:\nRating [1-5]', 'pos': (2, 1), 'size': (4, 1), 'color': '#F8F8F8'},
    ]
    
    # 绘制层
    for layer in layers:
        rect = FancyBboxPatch(
            layer['pos'], layer['size'][0], layer['size'][1],
            boxstyle="round,pad=0.1",
            facecolor=layer['color'],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # 添加文本
        ax.text(
            layer['pos'][0] + layer['size'][0]/2,
            layer['pos'][1] + layer['size'][1]/2,
            layer['name'],
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            wrap=True
        )
    
    # 绘制连接线
    connections = [
        # Input to embeddings
        ((3.5, 10), (1.75, 9.5)),
        ((3.5, 10), (4.75, 9.5)),
        
        # Embeddings to Layer 1
        ((1.75, 8.5), (3, 8)),
        ((4.75, 8.5), (5, 8)),
        
        # Teacher connections
        ((7, 9.75), (11, 10.5)),
        ((7, 9.25), (11, 8.5)),
        ((10, 9), (11, 6.5)),
        
        # Layer connections
        ((4, 7), (4, 6.5)),
        ((4, 5.5), (4, 5)),
        
        # Fusion connections
        ((6, 7), (7, 7)),
        ((6, 6), (7, 7)),
        ((6, 4.5), (7, 6.5)),
        
        # To output
        ((4, 4), (4, 3.5)),
        ((4, 2.5), (4, 2)),
    ]
    
    for start, end in connections:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.7)
    
    # 添加注释
    ax.text(7, 11.5, 'Multi-Teacher Knowledge Distillation', fontsize=14, fontweight='bold', ha='center')
    ax.text(1, 6, 'Layerwise\nAdapter\nStack', fontsize=12, fontweight='bold', ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
    
    # 设置坐标轴
    ax.set_xlim(-1, 15)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('LayerwiseAdapter V2 Architecture\n(Multi-Teacher Fusion for Recommendation)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 保存图形
    save_dir = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/analysis"
    plt.savefig(os.path.join(save_dir, 'architecture_diagram.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 架构图已保存到: {save_dir}/architecture_diagram.png")

def create_detailed_flow_diagram():
    """创建详细的数据流图"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    
    # 定义详细的数据流
    flows = [
        # 输入层
        {'name': 'User ID\n(user_id)', 'pos': (1, 9), 'size': (1.5, 0.8), 'color': '#E3F2FD'},
        {'name': 'Item ID\n(item_id)', 'pos': (3, 9), 'size': (1.5, 0.8), 'color': '#E3F2FD'},
        
        # 嵌入层
        {'name': 'User Emb\n[B, 64]', 'pos': (1, 7.5), 'size': (1.5, 0.8), 'color': '#BBDEFB'},
        {'name': 'Item Emb\n[B, 64]', 'pos': (3, 7.5), 'size': (1.5, 0.8), 'color': '#BBDEFB'},
        
        # Teacher知识
        {'name': 'Teacher\nKnowledge', 'pos': (6, 8.5), 'size': (2, 1), 'color': '#C8E6C9'},
        {'name': 'Attention\nContext [1, 64]', 'pos': (9, 9), 'size': (2, 0.8), 'color': '#DCEDC8'},
        {'name': 'Reasoning\nContext [1, 64]', 'pos': (9, 7.5), 'size': (2, 0.8), 'color': '#DCEDC8'},
        
        # Layer 1: EmbeddingAdapter
        {'name': 'Embedding\nAdapter', 'pos': (1, 6), 'size': (3.5, 1), 'color': '#FFE0B2'},
        {'name': 'User Emb\n[B, 64]', 'pos': (6, 6.5), 'size': (1.5, 0.6), 'color': '#FFECB3'},
        {'name': 'Item Emb\n[B, 64]', 'pos': (6, 5.5), 'size': (1.5, 0.6), 'color': '#FFECB3'},
        
        # Layer 2: InteractionAdapter
        {'name': 'Interaction\nAdapter', 'pos': (1, 4.5), 'size': (3.5, 1), 'color': '#FFCC80'},
        {'name': 'Multi-Head\nAttention', 'pos': (6, 4.8), 'size': (2, 0.8), 'color': '#FFD54F'},
        {'name': 'Interaction\nEmb [B, 64]', 'pos': (9, 4.5), 'size': (2, 0.8), 'color': '#FFE082'},
        
        # Layer 3: ReasoningAdapter
        {'name': 'Reasoning\nAdapter', 'pos': (1, 3), 'size': (3.5, 1), 'color': '#FFB74D'},
        {'name': 'Reasoning\nLayers', 'pos': (6, 3.3), 'size': (2, 0.8), 'color': '#FFAB40'},
        {'name': 'Final Emb\n[B, 64]', 'pos': (9, 3), 'size': (2, 0.8), 'color': '#FF9800'},
        
        # 输出层
        {'name': 'Output Layer\nLinear(64→1)', 'pos': (1, 1.5), 'size': (3.5, 1), 'color': '#FFCDD2'},
        {'name': 'Sigmoid +\nScale [1,5]', 'pos': (6, 1.8), 'size': (2, 0.8), 'color': '#F8BBD9'},
        {'name': 'Rating\nPrediction', 'pos': (9, 1.5), 'size': (2, 0.8), 'color': '#E91E63'},
        
        # 多Teacher融合模块
        {'name': 'Multi-Teacher\nFusion', 'pos': (12, 6), 'size': (2.5, 2), 'color': '#E1BEE7'},
        {'name': 'Fisher-Guided\nWeighting', 'pos': (15, 7), 'size': (2, 1), 'color': '#CE93D8'},
        {'name': 'Knowledge\nDistillation', 'pos': (15, 5), 'size': (2, 1), 'color': '#BA68C8'},
    ]
    
    # 绘制所有组件
    for flow in flows:
        rect = FancyBboxPatch(
            flow['pos'], flow['size'][0], flow['size'][1],
            boxstyle="round,pad=0.05",
            facecolor=flow['color'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(rect)
        
        ax.text(
            flow['pos'][0] + flow['size'][0]/2,
            flow['pos'][1] + flow['size'][1]/2,
            flow['name'],
            ha='center', va='center',
            fontsize=8, fontweight='bold'
        )
    
    # 绘制数据流箭头
    flows_arrows = [
        # 主要数据流
        ((1.75, 9), (1.75, 8.3)),  # User ID -> User Emb
        ((3.75, 9), (3.75, 8.3)),  # Item ID -> Item Emb
        ((2.75, 7.5), (2.75, 7)),  # Emb -> Layer 1
        ((2.75, 6), (2.75, 5.5)),  # Layer 1 -> Layer 2
        ((2.75, 4.5), (2.75, 4)),  # Layer 2 -> Layer 3
        ((2.75, 3), (2.75, 2.5)),  # Layer 3 -> Output
        ((2.75, 1.5), (6, 1.8)),   # Output -> Sigmoid
        ((8, 1.8), (9, 1.9)),      # Sigmoid -> Final
        
        # Teacher知识流
        ((6, 8), (9, 8.6)),        # Teacher -> Attention Context
        ((6, 8), (9, 7.9)),        # Teacher -> Reasoning Context
        ((4.5, 6.5), (6, 6.8)),    # Layer 1 -> User Emb out
        ((4.5, 6), (6, 5.8)),      # Layer 1 -> Item Emb out
        ((4.5, 5), (6, 4.8)),      # Layer 2 -> Multi-Head Attention
        ((8, 4.8), (9, 4.9)),      # Attention -> Interaction Emb
        ((4.5, 3.5), (6, 3.7)),    # Layer 3 -> Reasoning Layers
        ((8, 3.3), (9, 3.4)),      # Reasoning -> Final Emb
        
        # Fusion连接
        ((8, 6), (12, 6.5)),       # 到Fusion模块
        ((12, 7), (15, 7.5)),      # 到Fisher Guide
        ((12, 5.5), (15, 5.5)),    # 到Knowledge Distillation
    ]
    
    for start, end in flows_arrows:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.08, head_length=0.08, fc='blue', ec='blue', alpha=0.6)
    
    # 添加维度标注
    ax.text(0.2, 8.5, 'Batch Size: B\nEmbedding Dim: 64\nHidden Dim: 128', 
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    ax.text(12, 8.5, 'Teacher Models:\n• SVD\n• AutoInt\n• xDeepFM\n• Llama3', 
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    
    plt.title('LayerwiseAdapter V2 - Detailed Data Flow\n(Tensor Shapes and Processing Pipeline)', 
              fontsize=14, fontweight='bold', pad=20)
    
    save_dir = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/analysis"
    plt.savefig(os.path.join(save_dir, 'detailed_flow_diagram.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 详细流程图已保存到: {save_dir}/detailed_flow_diagram.png")

def generate_architecture_report(architecture_info):
    """生成架构分析报告"""
    save_dir = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/analysis"
    report_path = os.path.join(save_dir, "architecture_analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# LayerwiseAdapter V2 架构分析报告\n\n")
        
        f.write("## 1. 模型概览\n\n")
        f.write(f"- **总参数量**: {architecture_info['total_parameters']:,}\n")
        f.write(f"- **可训练参数**: {architecture_info['trainable_parameters']:,}\n")
        f.write(f"- **模型大小**: {architecture_info['memory_usage']:.2f} MB\n")
        f.write(f"- **参数效率**: {architecture_info['trainable_parameters']/architecture_info['total_parameters']*100:.1f}%\n\n")
        
        f.write("## 2. 层级结构分析\n\n")
        for layer_name, layer_info in architecture_info['layers'].items():
            f.write(f"### {layer_name}\n")
            f.write(f"- **参数量**: {layer_info['parameters']:,}\n")
            f.write(f"- **占比**: {layer_info['parameters']/architecture_info['total_parameters']*100:.1f}%\n")
            f.write("- **组件详情**:\n")
            for component in layer_info['shape_info']:
                f.write(f"  - {component['name']}: {component['shape']} ({component['parameters']:,} 参数)\n")
            f.write("\n")
        
        f.write("## 3. 架构特点\n\n")
        f.write("### 3.1 多Teacher融合架构\n")
        f.write("- **Teacher模型**: Ensemble(SVD+AutoInt+xDeepFM) + LLM(Llama3)\n")
        f.write("- **融合策略**: Fisher引导的权重分配\n")
        f.write("- **知识蒸馏**: 分层适配器逐步精炼\n\n")
        
        f.write("### 3.2 分层适配器设计\n")
        f.write("- **Layer 1 - EmbeddingAdapter**: 多源嵌入融合\n")
        f.write("- **Layer 2 - InteractionAdapter**: 跨Teacher注意力机制\n")
        f.write("- **Layer 3 - ReasoningAdapter**: 集成推理和残差连接\n\n")
        
        f.write("### 3.3 关键创新点\n")
        f.write("- **自适应蒸馏**: 根据样本复杂度动态调整Teacher权重\n")
        f.write("- **分层知识传递**: 不同层次捕获不同类型的知识\n")
        f.write("- **多模态融合**: 传统推荐算法与大语言模型的有效结合\n")
        f.write("- **轻量级设计**: 相比Teacher模型显著减少参数量\n\n")
    
    print(f"📋 架构分析报告已保存到: {report_path}")

def main():
    """主函数"""
    print("=" * 80)
    print("LayerwiseAdapter V2 架构可视化与分析")
    print("=" * 80)
    
    # 1. 导出ONNX模型
    print("🔄 导出ONNX模型...")
    model, onnx_path = export_model_for_netron()
    
    # 2. 分析模型架构
    if model is not None:
        print("🔍 分析模型架构...")
        architecture_info = analyze_model_architecture(model)
        
        print(f"📊 模型统计:")
        print(f"   总参数量: {architecture_info['total_parameters']:,}")
        print(f"   模型大小: {architecture_info['memory_usage']:.2f} MB")
        
        # 3. 生成架构报告
        print("📝 生成架构分析报告...")
        generate_architecture_report(architecture_info)
    
    # 4. 创建架构图
    print("🎨 创建架构图...")
    create_architecture_diagram()
    
    # 5. 创建详细流程图
    print("🔀 创建详细流程图...")
    create_detailed_flow_diagram()
    
    print("\n✅ 架构可视化完成!")
    print(f"📁 结果保存在: /home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/analysis/")
    
    if onnx_path:
        print(f"🌐 Netron可视化: 访问 https://netron.app/ 并上传 {onnx_path}")

if __name__ == "__main__":
    main()
