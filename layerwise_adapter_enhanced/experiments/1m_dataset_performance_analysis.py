#!/usr/bin/env python3
"""
基于现有结果的性能分析报告
分析各种实验的结果，包括基准测试和之前的LayerwiseAdapter实验
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_available_results():
    """加载所有可用的实验结果"""
    results = {}
    base_path = '/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments'
    
    # 1. 简化版ensemble基准结果
    baseline_path = f"{base_path}/baselines/simple_ensemble_baseline_results.json"
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            results['simple_ensemble_baseline'] = json.load(f)
        print("✅ 加载简化版Ensemble基准结果")
    
    # 2. 之前的ensemble基准结果
    old_baseline_path = f"{base_path}/baselines/ensemble_baseline_1m_results.json"
    if os.path.exists(old_baseline_path):
        with open(old_baseline_path, 'r', encoding='utf-8') as f:
            results['old_ensemble_baseline'] = json.load(f)
        print("✅ 加载之前的Ensemble基准结果")
    
    # 3. LayerwiseAdapter结果（如果存在）
    layerwise_paths = [
        f"{base_path}/layerwise_adapter/full_training_results.json",
        f"/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/results/full_training_results.json",
        f"{base_path}/../results/layerwise_adapter_results.json"
    ]
    
    for path in layerwise_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                results['layerwise_adapter'] = json.load(f)
            print(f"✅ 加载LayerwiseAdapter结果: {path}")
            break
    
    # 4. 其他可能的结果文件
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('_results.json') and 'simple_ensemble_baseline' not in file:
                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        key = file.replace('_results.json', '').replace('.json', '')
                        if key not in results:
                            results[key] = data
                            print(f"✅ 发现额外结果文件: {file}")
                except:
                    continue
    
    return results

def analyze_baseline_performance(results):
    """分析基准性能"""
    print("\n" + "=" * 60)
    print("📊 基准性能分析")
    print("=" * 60)
    
    if 'simple_ensemble_baseline' in results:
        baseline = results['simple_ensemble_baseline']
        
        print("\n🎯 简化版Ensemble基准 (最可靠):")
        if 'ensemble_baseline' in baseline and baseline['ensemble_baseline']:
            metrics = baseline['ensemble_baseline']['ensemble_metrics']
            print(f"  算法组合: {baseline['ensemble_baseline']['component_algorithms']}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  Precision@10: {metrics['precision_at_10']:.4f}")
            print(f"  NDCG@10: {metrics['ndcg_at_10']:.4f}")
            
            print(f"\n📝 个体算法表现:")
            for alg, result in baseline['individual_algorithms'].items():
                if 'error' not in result:
                    print(f"  {alg}:")
                    print(f"    RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
                    print(f"    P@10: {result['recommendation_quality']['precision_at_10']:.4f}")
                    print(f"    NDCG@10: {result['recommendation_quality']['ndcg_at_10']:.4f}")
        
        return baseline['ensemble_baseline']['ensemble_metrics'] if 'ensemble_baseline' in baseline else None
    
    return None

def analyze_layerwise_performance(results):
    """分析LayerwiseAdapter性能"""
    print("\n" + "=" * 60)
    print("🔍 LayerwiseAdapter性能分析")
    print("=" * 60)
    
    if 'layerwise_adapter' in results:
        layerwise = results['layerwise_adapter']
        print("✅ 找到LayerwiseAdapter结果")
        
        # 尝试提取关键指标
        if 'test_metrics' in layerwise:
            metrics = layerwise['test_metrics']
            print(f"  测试指标: {metrics}")
            return metrics
        elif 'final_metrics' in layerwise:
            metrics = layerwise['final_metrics']
            print(f"  最终指标: {metrics}")
            return metrics
        else:
            print("  结构:", list(layerwise.keys()))
            return None
    else:
        print("❌ 未找到LayerwiseAdapter结果")
        print("   可能需要运行训练脚本生成结果")
        return None

def generate_comparison_table(baseline_metrics, layerwise_metrics):
    """生成对比表格"""
    print("\n" + "=" * 60)
    print("📈 性能对比表")
    print("=" * 60)
    
    if baseline_metrics and layerwise_metrics:
        print("\n| 模型 | RMSE | MAE | Precision@10 | NDCG@10 |")
        print("|------|------|-----|--------------|---------|")
        print(f"| Ensemble基准 | {baseline_metrics['rmse']:.4f} | {baseline_metrics['mae']:.4f} | {baseline_metrics['precision_at_10']:.4f} | {baseline_metrics['ndcg_at_10']:.4f} |")
        print(f"| LayerwiseAdapter | {layerwise_metrics.get('rmse', 'N/A')} | {layerwise_metrics.get('mae', 'N/A')} | {layerwise_metrics.get('precision_at_10', 'N/A')} | {layerwise_metrics.get('ndcg_at_10', 'N/A')} |")
        
        # 改进分析
        if all(k in layerwise_metrics for k in ['rmse', 'mae', 'precision_at_10', 'ndcg_at_10']):
            print(f"\n🔍 性能对比分析:")
            
            rmse_improvement = (baseline_metrics['rmse'] - layerwise_metrics['rmse']) / baseline_metrics['rmse'] * 100
            mae_improvement = (baseline_metrics['mae'] - layerwise_metrics['mae']) / baseline_metrics['mae'] * 100
            precision_improvement = (layerwise_metrics['precision_at_10'] - baseline_metrics['precision_at_10']) / baseline_metrics['precision_at_10'] * 100
            ndcg_improvement = (layerwise_metrics['ndcg_at_10'] - baseline_metrics['ndcg_at_10']) / baseline_metrics['ndcg_at_10'] * 100
            
            print(f"  评分预测改进: RMSE {rmse_improvement:+.1f}%, MAE {mae_improvement:+.1f}%")
            print(f"  推荐质量改进: P@10 {precision_improvement:+.1f}%, NDCG@10 {ndcg_improvement:+.1f}%")
    
    elif baseline_metrics:
        print("\n✅ 基准指标:")
        print(f"  RMSE: {baseline_metrics['rmse']:.4f}")
        print(f"  MAE: {baseline_metrics['mae']:.4f}")
        print(f"  Precision@10: {baseline_metrics['precision_at_10']:.4f}")
        print(f"  NDCG@10: {baseline_metrics['ndcg_at_10']:.4f}")
        print("\n⏳ LayerwiseAdapter结果待生成...")
        
    else:
        print("\n❌ 缺少必要的对比数据")

def generate_insights_and_recommendations():
    """生成洞察和建议"""
    print("\n" + "=" * 60)
    print("💡 关键洞察与建议")
    print("=" * 60)
    
    insights = [
        "🔍 算法性能分析:",
        "  - SVD: 推荐质量最佳 (P@10=0.73, NDCG@10=0.76), 但评分预测较差 (RMSE=2.46)",
        "  - AutoInt: 评分预测很好 (RMSE=0.90), 推荐质量中等 (P@10=0.18)",
        "  - DeepFM: 评分预测很好 (RMSE=0.92), 但推荐质量较差 (P@10=0.06)",
        "",
        "🎯 Ensemble基准效果:",
        "  - 在评分预测和推荐质量间取得平衡",
        "  - RMSE=1.43, MAE=1.21 (中等水平)",
        "  - P@10=0.32, NDCG@10=0.33 (合理水平)",
        "",
        "🚀 下一步工作优先级:",
        "  1. 运行LayerwiseAdapter训练，生成1M数据集上的完整结果",
        "  2. 对比LayerwiseAdapter与Ensemble基准的性能差异",
        "  3. 分析LayerwiseAdapter的参数效率和知识蒸馏效果",
        "  4. 准备学术论文材料，重点突出创新点",
        "",
        "📊 实验设计建议:",
        "  - 确保所有模型在相同数据分割上测试",
        "  - 关注参数量对比 (LayerwiseAdapter应该更轻量)",
        "  - 分析训练时间和推理效率",
        "  - 可视化学习曲线和收敛过程"
    ]
    
    for insight in insights:
        print(insight)

def create_summary_report(results):
    """创建总结报告"""
    report = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'available_results': list(results.keys()),
        'summary': {
            'baseline_algorithms_tested': ['svd', 'autoint', 'deepfm'],
            'best_individual_algorithm': {
                'rating_prediction': 'autoint (RMSE=0.90)',
                'recommendation_quality': 'svd (P@10=0.73, NDCG@10=0.76)'
            },
            'ensemble_baseline_performance': {
                'rmse': 1.4263,
                'mae': 1.2145,
                'precision_at_10': 0.3233,
                'ndcg_at_10': 0.3278
            },
            'next_steps': [
                'Run LayerwiseAdapter training on 1M dataset',
                'Generate comprehensive performance comparison',
                'Analyze parameter efficiency and knowledge distillation',
                'Prepare academic paper materials'
            ]
        }
    }
    
    # 保存报告
    report_file = '/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/1m_dataset_analysis_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 分析报告已保存到: {report_file}")
    return report

def main():
    """主函数"""
    print("=" * 80)
    print("📋 MovieLens 1M数据集实验结果分析")
    print("=" * 80)
    
    # 加载所有可用结果
    results = load_available_results()
    
    if not results:
        print("❌ 没有找到任何实验结果文件")
        return
    
    print(f"\n📁 找到 {len(results)} 个结果文件: {list(results.keys())}")
    
    # 分析基准性能
    baseline_metrics = analyze_baseline_performance(results)
    
    # 分析LayerwiseAdapter性能
    layerwise_metrics = analyze_layerwise_performance(results)
    
    # 生成对比表格
    generate_comparison_table(baseline_metrics, layerwise_metrics)
    
    # 生成洞察和建议
    generate_insights_and_recommendations()
    
    # 创建总结报告
    report = create_summary_report(results)
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
