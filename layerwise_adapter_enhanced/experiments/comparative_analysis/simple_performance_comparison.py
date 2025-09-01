#!/usr/bin/env python3
"""
简化版性能对比分析

对比LayerwiseAdapter和Ensemble基准的核心性能指标
"""

import json
import numpy as np
from typing import Dict, Any

def load_layerwise_results(file_path: str) -> Dict[str, Any]:
    """加载LayerwiseAdapter结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return {
        'model_name': 'LayerwiseAdapter Enhanced',
        'rmse': data['test_results']['rmse'],
        'mae': data['test_results']['mae'],
        'accuracy': data['test_results']['accuracy'],
        'num_samples': data['test_results']['num_samples'],
        'num_parameters': data.get('model_parameters', 'N/A'),
        'training_time': sum(data['training_history']['epoch_times']) if 'training_history' in data else 'N/A'
    }

def load_ensemble_results(file_path: str) -> Dict[str, Any]:
    """加载Ensemble基准结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ensemble_data = data['ensemble_baseline']
    return {
        'model_name': 'Ensemble Baseline',
        'rmse': ensemble_data['ensemble_metrics']['rmse'],
        'mae': ensemble_data['ensemble_metrics']['mae'],
        'precision_at_10': ensemble_data['ensemble_metrics']['precision_at_10'],
        'ndcg_at_10': ensemble_data['ensemble_metrics']['ndcg_at_10'],
        'individual_results': ensemble_data['individual_results']
    }

def create_comparison_report(layerwise_results: Dict, ensemble_results: Dict):
    """创建对比报告"""
    
    print("=" * 80)
    print("🎯 LayerwiseAdapter vs Ensemble Baseline 性能对比")
    print("=" * 80)
    
    # 核心指标对比
    print("\n📊 核心评分预测指标:")
    print("-" * 50)
    print(f"{'指标':<20} {'LayerwiseAdapter':<20} {'Ensemble基准':<20} {'改进':<15}")
    print("-" * 50)
    
    # RMSE对比
    layerwise_rmse = layerwise_results['rmse']
    ensemble_rmse = ensemble_results['rmse']
    rmse_improvement = ((ensemble_rmse - layerwise_rmse) / ensemble_rmse) * 100
    print(f"{'RMSE':<20} {layerwise_rmse:<20.4f} {ensemble_rmse:<20.4f} {rmse_improvement:>+.2f}%")
    
    # MAE对比
    layerwise_mae = layerwise_results['mae']
    ensemble_mae = ensemble_results['mae']
    mae_improvement = ((ensemble_mae - layerwise_mae) / ensemble_mae) * 100
    print(f"{'MAE':<20} {layerwise_mae:<20.4f} {ensemble_mae:<20.4f} {mae_improvement:>+.2f}%")
    
    # 准确率
    if 'accuracy' in layerwise_results:
        print(f"{'准确率(±0.5)':<20} {layerwise_results['accuracy']*100:<20.2f}% {'N/A':<20} {'N/A':<15}")
    
    print("-" * 75)
    
    # 推荐质量指标（仅Ensemble有）
    print("\n🎯 推荐质量指标 (仅Ensemble基准):")
    print("-" * 50)
    print(f"Precision@10: {ensemble_results['precision_at_10']:.4f}")
    print(f"NDCG@10: {ensemble_results['ndcg_at_10']:.4f}")
    
    # 个体算法表现
    print("\n🔍 Ensemble中个体算法表现:")
    print("-" * 50)
    for algo_name, results in ensemble_results['individual_results'].items():
        print(f"{algo_name.upper()}:")
        print(f"  RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}")
        if 'recommendation_quality' in results:
            rq = results['recommendation_quality']
            print(f"  P@10: {rq['precision_at_10']:.4f}, NDCG@10: {rq['ndcg_at_10']:.4f}")
        print()
    
    # 模型复杂度对比
    print("📈 模型复杂度对比:")
    print("-" * 50)
    if layerwise_results['num_parameters'] != 'N/A':
        print(f"LayerwiseAdapter参数量: {layerwise_results['num_parameters']:,}")
    
    # 计算ensemble基准的参数量（估算）
    print("Ensemble基准: 约722,950参数 (SVD + AutoInt + DeepFM)")
    
    # 性能分析
    print("\n🎭 性能分析:")
    print("-" * 50)
    if rmse_improvement > 0:
        print(f"✅ LayerwiseAdapter在RMSE上表现更好，改进了{rmse_improvement:.2f}%")
    else:
        print(f"❌ LayerwiseAdapter在RMSE上表现较差，下降了{abs(rmse_improvement):.2f}%")
    
    if mae_improvement > 0:
        print(f"✅ LayerwiseAdapter在MAE上表现更好，改进了{mae_improvement:.2f}%")
    else:
        print(f"❌ LayerwiseAdapter在MAE上表现较差，下降了{abs(mae_improvement):.2f}%")
    
    # 最佳个体算法对比
    best_individual_rmse = min(results['rmse'] for results in ensemble_results['individual_results'].values())
    best_individual_name = min(ensemble_results['individual_results'].items(), key=lambda x: x[1]['rmse'])[0]
    
    individual_improvement = ((best_individual_rmse - layerwise_rmse) / best_individual_rmse) * 100
    
    print(f"\n🏆 与最佳个体算法({best_individual_name.upper()})对比:")
    print(f"   最佳个体RMSE: {best_individual_rmse:.4f}")
    print(f"   LayerwiseAdapter RMSE: {layerwise_rmse:.4f}")
    if individual_improvement > 0:
        print(f"   ✅ 改进了{individual_improvement:.2f}%")
    else:
        print(f"   ❌ 下降了{abs(individual_improvement):.2f}%")
    
    # 结论
    print("\n📝 结论:")
    print("-" * 50)
    if rmse_improvement > 0 and mae_improvement > 0:
        print("🎉 LayerwiseAdapter在评分预测任务上优于Ensemble基准")
    elif individual_improvement > 0:
        print("✨ LayerwiseAdapter优于最佳个体算法，证明了多Teacher融合的有效性")
    else:
        print("⚠️ LayerwiseAdapter需要进一步优化，当前表现未达到预期")
        print("建议：")
        print("  - 调整多Teacher融合权重")
        print("  - 增加训练轮数")
        print("  - 优化Fisher引导策略")
        print("  - 加入更多正则化")

def main():
    """主函数"""
    print("🔍 开始性能对比分析...")
    
    # 文件路径
    layerwise_file = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/layerwise_adapter/full_training_results.json"
    ensemble_file = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/baselines/simple_ensemble_baseline_results.json"
    
    try:
        # 加载结果
        layerwise_results = load_layerwise_results(layerwise_file)
        ensemble_results = load_ensemble_results(ensemble_file)
        
        # 生成对比报告
        create_comparison_report(layerwise_results, ensemble_results)
        
        print("\n" + "=" * 80)
        print("✅ 对比分析完成")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

if __name__ == "__main__":
    main()
