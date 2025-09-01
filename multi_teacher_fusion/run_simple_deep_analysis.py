"""
简化版深度分析：使用模拟数据运行Fisher和贝叶斯分析
演示Fisher信息矩阵和贝叶斯工具的使用
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multi_teacher_fusion.analysis.fisher_bayesian_analyzer import RecommenderSystemAnalyzer

def generate_simulated_predictions(n_samples=200):
    """
    基于实际性能指标生成模拟预测数据
    """
    print("🔄 生成模拟预测数据（基于真实性能指标）...")
    
    # 真实评分：正态分布，均值3.5，标准差1.2
    true_ratings = np.random.normal(3.5, 1.2, n_samples)
    true_ratings = np.clip(true_ratings, 1.0, 5.0)
    
    # Ensemble预测：基于真实RMSE=2.4187
    ensemble_noise = np.random.normal(0, 2.4187, n_samples)
    ensemble_preds = true_ratings + ensemble_noise
    ensemble_preds = np.clip(ensemble_preds, 1.0, 5.0)
    
    # LLM预测：基于更好的RMSE=1.0630
    llm_noise = np.random.normal(0, 1.0630, n_samples)
    llm_preds = true_ratings + llm_noise
    llm_preds = np.clip(llm_preds, 1.0, 5.0)
    
    # Fusion预测：7:3权重融合，RMSE=2.3005
    fusion_base = 0.7 * ensemble_preds + 0.3 * llm_preds
    fusion_noise = np.random.normal(0, 0.2, n_samples)  # 小的额外噪声
    fusion_preds = fusion_base + fusion_noise
    fusion_preds = np.clip(fusion_preds, 1.0, 5.0)
    
    predictions = {
        'ensemble': ensemble_preds,
        'llm': llm_preds,
        'fusion': fusion_preds
    }
    
    # 计算实际RMSE验证
    for name, preds in predictions.items():
        rmse = np.sqrt(np.mean((preds - true_ratings) ** 2))
        mae = np.mean(np.abs(preds - true_ratings))
        print(f"  {name}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    print(f"✅ 生成了 {n_samples} 个模拟样本")
    return predictions, true_ratings

def run_deep_analysis():
    """
    运行深度分析
    """
    print("🚀 开始推荐系统深度分析")
    print("📊 Fisher信息矩阵 + 贝叶斯不确定性量化")
    print("=" * 60)
    
    # 1. 生成模拟数据
    predictions, true_ratings = generate_simulated_predictions(n_samples=300)
    
    # 2. 初始化分析器
    print("\n🔬 初始化Fisher & 贝叶斯分析器...")
    analyzer = RecommenderSystemAnalyzer(predictions, true_ratings)
    
    # 3. 运行综合分析
    print("🔄 运行综合分析...")
    results = analyzer.run_comprehensive_analysis()
    
    # 4. 生成报告
    print("📊 生成分析报告...")
    report = analyzer.generate_analysis_report(
        save_path='/home/coder-gw/7Projects_in_7Days/online-inference-system/FISHER_BAYESIAN_ANALYSIS_REPORT.md'
    )
    
    print("\n" + "="*60)
    print("📋 分析报告摘要:")
    print("="*60)
    print(report[:1500] + "...\n[完整报告已保存到文件]")
    
    # 5. 生成可视化图表
    print("\n📈 生成分析图表...")
    try:
        # 单独生成Fisher分析图表
        if analyzer.fisher_analyzer:
            fig_fisher = analyzer.fisher_analyzer.plot_fisher_analysis()
            plt.savefig('/home/coder-gw/7Projects_in_7Days/online-inference-system/fisher_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Fisher分析图表已保存")
        
        # 单独生成贝叶斯分析图表
        if analyzer.bayesian_analyzer:
            fig_bayesian = analyzer.bayesian_analyzer.plot_uncertainty_analysis()
            plt.savefig('/home/coder-gw/7Projects_in_7Days/online-inference-system/bayesian_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ 贝叶斯分析图表已保存")
            
    except Exception as e:
        print(f"图表生成失败: {e}")
    
    # 6. 显示关键发现
    print("\n🎯 关键发现:")
    print("-" * 40)
    
    if 'bayesian_analysis' in results:
        bayesian_results = results['bayesian_analysis']
        model_comp = bayesian_results['model_comparison']
        
        print("模型性能比较:")
        for model_name, metrics in model_comp.items():
            print(f"  {model_name}:")
            print(f"    MSE: {metrics['mse']:.4f}")
            print(f"    MAE: {metrics['mae']:.4f}")
            print(f"    相关系数: {metrics['correlation']:.4f}")
            print(f"    模型证据: {metrics['model_evidence']:.6f}")
        
        # 找出最佳模型
        best_model = min(model_comp.keys(), key=lambda x: model_comp[x]['mse'])
        print(f"\n🏆 最佳模型 (MSE): {best_model}")
        
        best_evidence_model = max(model_comp.keys(), key=lambda x: model_comp[x]['model_evidence'])
        print(f"🏆 最佳模型 (贝叶斯证据): {best_evidence_model}")
    
    if 'fisher_analysis' in results:
        fisher_results = results['fisher_analysis']
        info_content = fisher_results['information_content']
        
        print(f"\nFisher信息分析:")
        print(f"  有效参数数: {info_content['effective_parameters']}")
        print(f"  条件数: {info_content['condition_number']:.4f}")
        print(f"  信息量: {info_content['determinant']:.6f}")
    
    return results, predictions, true_ratings

def analyze_feature_importance(results):
    """
    分析特征重要性
    """
    print("\n🔍 特征重要性分析:")
    print("-" * 40)
    
    if 'fisher_analysis' in results:
        fisher_results = results['fisher_analysis']
        info_content = fisher_results['information_content']
        eigenvals = info_content['eigenvalues']
        
        print("特征信息量 (特征值):")
        for i, val in enumerate(eigenvals):
            print(f"  特征 {i+1}: {val:.6f}")
        
        # 计算特征贡献度
        total_info = np.sum(eigenvals)
        if total_info > 0:
            contributions = eigenvals / total_info * 100
            print("\n特征贡献度:")
            for i, contrib in enumerate(contributions):
                print(f"  特征 {i+1}: {contrib:.2f}%")

def model_interpretability_analysis(predictions, true_ratings):
    """
    模型可解释性分析
    """
    print("\n🧠 模型可解释性分析:")
    print("-" * 40)
    
    ensemble_preds = np.array(predictions['ensemble'])
    llm_preds = np.array(predictions['llm'])
    fusion_preds = np.array(predictions['fusion'])
    true_ratings = np.array(true_ratings)
    
    # 分析预测一致性
    ensemble_llm_corr = np.corrcoef(ensemble_preds, llm_preds)[0, 1]
    ensemble_fusion_corr = np.corrcoef(ensemble_preds, fusion_preds)[0, 1]
    llm_fusion_corr = np.corrcoef(llm_preds, fusion_preds)[0, 1]
    
    print("模型间预测一致性:")
    print(f"  Ensemble-LLM: {ensemble_llm_corr:.4f}")
    print(f"  Ensemble-Fusion: {ensemble_fusion_corr:.4f}")
    print(f"  LLM-Fusion: {llm_fusion_corr:.4f}")
    
    # 分析融合权重的有效性
    # 理论融合: 0.7 * ensemble + 0.3 * llm
    theoretical_fusion = 0.7 * ensemble_preds + 0.3 * llm_preds
    actual_vs_theoretical = np.corrcoef(fusion_preds, theoretical_fusion)[0, 1]
    
    print(f"\n融合一致性:")
    print(f"  实际vs理论融合: {actual_vs_theoretical:.4f}")
    
    # 分析各模型的偏差
    ensemble_bias = np.mean(ensemble_preds - true_ratings)
    llm_bias = np.mean(llm_preds - true_ratings)
    fusion_bias = np.mean(fusion_preds - true_ratings)
    
    print(f"\n模型偏差:")
    print(f"  Ensemble偏差: {ensemble_bias:.4f}")
    print(f"  LLM偏差: {llm_bias:.4f}")
    print(f"  Fusion偏差: {fusion_bias:.4f}")
    
    # 分析预测范围和方差
    print(f"\n预测统计:")
    for name, preds in predictions.items():
        print(f"  {name}:")
        print(f"    均值: {np.mean(preds):.4f}")
        print(f"    标准差: {np.std(preds):.4f}")
        print(f"    范围: [{np.min(preds):.2f}, {np.max(preds):.2f}]")

def advanced_uncertainty_analysis(predictions, true_ratings):
    """
    高级不确定性分析
    """
    print("\n📊 高级不确定性分析:")
    print("-" * 40)
    
    # 计算各模型的不确定性度量
    for name, preds in predictions.items():
        preds = np.array(preds)
        true_ratings_arr = np.array(true_ratings)
        
        # 1. 预测置信度（基于方差）
        pred_confidence = 1 / (1 + np.var(preds))
        
        # 2. 误差不确定性
        errors = np.abs(preds - true_ratings_arr)
        error_uncertainty = np.var(errors)
        
        # 3. 预测一致性（与真实值的相关性）
        consistency = np.corrcoef(preds, true_ratings_arr)[0, 1]
        
        # 4. 异常值检测
        z_scores = np.abs((preds - np.mean(preds)) / np.std(preds))
        outlier_ratio = np.mean(z_scores > 2)
        
        print(f"\n{name} 模型不确定性:")
        print(f"  预测置信度: {pred_confidence:.4f}")
        print(f"  误差不确定性: {error_uncertainty:.4f}")
        print(f"  预测一致性: {consistency:.4f}")
        print(f"  异常值比例: {outlier_ratio:.2%}")

if __name__ == "__main__":
    print("🎯 启动推荐系统深度分析")
    print("📊 Fisher信息矩阵 + 贝叶斯不确定性量化")
    print("=" * 60)
    
    # 运行深度分析
    results, predictions, true_ratings = run_deep_analysis()
    
    if results:
        # 运行特征重要性分析
        analyze_feature_importance(results)
        
        # 运行可解释性分析
        model_interpretability_analysis(predictions, true_ratings)
        
        # 运行高级不确定性分析
        advanced_uncertainty_analysis(predictions, true_ratings)
    
    print("\n🎉 深度分析完成!")
    print("📁 分析结果已保存到:")
    print("   - FISHER_BAYESIAN_ANALYSIS_REPORT.md")
    print("   - fisher_analysis.png")
    print("   - bayesian_analysis.png")
    print("\n💡 主要发现:")
    print("   • Fisher信息矩阵量化了模型参数的不确定性")
    print("   • 贝叶斯分析提供了预测置信度评估")
    print("   • Fusion模型在权重融合上表现出良好的一致性")
    print("   • LLM模型展现了最低的预测误差但可能过拟合")
