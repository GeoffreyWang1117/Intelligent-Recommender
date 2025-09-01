"""
深度分析推荐系统内部机制
使用Fisher信息矩阵和贝叶斯工具分析多Teacher融合系统
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multi_teacher_fusion.analysis.fisher_bayesian_analyzer import RecommenderSystemAnalyzer
from multi_teacher_fusion.utils.data_processor import MovieLensProcessor
from multi_teacher_fusion.teachers.ensemble_teacher import EnsembleTeacher
from multi_teacher_fusion.teachers.fusion_teacher import MultiTeacherFusion
from multi_teacher_fusion.config.teacher_config import EnsembleConfig, LLMConfig, FusionConfig

def load_test_data(sample_size=200):
    """
    加载测试数据
    """
    print("🔄 加载MovieLens测试数据...")
    
    # 使用现有的MovieLensProcessor
    processor = MovieLensProcessor('/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens')
    train_df, test_df, movies_df, stats = processor.load_and_process()
    
    # 取样本进行深度分析
    if len(test_df) > sample_size:
        test_sample = test_df.sample(n=sample_size, random_state=42)
    else:
        test_sample = test_df
    
    print(f"✅ 加载了 {len(test_sample)} 条测试样本")
    return train_df, test_sample, movies_df

def generate_predictions(train_data, test_data, movies_df):
    """
    生成各模型的预测结果
    """
    print("🔄 生成模型预测...")
    
    # 创建配置
    ensemble_config = EnsembleConfig()
    llm_config = LLMConfig()
    fusion_config = FusionConfig()
    
    # 1. 初始化Ensemble Teacher
    print("初始化Ensemble Teacher...")
    ensemble_teacher = EnsembleTeacher(ensemble_config.model_dump())
    
    # 2. 初始化Fusion Teacher
    print("初始化Fusion Teacher...")
    fusion_teacher = MultiTeacherFusion(
        ensemble_config=ensemble_config,
        llm_config=llm_config,
        fusion_config=fusion_config,
        data_path='/home/coder-gw/7Projects_in_7Days/online-inference-system/data/movielens'
    )
    
    predictions = {
        'ensemble': [],
        'llm': [],
        'fusion': []
    }
    
    true_ratings = []
    
    print(f"开始预测 {len(test_data)} 个样本...")
    
    for idx, row in test_data.iterrows():
        user_id = row['user_id']
        item_id = row['item_id'] 
        true_rating = row['rating']
        
        try:
            # Ensemble预测
            ensemble_pred = ensemble_teacher.predict(user_id, item_id)
            predictions['ensemble'].append(ensemble_pred)
            
            # LLM预测（模拟）
            llm_pred = np.random.normal(3.5, 0.8)  # 临时模拟LLM预测
            llm_pred = max(1.0, min(5.0, llm_pred))  # 限制在1-5范围
            predictions['llm'].append(llm_pred)
            
            # Fusion预测（简单加权）
            fusion_pred = 0.7 * ensemble_pred + 0.3 * llm_pred
            predictions['fusion'].append(fusion_pred)
            
            true_ratings.append(true_rating)
            
            if len(predictions['ensemble']) % 50 == 0:
                print(f"完成 {len(predictions['ensemble'])} 个预测...")
                
        except Exception as e:
            print(f"预测失败 (user={user_id}, item={item_id}): {e}")
            continue
    
    print(f"✅ 成功生成 {len(predictions['ensemble'])} 个预测")
    return predictions, true_ratings

def run_deep_analysis():
    """
    运行深度分析
    """
    print("🚀 开始推荐系统深度分析")
    print("=" * 60)
    
    # 1. 加载数据
    train_data, test_data = load_test_data(sample_size=100)  # 先用小样本测试
    
    # 2. 生成预测
    predictions, true_ratings = generate_predictions(train_data, test_data)
    
    if len(true_ratings) == 0:
        print("❌ 没有成功的预测，无法进行分析")
        return
    
    # 3. 初始化分析器
    print("\n🔬 初始化Fisher & 贝叶斯分析器...")
    analyzer = RecommenderSystemAnalyzer(predictions, true_ratings)
    
    # 4. 运行综合分析
    print("🔄 运行综合分析...")
    results = analyzer.run_comprehensive_analysis()
    
    # 5. 生成报告
    print("📊 生成分析报告...")
    report = analyzer.generate_analysis_report(
        save_path='/home/coder-gw/7Projects_in_7Days/online-inference-system/FISHER_BAYESIAN_ANALYSIS_REPORT.md'
    )
    
    print("\n" + "="*60)
    print("📋 分析报告摘要:")
    print("="*60)
    print(report[:1500] + "...\n[完整报告已保存到文件]")
    
    # 6. 生成可视化图表
    print("📈 生成分析图表...")
    try:
        fig = analyzer.plot_comprehensive_analysis(
            save_path='/home/coder-gw/7Projects_in_7Days/online-inference-system/fisher_bayesian_analysis.png'
        )
        print("✅ 图表已保存")
    except Exception as e:
        print(f"图表生成失败: {e}")
    
    # 7. 显示关键发现
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
    
    return results

def analyze_feature_importance(results):
    """
    分析特征重要性
    """
    print("\n🔍 特征重要性分析:")
    print("-" * 40)
    
    if 'fisher_analysis' in results:
        fisher_results = results['fisher_analysis']
        eigenvals = fisher_results['information_content']['eigenvalues']
        
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

if __name__ == "__main__":
    print("🎯 启动推荐系统深度分析")
    print("📊 Fisher信息矩阵 + 贝叶斯不确定性量化")
    print("=" * 60)
    
    # 运行深度分析
    results = run_deep_analysis()
    
    if results:
        # 运行特征重要性分析
        analyze_feature_importance(results)
        
        # 运行可解释性分析
        predictions = results.get('predictions', {})
        true_ratings = results.get('true_ratings', [])
        
        if predictions and true_ratings:
            model_interpretability_analysis(predictions, true_ratings)
    
    print("\n🎉 深度分析完成!")
    print("📁 报告已保存到:")
    print("   - FISHER_BAYESIAN_ANALYSIS_REPORT.md")
    print("   - fisher_bayesian_analysis.png")
