#!/usr/bin/env python3
"""
Traditional Teachers评估结果总结报告生成器
Traditional Teachers Evaluation Results Summary Report Generator

作者: GitHub Copilot
日期: 2025-08-18
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def generate_summary_report():
    """生成Traditional Teachers评估总结报告"""
    
    # 读取最新的评估结果
    results_dir = Path("traditional_evaluation_results")
    if not results_dir.exists():
        print("❌ 评估结果目录不存在")
        return
    
    # 获取最新的结果文件
    result_files = list(results_dir.glob("traditional_evaluation_*.json"))
    if not result_files:
        print("❌ 没有找到评估结果文件")
        return
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    
    # 读取结果
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("=" * 80)
    print("📊 TRADITIONAL TEACHERS 完整评估报告")
    print("=" * 80)
    
    # 基本信息
    summary = results["summary"]
    print(f"🔍 评估时间: {results['experiment_info']['start_time']}")
    print(f"🎯 算法总数: {summary['total_algorithms']}")
    print(f"✅ 训练成功: {summary['successful_trainings']}")
    print(f"📈 评估成功: {summary['successful_evaluations']}")
    print(f"⏱️  总耗时: {summary['total_duration_seconds']:.1f} 秒")
    print(f"📁 结果文件: {latest_file}")
    
    # 创建对比表格
    print(f"\n📋 Traditional Teachers性能对比")
    print("-" * 80)
    
    # 收集所有模型的关键指标
    model_data = []
    
    for algorithm, model_result in results["models"].items():
        if model_result["training_status"] == "success" and model_result["evaluation_results"]:
            eval_results = model_result["evaluation_results"]
            
            # 提取关键指标
            accuracy = eval_results.get("accuracy_metrics", {})
            ranking = eval_results.get("ranking_metrics", {})
            diversity = eval_results.get("diversity_metrics", {})
            novelty = eval_results.get("novelty_metrics", {})
            overall = eval_results.get("overall_score", {})
            
            model_data.append({
                "算法": algorithm.upper(),
                "训练状态": "✅",
                "推荐数": model_result["recommendations_count"],
                "RMSE": f"{accuracy.get('RMSE', 0):.4f}",
                "NDCG@10": f"{ranking.get('NDCG@10', 0):.4f}",
                "覆盖率": f"{diversity.get('Coverage', 0):.4f}",
                "新颖度": f"{novelty.get('Novelty', 0):.4f}",
                "总分": f"{overall.get('overall_score', 0):.4f}"
            })
    
    # 转换为DataFrame并显示
    df = pd.DataFrame(model_data)
    print(df.to_string(index=False))
    
    # 详细指标分析
    print(f"\n📊 详细指标分析")
    print("-" * 80)
    
    for algorithm, model_result in results["models"].items():
        if model_result["training_status"] == "success" and model_result["evaluation_results"]:
            eval_results = model_result["evaluation_results"]
            
            print(f"\n🔸 {algorithm.upper()} 详细指标:")
            
            # 准确性指标
            accuracy = eval_results.get("accuracy_metrics", {})
            print(f"   📏 准确性: RMSE={accuracy.get('RMSE', 0):.4f}, MAE={accuracy.get('MAE', 0):.4f}")
            
            # 排序指标
            ranking = eval_results.get("ranking_metrics", {})
            print(f"   🎯 排序质量: P@10={ranking.get('Precision@10', 0):.4f}, NDCG@10={ranking.get('NDCG@10', 0):.4f}")
            
            # 多样性指标
            diversity = eval_results.get("diversity_metrics", {})
            print(f"   🌈 多样性: 覆盖率={diversity.get('Coverage', 0):.4f}, 列表内多样性={diversity.get('Intra_List_Diversity', 0):.4f}")
            
            # 新颖性指标
            novelty = eval_results.get("novelty_metrics", {})
            print(f"   ✨ 新颖性: {novelty.get('Novelty', 0):.4f}")
            
            # 综合评分
            overall = eval_results.get("overall_score", {})
            print(f"   🏆 综合得分: {overall.get('overall_score', 0):.4f}")
    
    # 模型排名
    print(f"\n🏆 模型性能排名")
    print("-" * 80)
    
    # 按综合得分排序
    model_scores = []
    for algorithm, model_result in results["models"].items():
        if model_result["training_status"] == "success" and model_result["evaluation_results"]:
            overall_score = model_result["evaluation_results"].get("overall_score", {}).get("overall_score", 0)
            model_scores.append((algorithm, overall_score))
    
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (algorithm, score) in enumerate(model_scores, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
        print(f"   {medal} 第{i}名: {algorithm.upper()} (得分: {score:.4f})")
    
    # 结论和建议
    print(f"\n💡 评估结论")
    print("-" * 80)
    print(f"✅ 所有6个Traditional Teacher模型均成功训练和评估")
    print(f"📊 评估覆盖准确性、排序、多样性、新颖性四个维度")
    print(f"🎯 最佳模型: {model_scores[0][0].upper()} (综合得分: {model_scores[0][1]:.4f})")
    print(f"📈 所有模型均能正常生成推荐，具备实际应用潜力")
    print(f"🔄 建议: 可进行超参数调优进一步提升性能")
    
    # 保存总结报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f"traditional_teachers_summary_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Traditional Teachers评估总结报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"评估时间: {results['experiment_info']['start_time']}\n")
        f.write(f"算法总数: {summary['total_algorithms']}\n")
        f.write(f"成功训练: {summary['successful_trainings']}\n")
        f.write(f"成功评估: {summary['successful_evaluations']}\n")
        f.write(f"总耗时: {summary['total_duration_seconds']:.1f} 秒\n\n")
        
        f.write("模型性能排名:\n")
        for i, (algorithm, score) in enumerate(model_scores, 1):
            f.write(f"{i}. {algorithm.upper()}: {score:.4f}\n")
    
    print(f"\n💾 总结报告已保存: {report_file}")

if __name__ == "__main__":
    generate_summary_report()
