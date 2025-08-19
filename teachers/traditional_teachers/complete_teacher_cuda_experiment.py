#!/usr/bin/env python3
"""
完整的6个Teacher模型CUDA实验 - 最终修复版本
确保所有模型都能正确生成推荐并参与一致性分析
包含最新的错误修复和接口统一
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List
from collections import defaultdict
import traceback

# CUDA优化设置
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from models import create_recommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_jaccard_similarity(set1: set, set2: set) -> float:
    """计算Jaccard相似度"""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def check_cuda_environment():
    """检查CUDA环境"""
    print("🔍 检查CUDA环境...")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   设备 {i}: {torch.cuda.get_device_name(i)}")
            print(f"   显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        torch.cuda.empty_cache()
        
        try:
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.mm(test_tensor, test_tensor.t())
            del test_tensor, result
            torch.cuda.empty_cache()
            print("   ✅ CUDA基本测试通过")
        except Exception as e:
            print(f"   ❌ CUDA测试失败: {e}")
            return False
    else:
        print("   ⚠️ CUDA不可用，将使用CPU模式")
        return False
    
    return True


def test_model_recommendation(model, model_name: str, test_user_ids: List[int]) -> Dict:
    """测试单个模型的推荐功能"""
    print(f"🔍 测试 {model_name} 的推荐功能...")
    
    success_count = 0
    error_count = 0
    sample_recommendations = []
    errors = []
    
    for user_id in test_user_ids[:10]:  # 先测试10个用户
        try:
            recs = model.get_user_recommendations(user_id, top_k=5)
            if recs and len(recs) > 0:
                success_count += 1
                if len(sample_recommendations) < 3:
                    sample_recommendations.append((user_id, recs[:2]))
            else:
                error_count += 1
                errors.append(f"用户{user_id}: 空推荐")
        except Exception as e:
            error_count += 1
            error_msg = str(e)
            errors.append(f"用户{user_id}: {error_msg[:50]}")
            if len(errors) <= 3:  # 只记录前3个错误
                print(f"   ⚠️ 用户{user_id}推荐失败: {error_msg[:50]}")
    
    success_rate = success_count / len(test_user_ids[:10])
    
    result = {
        'success_rate': success_rate,
        'success_count': success_count,
        'error_count': error_count,
        'sample_recommendations': sample_recommendations,
        'errors': errors,
        'working': success_rate > 0.5  # 超过50%成功率认为可用
    }
    
    if result['working']:
        print(f"   ✅ {model_name} 推荐功能正常 (成功率: {success_rate:.1%})")
        if sample_recommendations:
            print(f"   📝 推荐示例: 用户{sample_recommendations[0][0]} -> {[r.get('item_id', r) for r in sample_recommendations[0][1]]}")
    else:
        print(f"   ❌ {model_name} 推荐功能异常 (成功率: {success_rate:.1%})")
        if errors:
            print(f"   🔍 主要错误: {errors[0]}")
    
    return result


def run_complete_teacher_experiment():
    """运行完整的6个Teacher模型实验"""
    print("🎓 完整6个Teacher模型CUDA实验")
    print("=" * 60)
    
    # 检查CUDA环境
    cuda_available = check_cuda_environment()
    device = torch.device('cuda' if cuda_available else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 创建数据
    print("\\n📊 创建模拟推荐数据...")
    np.random.seed(42)
    
    num_interactions = 6000 if cuda_available else 3000
    num_users = 250 if cuda_available else 200
    num_items = 200 if cuda_available else 150
    
    interactions = []
    for _ in range(num_interactions):
        user_id = np.random.randint(1, num_users + 1)
        item_id = np.random.randint(1, num_items + 1)
        rating = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
        timestamp = int(time.time()) + np.random.randint(-86400*30, 86400*30)
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': timestamp
        })
    
    train_data = pd.DataFrame(interactions)
    train_data = train_data.drop_duplicates(['user_id', 'item_id'])
    
    print(f"✅ 生成了 {len(train_data)} 条唯一交互数据")
    print(f"   用户数: {train_data['user_id'].nunique()}")
    print(f"   物品数: {train_data['item_id'].nunique()}")
    
    # 6个Teacher模型配置
    algorithms = ['deepfm', 'autoint', 'transformer4rec', 'xdeepfm', 'din', 'dcnv2']
    
    if cuda_available:
        model_configs = {
            'deepfm': {
                'embedding_dim': 32,
                'learning_rate': 0.001,
                'epochs': 15,
                'batch_size': 128,
                'early_stopping': True,
                'patience': 3
            },
            'autoint': {
                'embedding_dim': 32,
                'learning_rate': 0.001,
                'epochs': 15,
                'num_heads': 8,
                'num_layers': 4,
                'batch_size': 128,
                'early_stopping': True,
                'patience': 3
            },
            'transformer4rec': {
                'embedding_dim': 64,
                'num_heads': 8,
                'num_layers': 4,
                'learning_rate': 0.001,
                'epochs': 12,
                'batch_size': 128,
                'early_stopping': True,
                'patience': 3
            },
            'xdeepfm': {
                'embedding_dim': 32,
                'cin_layer_sizes': [128, 64],
                'dnn_hidden_dims': [256, 128],
                'learning_rate': 0.001,
                'epochs': 12,
                'batch_size': 128,
                'early_stopping': True,
                'patience': 3
            },
            'din': {
                'embedding_dim': 32,
                'hidden_dims': [256, 128, 64],
                'attention_hidden_dim': 128,
                'learning_rate': 0.001,
                'epochs': 12,
                'batch_size': 128,
                'early_stopping': True,
                'patience': 3
            },
            'dcnv2': {
                'embedding_dim': 32,
                'cross_layers': 4,
                'deep_layers': [256, 128, 64],
                'learning_rate': 0.001,
                'epochs': 12,
                'batch_size': 128,
                'early_stopping': True,
                'patience': 3
            }
        }
    else:
        model_configs = {
            'deepfm': {'embedding_dim': 16, 'learning_rate': 0.001, 'epochs': 10, 'batch_size': 64, 'early_stopping': False},
            'autoint': {'embedding_dim': 16, 'learning_rate': 0.001, 'epochs': 10, 'num_heads': 4, 'num_layers': 3, 'early_stopping': False},
            'transformer4rec': {'embedding_dim': 32, 'num_heads': 4, 'num_layers': 3, 'learning_rate': 0.001, 'epochs': 8, 'batch_size': 64, 'early_stopping': False},
            'xdeepfm': {'embedding_dim': 16, 'cin_layer_sizes': [64, 32], 'dnn_hidden_dims': [128, 64], 'learning_rate': 0.001, 'epochs': 8, 'batch_size': 64, 'early_stopping': False},
            'din': {'embedding_dim': 16, 'hidden_dims': [128, 64], 'attention_hidden_dim': 64, 'learning_rate': 0.001, 'epochs': 8, 'batch_size': 64, 'early_stopping': False},
            'dcnv2': {'embedding_dim': 16, 'cross_layers': 3, 'deep_layers': [128, 64], 'learning_rate': 0.001, 'epochs': 8, 'batch_size': 64, 'early_stopping': False}
        }
    
    print(f"\\n🎯 将训练 {len(algorithms)} 个Teacher模型:")
    for algo in algorithms:
        print(f"   - {algo}")
    
    # 训练模型
    trained_models = {}
    training_times = {}
    training_errors = {}
    
    for algo_name in algorithms:
        print(f"\\n📚 训练Teacher模型: {algo_name}")
        config = model_configs[algo_name]
        
        try:
            start_time = time.time()
            
            if cuda_available:
                torch.cuda.empty_cache()
            
            model = create_recommender(algo_name, **config)
            
            if model is None:
                print(f"❌ 无法创建算法: {algo_name}")
                training_errors[algo_name] = "算法创建失败"
                continue
            
            if hasattr(model, 'device'):
                model.device = device
            
            try:
                model.fit(train_data, **config)
                end_time = time.time()
                
                training_time = end_time - start_time
                training_times[algo_name] = training_time
                
                if model.is_trained:
                    trained_models[algo_name] = model
                    print(f"✅ {algo_name} 训练成功! 耗时: {training_time:.2f}秒")
                else:
                    print(f"❌ {algo_name} 训练失败 - 模型未完成训练")
                    training_errors[algo_name] = "训练未完成"
                    
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    print(f"⚠️ {algo_name} CUDA错误，尝试CPU模式...")
                    
                    if cuda_available:
                        torch.cuda.empty_cache()
                    
                    cpu_config = config.copy()
                    cpu_config['batch_size'] = min(cpu_config.get('batch_size', 64), 32)
                    cpu_config['epochs'] = min(cpu_config.get('epochs', 10), 6)
                    
                    model = create_recommender(algo_name, **cpu_config)
                    if hasattr(model, 'device'):
                        model.device = torch.device('cpu')
                    
                    model.fit(train_data, **cpu_config)
                    end_time = time.time()
                    
                    training_time = end_time - start_time
                    training_times[algo_name] = training_time
                    
                    if model.is_trained:
                        trained_models[algo_name] = model
                        print(f"✅ {algo_name} CPU模式训练成功! 耗时: {training_time:.2f}秒")
                    else:
                        print(f"❌ {algo_name} CPU模式也失败")
                        training_errors[algo_name] = f"CUDA和CPU都失败"
                else:
                    raise e
                    
        except Exception as e:
            training_time = time.time() - start_time
            training_times[algo_name] = training_time
            training_errors[algo_name] = f"{type(e).__name__}: {str(e)[:50]}"
            print(f"❌ 训练 {algo_name} 时出错: {str(e)[:50]}...")
            
            if cuda_available:
                torch.cuda.empty_cache()
    
    print(f"\\n🎯 训练阶段完成!")
    print(f"   成功训练: {len(trained_models)} 个Teacher模型")
    print(f"   成功的模型: {list(trained_models.keys())}")
    
    if training_errors:
        print(f"   失败的模型: {list(training_errors.keys())}")
    
    if len(trained_models) < 2:
        print("❌ 训练成功的模型少于2个，无法进行一致性分析")
        return None
    
    # 测试推荐功能
    print("\\n🔍 测试所有模型的推荐功能...")
    test_users = list(train_data['user_id'].unique())[:50]
    
    working_models = {}
    recommendation_tests = {}
    
    for algo_name, model in trained_models.items():
        test_result = test_model_recommendation(model, algo_name, test_users)
        recommendation_tests[algo_name] = test_result
        
        if test_result['working']:
            working_models[algo_name] = model
    
    print(f"\\n📊 推荐功能测试结果:")
    print(f"   可用模型: {list(working_models.keys())}")
    if len(working_models) != len(trained_models):
        failed_recs = set(trained_models.keys()) - set(working_models.keys())
        print(f"   推荐失败: {list(failed_recs)}")
    
    if len(working_models) < 2:
        print("❌ 可用于推荐的模型少于2个，无法进行一致性分析")
        return None
    
    # 生成推荐并分析一致性
    print("\\n📊 生成完整推荐数据集...")
    test_users_full = list(train_data['user_id'].unique())[:100 if cuda_available else 50]
    
    all_recommendations = {}
    performance_stats = {}
    
    for algo_name, model in working_models.items():
        print(f"🔍 生成 {algo_name} 的推荐...")
        
        recommendations = {}
        success_count = 0
        total_rec_count = 0
        error_count = 0
        
        for user_id in test_users_full:
            try:
                recs = model.get_user_recommendations(user_id, top_k=10)
                if recs and len(recs) > 0:
                    # 确保推荐格式正确
                    if isinstance(recs[0], dict) and 'item_id' in recs[0]:
                        rec_items = [rec['item_id'] for rec in recs]
                    elif isinstance(recs[0], tuple):
                        rec_items = [rec[0] for rec in recs]
                    else:
                        rec_items = recs
                    
                    recommendations[user_id] = rec_items
                    success_count += 1
                    total_rec_count += len(rec_items)
            except Exception as e:
                error_count += 1
                continue
        
        if recommendations:
            all_recommendations[algo_name] = recommendations
            performance_stats[algo_name] = {
                'success_rate': success_count / len(test_users_full),
                'avg_rec_length': total_rec_count / success_count if success_count > 0 else 0,
                'total_recommendations': len(recommendations),
                'training_time': training_times.get(algo_name, 0),
                'error_count': error_count
            }
            print(f"   ✅ 成功为 {len(recommendations)} 个用户生成推荐")
            print(f"   📈 成功率: {success_count/len(test_users_full):.2%}")
            if error_count > 0:
                print(f"   ⚠️ 错误数: {error_count}")
    
    # 计算一致性矩阵
    print("\\n🔍 计算Teacher模型间一致性...")
    
    if len(all_recommendations) < 2:
        print("❌ 无法进行一致性分析 - 推荐生成失败")
        return None
    
    algorithms_list = list(all_recommendations.keys())
    jaccard_matrix = {}
    
    print(f"📊 分析 {len(algorithms_list)} 个模型间的一致性:")
    
    for i, algo1 in enumerate(algorithms_list):
        jaccard_matrix[algo1] = {}
        
        for j, algo2 in enumerate(algorithms_list):
            if algo1 == algo2:
                jaccard_matrix[algo1][algo2] = 1.0
            else:
                similarities = []
                
                common_users = set(all_recommendations[algo1].keys()) & set(all_recommendations[algo2].keys())
                
                for user_id in common_users:
                    recs1 = set(all_recommendations[algo1][user_id][:10])
                    recs2 = set(all_recommendations[algo2][user_id][:10])
                    
                    sim = simple_jaccard_similarity(recs1, recs2)
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities) if similarities else 0.0
                jaccard_matrix[algo1][algo2] = avg_similarity
        
        print(f"   ✅ 完成 {algo1} 的一致性计算")
    
    # 分析结果
    print("\\n📈 分析Teacher模型一致性结果...")
    
    min_jaccard = float('inf')
    max_jaccard = 0.0
    min_pair = None
    max_pair = None
    
    all_similarities = []
    
    for algo1 in algorithms_list:
        for algo2 in algorithms_list:
            if algo1 != algo2:
                similarity = jaccard_matrix[algo1][algo2]
                all_similarities.append(similarity)
                
                if similarity < min_jaccard:
                    min_jaccard = similarity
                    min_pair = (algo1, algo2)
                if similarity > max_jaccard:
                    max_jaccard = similarity
                    max_pair = (algo1, algo2)
    
    avg_jaccard = np.mean(all_similarities) if all_similarities else 0.0
    
    # 生成报告
    print("\\n📝 生成完整Teacher模型分析报告...")
    
    report = []
    report.append("# 🎓 完整6个Teacher模型一致性分析报告")
    report.append(f"**实验时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**运行设备**: {device}")
    report.append("")
    
    report.append("## 📊 实验概述")
    report.append(f"- **目标Teacher模型**: {', '.join(algorithms)}")
    report.append(f"- **成功训练模型**: {', '.join(trained_models.keys())}")
    report.append(f"- **推荐可用模型**: {', '.join(working_models.keys())}")
    report.append(f"- **参与一致性分析**: {', '.join(algorithms_list)}")
    report.append(f"- **数据集规模**: {len(train_data)} 条交互")
    report.append(f"- **测试用户数**: {len(test_users_full)}")
    report.append(f"- **运行模式**: {'CUDA加速' if cuda_available else 'CPU模式'}")
    report.append("")
    
    # 训练性能
    report.append("## ⏱️ Teacher模型训练性能")
    report.append("")
    report.append("| 模型 | 训练时间(秒) | 训练状态 | 推荐状态 |")
    report.append("|------|-------------|----------|----------|")
    
    for algo in algorithms:
        train_status = "✅ 成功" if algo in trained_models else "❌ 失败"
        rec_status = "✅ 可用" if algo in working_models else "❌ 不可用"
        train_time = training_times.get(algo, 0)
        if train_time > 0:
            report.append(f"| {algo} | {train_time:.2f} | {train_status} | {rec_status} |")
        else:
            report.append(f"| {algo} | - | {train_status} | {rec_status} |")
    report.append("")
    
    # 推荐性能
    if performance_stats:
        report.append("## 🏆 Teacher模型推荐性能")
        report.append("")
        report.append("| 模型 | 成功率 | 平均推荐数 | 总推荐数 | 训练时间(秒) |")
        report.append("|------|--------|------------|----------|-------------|")
        
        for algo_name, stats in performance_stats.items():
            report.append(f"| {algo_name} | {stats['success_rate']:.2%} | {stats['avg_rec_length']:.1f} | {stats['total_recommendations']} | {stats['training_time']:.2f} |")
        report.append("")
    
    # 一致性分析
    if len(algorithms_list) >= 2:
        report.append("## 🔄 Teacher模型一致性分析")
        report.append("")
        
        report.append("### 📊 关键指标")
        report.append(f"- **参与分析的模型数**: {len(algorithms_list)}")
        report.append(f"- **平均Jaccard相似度**: {avg_jaccard:.4f}")
        if min_pair:
            report.append(f"- **最强互补组合**: {min_pair[0]} + {min_pair[1]} (Jaccard={min_jaccard:.4f})")
        if max_pair:
            report.append(f"- **最相似组合**: {max_pair[0]} + {max_pair[1]} (Jaccard={max_jaccard:.4f})")
        report.append("")
        
        # Jaccard相似度矩阵
        report.append("### 🔢 完整Jaccard相似度矩阵")
        report.append("")
        
        header = "| 算法 |" + "".join([f" {algo} |" for algo in algorithms_list])
        separator = "|" + "".join(["------|" for _ in range(len(algorithms_list) + 1)])
        
        report.append(header)
        report.append(separator)
        
        for algo1 in algorithms_list:
            row = f"| **{algo1}** |"
            for algo2 in algorithms_list:
                if algo1 == algo2:
                    row += " 1.0000 |"
                else:
                    similarity = jaccard_matrix[algo1][algo2]
                    row += f" {similarity:.4f} |"
            report.append(row)
        
        report.append("")
        
        # Ensemble建议
        report.append("## 🎯 完整Teacher模型Ensemble策略")
        report.append("")
        
        if min_pair and max_pair:
            report.append("### 🥇 基于完整分析的最佳组合")
            report.append("")
            report.append(f"**最强互补策略**: {min_pair[0]} + {min_pair[1]}")
            report.append(f"- **Jaccard相似度**: {min_jaccard:.4f}")
            report.append(f"- **特点**: 在所有{len(algorithms_list)}个可用模型中推荐重叠度最低")
            report.append("")
            
            report.append(f"**最强一致策略**: {max_pair[0]} + {max_pair[1]}")
            report.append(f"- **Jaccard相似度**: {max_jaccard:.4f}")  
            report.append(f"- **特点**: 在所有{len(algorithms_list)}个可用模型中推荐一致性最高")
            report.append("")
    
    # 修复总结
    if len(working_models) == len(algorithms):
        report.append("## ✅ 完整实验成功总结")
        report.append("")
        report.append("- **所有6个Teacher模型都成功训练并可用于推荐**")
        report.append("- **所有模型都参与了一致性分析**")
        report.append("- **推荐接口统一，数据格式一致**")
        report.append("- **CUDA环境运行稳定**")
    else:
        report.append("## ⚠️ 部分实验成功总结")
        report.append("")
        report.append(f"- **{len(working_models)}/{len(algorithms)} 个Teacher模型可用于推荐**")
        report.append(f"- **参与一致性分析的模型: {len(algorithms_list)} 个**")
        if len(working_models) < len(trained_models):
            failed_models = set(trained_models.keys()) - set(working_models.keys())
            report.append(f"- **推荐功能异常的模型**: {', '.join(failed_models)}")
    
    # 保存报告
    report_content = "\\n".join(report)
    
    with open("COMPLETE_TEACHER_MODEL_ANALYSIS.md", "w", encoding='utf-8') as f:
        f.write(report_content)
    
    print("✅ 完整Teacher模型分析报告已保存到 COMPLETE_TEACHER_MODEL_ANALYSIS.md")
    
    # 显示关键结果
    print("\\n🎉 完整6个Teacher模型实验完成！")
    print("\\n📊 关键结果:")
    print(f"📈 成功训练的Teacher模型 ({len(trained_models)}/{len(algorithms)}): {list(trained_models.keys())}")
    print(f"🔧 推荐可用的模型 ({len(working_models)}/{len(algorithms)}): {list(working_models.keys())}")
    print(f"📊 参与一致性分析: {len(algorithms_list)} 个模型")
    
    if avg_jaccard is not None:
        print(f"📊 平均Jaccard相似度: {avg_jaccard:.4f}")
    
    if min_pair:
        print(f"🏆 最佳互补组合: {min_pair[0]} + {min_pair[1]} (Jaccard={min_jaccard:.4f})")
    
    if max_pair:
        print(f"⚠️  最相似组合: {max_pair[0]} + {max_pair[1]} (Jaccard={max_jaccard:.4f})")
    
    # 显示完整一致性矩阵
    if len(algorithms_list) >= 2:
        print("\\n📋 完整Jaccard相似度矩阵:")
        print("     ", end="")
        for algo in algorithms_list:
            print(f"{algo:>12}", end="")
        print()
        
        for algo1 in algorithms_list:
            print(f"{algo1:>8}", end="")
            for algo2 in algorithms_list:
                if algo1 == algo2:
                    print(f"{'1.0000':>12}", end="")
                else:
                    print(f"{jaccard_matrix[algo1][algo2]:>12.4f}", end="")
            print()
    
    # 清理GPU内存
    if cuda_available:
        torch.cuda.empty_cache()
    
    return {
        'trained_models': list(trained_models.keys()),
        'working_models': list(working_models.keys()),
        'analyzed_models': algorithms_list,
        'jaccard_matrix': jaccard_matrix if len(algorithms_list) >= 2 else None,
        'best_complementary': min_pair,
        'most_similar': max_pair,
        'avg_jaccard': avg_jaccard,
        'performance_stats': performance_stats,
        'device': str(device),
        'recommendation_tests': recommendation_tests
    }


if __name__ == "__main__":
    result = run_complete_teacher_experiment()
    if result:
        analyzed_count = len(result['analyzed_models'])
        total_count = 6
        print(f"\\n✅ 实验完成！参与一致性分析的模型: {analyzed_count}/{total_count}")
        print(f"🔧 详细报告请查看 COMPLETE_TEACHER_MODEL_ANALYSIS.md")
        
        if analyzed_count == total_count:
            print("🎉 所有6个Teacher模型都成功参与了一致性分析！")
        else:
            print(f"⚠️ 还有 {total_count - analyzed_count} 个模型需要进一步修复")
    else:
        print(f"\\n❌ 实验失败，请检查错误信息")
