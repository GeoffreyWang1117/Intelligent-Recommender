#!/usr/bin/env python3
"""
Complete LLM Real Data Experiment - 完整的LLM真实数据实验
包含推荐生成、Fisher分析、PAKD实验的完整流程

功能:
1. 基于真实MovieLens数据的LLM推荐
2. LLM Fisher Information分析
3. LLM PAKD (Pruning-Aware Knowledge Distillation)
4. 完整的评估和报告生成

作者: GitHub Copilot
日期: 2025-08-29
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from real_movielens_llm_recommender import RealMovieLensLLMRecommender
    from llm_fisher_calculator import LLMFisherCalculator
    from llm_pakd_distiller import LLMPAKDDistiller
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保所有依赖文件位于正确路径")
    sys.exit(1)

class CompleteLLMRealDataExperiment:
    """完整的LLM真实数据实验"""
    
    def __init__(self, output_dir: str = "evaluation/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.recommender = RealMovieLensLLMRecommender()
        self.fisher_calculator = LLMFisherCalculator()
        self.pakd_distiller = LLMPAKDDistiller()
        
        # 实验配置
        self.experiment_config = {
            'dataset_size': 'small',
            'test_users_count': 30,
            'llm_models': ['llama3', 'qwen3'],
            'fisher_sample_size': 100,
            'pakd_distillation_samples': 200,
            'num_recommendations': 10
        }
        
        self.experiment_results = {}
        
        print("🧪 Complete LLM Real Data Experiment 初始化完成")
        print(f"📁 输出目录: {self.output_dir}")
        
    def run_complete_experiment(self) -> Dict[str, Any]:
        """运行完整的LLM实验"""
        print("🚀 开始完整的LLM真实数据实验")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1. 数据加载和验证
        if not self._load_and_validate_data():
            print("❌ 数据加载失败，实验终止")
            return {}
        
        # 2. LLM推荐生成实验
        recommendation_results = self._run_recommendation_experiment()
        
        # 3. Fisher Information分析
        fisher_results = self._run_fisher_analysis()
        
        # 4. PAKD实验
        pakd_results = self._run_pakd_experiment()
        
        # 5. 汇总和评估
        final_results = self._compile_final_results(
            recommendation_results, fisher_results, pakd_results, start_time
        )
        
        # 6. 保存结果和生成报告
        self._save_results_and_reports(final_results)
        
        print(f"\n🎉 完整实验完成！总耗时: {time.time() - start_time:.2f}秒")
        return final_results
    
    def _load_and_validate_data(self) -> bool:
        """加载和验证真实数据"""
        print("📊 步骤1: 加载和验证真实MovieLens数据...")
        
        # 加载真实数据
        success = self.recommender.load_real_movielens_data(self.experiment_config['dataset_size'])
        
        if not success:
            print("❌ 无法加载真实MovieLens数据")
            return False
        
        # 验证数据质量
        if len(self.recommender.user_profiles) < 10:
            print("❌ 用户数据不足，无法进行可靠实验")
            return False
        
        if len(self.recommender.movie_features) < 100:
            print("❌ 电影数据不足，无法进行可靠实验")
            return False
        
        print(f"✅ 数据验证通过:")
        print(f"   👥 用户数: {len(self.recommender.user_profiles)}")
        print(f"   🎬 电影数: {len(self.recommender.movie_features)}")
        print(f"   ⭐ 评分数: {len(self.recommender.ratings)}")
        
        return True
    
    def _run_recommendation_experiment(self) -> Dict[str, Any]:
        """运行LLM推荐实验"""
        print("\n🤖 步骤2: LLM推荐生成实验...")
        
        results = {
            'models': {},
            'experiment_stats': {},
            'performance_metrics': {}
        }
        
        # 选择测试用户
        import numpy as np
        test_user_ids = np.random.choice(
            list(self.recommender.user_profiles.keys()),
            size=min(self.experiment_config['test_users_count'], len(self.recommender.user_profiles)),
            replace=False
        )
        
        print(f"👥 选择了 {len(test_user_ids)} 个测试用户")
        
        # 对每个模型进行测试
        for model in self.experiment_config['llm_models']:
            print(f"\n🔧 测试模型: {model}")
            
            model_results = {
                'successful_recommendations': 0,
                'failed_recommendations': 0,
                'total_response_time': 0.0,
                'recommendations_data': [],
                'error_messages': []
            }
            
            for user_id in test_user_ids:
                try:
                    start_time = time.time()
                    
                    # 生成推荐
                    recommendation = self.recommender.generate_llm_recommendations(
                        user_id=user_id,
                        num_recommendations=self.experiment_config['num_recommendations'],
                        model=model,
                        explain=True
                    )
                    
                    response_time = time.time() - start_time
                    model_results['total_response_time'] += response_time
                    
                    if recommendation and recommendation.get('recommendations'):
                        model_results['successful_recommendations'] += 1
                        model_results['recommendations_data'].append(recommendation)
                        print(f"  ✅ 用户 {user_id}: {len(recommendation['recommendations'])} 个推荐 ({response_time:.2f}s)")
                    else:
                        model_results['failed_recommendations'] += 1
                        print(f"  ❌ 用户 {user_id}: 推荐生成失败")
                        
                except Exception as e:
                    model_results['failed_recommendations'] += 1
                    model_results['error_messages'].append(f"用户 {user_id}: {str(e)}")
                    print(f"  ⚠️  用户 {user_id}: 异常 - {str(e)}")
            
            # 计算模型统计
            total_attempts = model_results['successful_recommendations'] + model_results['failed_recommendations']
            model_results['success_rate'] = model_results['successful_recommendations'] / total_attempts if total_attempts > 0 else 0
            model_results['avg_response_time'] = model_results['total_response_time'] / model_results['successful_recommendations'] if model_results['successful_recommendations'] > 0 else 0
            
            results['models'][model] = model_results
            
            print(f"📊 {model} 结果:")
            print(f"   成功率: {model_results['success_rate']:.2%}")
            print(f"   平均响应时间: {model_results['avg_response_time']:.2f}s")
        
        return results
    
    def _run_fisher_analysis(self) -> Dict[str, Any]:
        """运行Fisher Information分析"""
        print("\n🔍 步骤3: Fisher Information分析...")
        
        # 检查是否有LLM交互数据
        if not hasattr(self.recommender, 'llm_interactions') or not self.recommender.llm_interactions:
            print("⚠️  没有LLM交互数据，跳过Fisher分析")
            return {'error': 'No LLM interaction data available'}
        
        interactions = self.recommender.llm_interactions[:self.experiment_config['fisher_sample_size']]
        print(f"📊 使用 {len(interactions)} 个交互样本进行Fisher分析")
        
        try:
            fisher_results = self.fisher_calculator.compute_llm_fisher_information(
                interactions, embedding_dim=512
            )
            
            print("✅ Fisher Information分析完成")
            return fisher_results
            
        except Exception as e:
            print(f"❌ Fisher分析失败: {e}")
            return {'error': str(e)}
    
    def _run_pakd_experiment(self) -> Dict[str, Any]:
        """运行PAKD实验"""
        print("\n🧠 步骤4: PAKD (Pruning-Aware Knowledge Distillation) 实验...")
        
        # 生成Teacher-Student数据对
        try:
            teacher_data, student_data = self._generate_teacher_student_data()
            
            if not teacher_data or not student_data:
                print("⚠️  无法生成足够的Teacher-Student数据，跳过PAKD实验")
                return {'error': 'Insufficient data for PAKD'}
            
            print(f"👨‍🏫 Teacher数据: {len(teacher_data)} 样本")
            print(f"👨‍🎓 Student数据: {len(student_data)} 样本")
            
            # 运行PAKD
            pakd_results = self.pakd_distiller.run_llm_pakd(
                teacher_data=teacher_data,
                student_data=student_data,
                distillation_config={
                    'temperature': 3.0,
                    'alpha': 0.7,
                    'pruning_ratio': 0.1
                }
            )
            
            print("✅ PAKD实验完成")
            return pakd_results
            
        except Exception as e:
            print(f"❌ PAKD实验失败: {e}")
            return {'error': str(e)}
    
    def _generate_teacher_student_data(self) -> tuple[List[Dict], List[Dict]]:
        """生成Teacher-Student数据对"""
        print("👥 生成Teacher-Student数据对...")
        
        teacher_data = []
        student_data = []
        
        # 选择用户子集
        import numpy as np
        user_ids = np.random.choice(
            list(self.recommender.user_profiles.keys()),
            size=min(self.experiment_config['pakd_distillation_samples'] // 2, len(self.recommender.user_profiles)),
            replace=False
        )
        
        for user_id in user_ids:
            try:
                # 生成Teacher推荐 (Llama3)
                teacher_rec = self.recommender.generate_llm_recommendations(
                    user_id=user_id,
                    model='llama3',
                    num_recommendations=5
                )
                
                # 生成Student推荐 (Qwen3)
                student_rec = self.recommender.generate_llm_recommendations(
                    user_id=user_id,
                    model='qwen3',
                    num_recommendations=5
                )
                
                if teacher_rec and teacher_rec.get('recommendations') and student_rec and student_rec.get('recommendations'):
                    teacher_data.append(teacher_rec)
                    student_data.append(student_rec)
                
                # 限制生成数量
                if len(teacher_data) >= self.experiment_config['pakd_distillation_samples']:
                    break
                    
            except Exception as e:
                print(f"⚠️  生成数据失败 (用户 {user_id}): {e}")
                continue
        
        return teacher_data, student_data
    
    def _compile_final_results(self, recommendation_results: Dict, fisher_results: Dict,
                             pakd_results: Dict, start_time: float) -> Dict[str, Any]:
        """汇总最终结果"""
        print("\n📋 步骤5: 汇总实验结果...")
        
        final_results = {
            'experiment_info': {
                'experiment_name': 'Complete LLM Real Data Experiment',
                'timestamp': time.time(),
                'duration_seconds': time.time() - start_time,
                'config': self.experiment_config,
                'data_source': 'Real MovieLens Dataset'
            },
            'dataset_info': {
                'num_users': len(self.recommender.user_profiles),
                'num_movies': len(self.recommender.movie_features),
                'num_ratings': len(self.recommender.ratings),
                'sparsity': float((1 - len(self.recommender.ratings) / (len(self.recommender.user_profiles) * len(self.recommender.movie_features))) * 100)
            },
            'recommendation_results': recommendation_results,
            'fisher_analysis': fisher_results,
            'pakd_experiment': pakd_results,
            'summary_metrics': self._compute_summary_metrics(recommendation_results, fisher_results, pakd_results)
        }
        
        return final_results
    
    def _compute_summary_metrics(self, rec_results: Dict, fisher_results: Dict, pakd_results: Dict) -> Dict[str, Any]:
        """计算汇总指标"""
        summary = {
            'overall_success': True,
            'recommendation_performance': {},
            'analysis_coverage': {},
            'experiment_quality': 'high'
        }
        
        # 推荐性能汇总
        if rec_results and 'models' in rec_results:
            total_success = 0
            total_attempts = 0
            avg_response_times = []
            
            for model, stats in rec_results['models'].items():
                total_success += stats.get('successful_recommendations', 0)
                total_attempts += stats.get('successful_recommendations', 0) + stats.get('failed_recommendations', 0)
                if stats.get('avg_response_time', 0) > 0:
                    avg_response_times.append(stats['avg_response_time'])
            
            summary['recommendation_performance'] = {
                'overall_success_rate': total_success / total_attempts if total_attempts > 0 else 0,
                'avg_response_time': sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0,
                'models_tested': len(rec_results['models'])
            }
        
        # 分析覆盖度
        summary['analysis_coverage'] = {
            'recommendation_experiment': 'error' not in rec_results,
            'fisher_analysis': 'error' not in fisher_results,
            'pakd_experiment': 'error' not in pakd_results,
            'comprehensive_coverage': all([
                'error' not in rec_results,
                'error' not in fisher_results,
                'error' not in pakd_results
            ])
        }
        
        # 实验质量评估
        if summary['recommendation_performance'].get('overall_success_rate', 0) < 0.5:
            summary['experiment_quality'] = 'low'
        elif not summary['analysis_coverage']['comprehensive_coverage']:
            summary['experiment_quality'] = 'medium'
        
        return summary
    
    def _save_results_and_reports(self, final_results: Dict[str, Any]):
        """保存结果和生成报告"""
        print("\n💾 步骤6: 保存结果和生成报告...")
        
        # 保存完整结果JSON
        results_file = self.output_dir / "complete_llm_real_data_experiment.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📄 完整结果已保存: {results_file}")
        
        # 生成可读报告
        report = self._generate_experiment_report(final_results)
        report_file = self.output_dir / "complete_llm_experiment_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📑 实验报告已生成: {report_file}")
        
        # 保存子实验结果
        if 'fisher_analysis' in final_results and 'error' not in final_results['fisher_analysis']:
            fisher_file = self.output_dir / "llm_fisher_analysis.json"
            self.fisher_calculator.save_fisher_analysis(final_results['fisher_analysis'], fisher_file)
        
        if 'pakd_experiment' in final_results and 'error' not in final_results['pakd_experiment']:
            pakd_file = self.output_dir / "llm_pakd_experiment.json"
            self.pakd_distiller.save_pakd_results(final_results['pakd_experiment'], pakd_file)
    
    def _generate_experiment_report(self, results: Dict[str, Any]) -> str:
        """生成实验报告"""
        report = "# Complete LLM Real Data Experiment Report\n\n"
        
        # 实验概述
        exp_info = results.get('experiment_info', {})
        report += "## 🧪 实验概述\n"
        report += f"- **实验名称**: {exp_info.get('experiment_name', 'N/A')}\n"
        report += f"- **执行时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp_info.get('timestamp', 0)))}\n"
        report += f"- **总耗时**: {exp_info.get('duration_seconds', 0):.2f} 秒\n"
        report += f"- **数据源**: {exp_info.get('data_source', 'N/A')}\n\n"
        
        # 数据集信息
        dataset_info = results.get('dataset_info', {})
        report += "## 📊 数据集信息\n"
        report += f"- **用户数**: {dataset_info.get('num_users', 0):,}\n"
        report += f"- **电影数**: {dataset_info.get('num_movies', 0):,}\n"
        report += f"- **评分数**: {dataset_info.get('num_ratings', 0):,}\n"
        report += f"- **稀疏度**: {dataset_info.get('sparsity', 0):.2f}%\n\n"
        
        # 推荐实验结果
        rec_results = results.get('recommendation_results', {})
        if 'error' not in rec_results and 'models' in rec_results:
            report += "## 🤖 LLM推荐实验结果\n"
            for model, stats in rec_results['models'].items():
                report += f"### {model.upper()}\n"
                report += f"- **成功率**: {stats.get('success_rate', 0):.2%}\n"
                report += f"- **成功推荐数**: {stats.get('successful_recommendations', 0)}\n"
                report += f"- **失败推荐数**: {stats.get('failed_recommendations', 0)}\n"
                report += f"- **平均响应时间**: {stats.get('avg_response_time', 0):.2f} 秒\n\n"
        
        # Fisher分析结果
        fisher_results = results.get('fisher_analysis', {})
        if 'error' not in fisher_results:
            report += "## 🔍 Fisher Information分析结果\n"
            report += f"- **分析样本数**: {fisher_results.get('interaction_count', 0)}\n"
            report += f"- **嵌入维度**: {fisher_results.get('embedding_dimension', 0)}\n"
            
            # Fisher矩阵统计
            fisher_matrices = fisher_results.get('fisher_matrices', {})
            for emb_type, matrix_info in fisher_matrices.items():
                if 'error' not in matrix_info:
                    report += f"### {emb_type}\n"
                    report += f"- **Fisher迹**: {matrix_info.get('trace', 0):.6f}\n"
                    report += f"- **信息内容**: {matrix_info.get('information_content', 0):.6f}\n"
            report += "\n"
        else:
            report += "## 🔍 Fisher Information分析\n"
            report += f"⚠️ 分析失败: {fisher_results.get('error', 'Unknown error')}\n\n"
        
        # PAKD实验结果
        pakd_results = results.get('pakd_experiment', {})
        if 'error' not in pakd_results:
            report += "## 🧠 PAKD实验结果\n"
            
            config = pakd_results.get('config', {})
            report += f"- **蒸馏温度**: {config.get('temperature', 'N/A')}\n"
            report += f"- **Alpha权重**: {config.get('alpha', 'N/A')}\n"
            report += f"- **剪枝比例**: {config.get('pruning_ratio', 'N/A')}\n"
            
            data_info = pakd_results.get('data_info', {})
            report += f"- **Teacher样本数**: {data_info.get('teacher_samples', 0)}\n"
            report += f"- **Student样本数**: {data_info.get('student_samples', 0)}\n"
            report += f"- **对齐数据对**: {data_info.get('aligned_pairs', 0)}\n"
            
            # 蒸馏结果
            dist_results = pakd_results.get('distillation_results', {})
            if 'error' not in dist_results:
                report += f"- **蒸馏损失**: {dist_results.get('distillation_loss', 0):.6f}\n"
                report += f"- **总损失**: {dist_results.get('total_loss', 0):.6f}\n"
            
            report += "\n"
        else:
            report += "## 🧠 PAKD实验\n"
            report += f"⚠️ 实验失败: {pakd_results.get('error', 'Unknown error')}\n\n"
        
        # 总结
        summary = results.get('summary_metrics', {})
        report += "## 📈 实验总结\n"
        report += f"- **实验质量**: {summary.get('experiment_quality', 'unknown').upper()}\n"
        
        rec_perf = summary.get('recommendation_performance', {})
        if rec_perf:
            report += f"- **整体推荐成功率**: {rec_perf.get('overall_success_rate', 0):.2%}\n"
            report += f"- **平均响应时间**: {rec_perf.get('avg_response_time', 0):.2f} 秒\n"
        
        coverage = summary.get('analysis_coverage', {})
        if coverage:
            report += f"- **分析覆盖度**: {'全面' if coverage.get('comprehensive_coverage') else '部分'}\n"
        
        report += "\n## ✅ 实验完成\n"
        report += "所有实验流程均基于真实MovieLens数据集，确保了结果的可靠性和实用性。\n"
        
        return report


def main():
    """主函数"""
    print("🚀 Complete LLM Real Data Experiment 启动")
    print("🎯 目标: 完整的LLM推荐+Fisher+PAKD实验流程")
    print("📊 数据: 仅使用真实MovieLens数据")
    print("=" * 80)
    
    # 创建实验实例
    experiment = CompleteLLMRealDataExperiment()
    
    # 运行完整实验
    results = experiment.run_complete_experiment()
    
    if results:
        print("\n🎉 实验成功完成！")
        print(f"📁 结果保存在: {experiment.output_dir}")
        
        # 简要总结
        summary = results.get('summary_metrics', {})
        if summary:
            print(f"📊 实验质量: {summary.get('experiment_quality', 'unknown').upper()}")
            
            rec_perf = summary.get('recommendation_performance', {})
            if rec_perf:
                print(f"🤖 推荐成功率: {rec_perf.get('overall_success_rate', 0):.2%}")
            
            coverage = summary.get('analysis_coverage', {})
            if coverage:
                print(f"🔍 分析覆盖: {'全面' if coverage.get('comprehensive_coverage') else '部分'}")
    else:
        print("❌ 实验失败")


if __name__ == "__main__":
    main()
