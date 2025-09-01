#!/usr/bin/env python3
"""
LayerwiseAdapter vs Ensemble Baseline 性能对比分析

对比分析LayerwiseAdapter增强版与基准Ensemble模型在MovieLens 1M数据集上的性能
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os

class PerformanceComparator:
    """性能对比分析器"""
    
    def __init__(self):
        self.results = {}
        
    def load_results(self, model_name: str, results_file: str):
        """加载模型结果"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.results[model_name] = json.load(f)
            print(f"✅ 加载 {model_name} 结果成功")
        except Exception as e:
            print(f"❌ 加载 {model_name} 结果失败: {e}")
    
    def compare_metrics(self):
        """对比核心指标"""
        if len(self.results) < 2:
            print("⚠️ 需要至少两个模型的结果进行对比")
            return
            
        print("\n" + "=" * 80)
        print("📊 性能指标对比分析")
        print("=" * 80)
        
        # 提取指标
        metrics_comparison = {}
        model_names = list(self.results.keys())
        
        # 核心指标列表
        core_metrics = ['rmse', 'mae', 'precision_at_10', 'ndcg_at_10', 'coverage', 'diversity']
        
        for metric in core_metrics:
            metrics_comparison[metric] = {}
            for model_name in model_names:
                try:
                    value = self.results[model_name]['test_metrics'][metric]
                    metrics_comparison[metric][model_name] = value
                except KeyError:
                    print(f"⚠️ {model_name} 缺少指标: {metric}")
                    metrics_comparison[metric][model_name] = None
        
        # 生成对比表格
        self._print_comparison_table(metrics_comparison, model_names)
        
        # 计算改进率
        self._calculate_improvements(metrics_comparison, model_names)
        
        return metrics_comparison
    
    def _print_comparison_table(self, metrics_comparison: Dict, model_names: list):
        """打印对比表格"""
        print(f"\n{'指标':<15}", end="")
        for name in model_names:
            print(f"{name:<20}", end="")
        print("改进率")
        print("-" * (15 + 20 * len(model_names) + 10))
        
        for metric, values in metrics_comparison.items():
            print(f"{metric:<15}", end="")
            metric_values = []
            for name in model_names:
                value = values.get(name)
                if value is not None:
                    print(f"{value:<20.4f}", end="")
                    metric_values.append(value)
                else:
                    print(f"{'N/A':<20}", end="")
            
            # 计算改进率（假设第一个是基准，第二个是改进版）
            if len(metric_values) >= 2:
                if metric in ['rmse', 'mae']:  # 越小越好
                    improvement = (metric_values[0] - metric_values[1]) / metric_values[0] * 100
                else:  # 越大越好
                    improvement = (metric_values[1] - metric_values[0]) / metric_values[0] * 100
                print(f"{improvement:+.2f}%")
            else:
                print("N/A")
    
    def _calculate_improvements(self, metrics_comparison: Dict, model_names: list):
        """计算详细改进分析"""
        if len(model_names) != 2:
            return
            
        baseline_name = model_names[0]
        enhanced_name = model_names[1]
        
        print(f"\n🔍 详细改进分析 ({baseline_name} → {enhanced_name}):")
        print("-" * 60)
        
        significant_improvements = []
        minor_improvements = []
        regressions = []
        
        for metric, values in metrics_comparison.items():
            baseline_val = values.get(baseline_name)
            enhanced_val = values.get(enhanced_name)
            
            if baseline_val is not None and enhanced_val is not None:
                if metric in ['rmse', 'mae']:  # 越小越好
                    improvement = (baseline_val - enhanced_val) / baseline_val * 100
                else:  # 越大越好
                    improvement = (enhanced_val - baseline_val) / baseline_val * 100
                
                if improvement > 5:
                    significant_improvements.append((metric, improvement))
                elif improvement > 0:
                    minor_improvements.append((metric, improvement))
                else:
                    regressions.append((metric, improvement))
        
        if significant_improvements:
            print("🚀 显著改进 (>5%):")
            for metric, imp in significant_improvements:
                print(f"  - {metric}: {imp:+.2f}%")
        
        if minor_improvements:
            print("📈 轻微改进 (0-5%):")
            for metric, imp in minor_improvements:
                print(f"  - {metric}: {imp:+.2f}%")
        
        if regressions:
            print("📉 性能下降:")
            for metric, imp in regressions:
                print(f"  - {metric}: {imp:+.2f}%")
    
    def compare_model_complexity(self):
        """对比模型复杂度"""
        print("\n" + "=" * 80)
        print("🔧 模型复杂度对比")
        print("=" * 80)
        
        print(f"{'模型':<25} {'参数量':<15} {'可训练参数':<15} {'相对复杂度'}")
        print("-" * 70)
        
        param_counts = []
        for name, result in self.results.items():
            total_params = result['model_info']['total_parameters']
            trainable_params = result['model_info']['trainable_parameters']
            param_counts.append(total_params)
            
            print(f"{name:<25} {total_params:<15,} {trainable_params:<15,}", end="")
            
        # 计算相对复杂度
        if len(param_counts) >= 2:
            baseline_params = param_counts[0]
            for i, params in enumerate(param_counts):
                relative_complexity = params / baseline_params
                print(f" {relative_complexity:.2f}x")
        else:
            print(" 1.0x")
    
    def analyze_training_efficiency(self):
        """分析训练效率"""
        print("\n" + "=" * 80)
        print("⚡ 训练效率分析")
        print("=" * 80)
        
        for name, result in self.results.items():
            if 'training_history' in result:
                history = result['training_history']
                
                print(f"\n{name}:")
                print(f"  - 最终训练损失: {history['train_loss'][-1]:.4f}")
                print(f"  - 最终验证RMSE: {history['val_rmse'][-1]:.4f}")
                print(f"  - 最佳验证RMSE: {min(history['val_rmse']):.4f}")
                
                # 收敛分析
                val_rmse = history['val_rmse']
                if len(val_rmse) >= 5:
                    last_5_std = np.std(val_rmse[-5:])
                    if last_5_std < 0.01:
                        print(f"  - 收敛状态: ✅ 已收敛 (std={last_5_std:.4f})")
                    else:
                        print(f"  - 收敛状态: ⚠️ 未完全收敛 (std={last_5_std:.4f})")
    
    def generate_visual_comparison(self, save_dir: str):
        """生成可视化对比图"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 核心指标雷达图
        self._create_radar_chart(save_dir)
        
        # 2. 参数量对比柱状图
        self._create_complexity_chart(save_dir)
        
        # 3. 训练历史对比
        self._create_training_history_chart(save_dir)
        
        print(f"\n📊 可视化图表已保存到: {save_dir}")
    
    def _create_radar_chart(self, save_dir: str):
        """创建雷达图对比核心指标"""
        if len(self.results) < 2:
            return
        
        metrics = ['rmse', 'mae', 'precision_at_10', 'ndcg_at_10', 'coverage', 'diversity']
        model_names = list(self.results.keys())
        
        # 提取数据并归一化
        data = {}
        for name in model_names:
            values = []
            for metric in metrics:
                try:
                    value = self.results[name]['test_metrics'][metric]
                    # 对RMSE和MAE取倒数，使得所有指标都是越大越好
                    if metric in ['rmse', 'mae']:
                        value = 1 / (1 + value)
                    values.append(value)
                except:
                    values.append(0)
            data[name] = values
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for name, values in data.items():
            values += values[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title('模型性能雷达图对比', size=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_complexity_chart(self, save_dir: str):
        """创建模型复杂度对比图"""
        model_names = list(self.results.keys())
        param_counts = []
        
        for name in model_names:
            params = self.results[name]['model_info']['total_parameters']
            param_counts.append(params / 1000)  # 转换为K
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, param_counts, color=['#FF9999', '#66B2FF'])
        
        # 添加数值标签
        for bar, count in zip(bars, param_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_counts)*0.01,
                    f'{count:.1f}K', ha='center', va='bottom', fontweight='bold')
        
        plt.title('模型参数量对比', fontsize=16, fontweight='bold')
        plt.ylabel('参数量 (K)', fontsize=12)
        plt.xlabel('模型', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_complexity_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_training_history_chart(self, save_dir: str):
        """创建训练历史对比图"""
        plt.figure(figsize=(15, 5))
        
        # 子图1: 训练损失
        plt.subplot(1, 3, 1)
        for name, result in self.results.items():
            if 'training_history' in result:
                history = result['training_history']
                epochs = range(1, len(history['train_loss']) + 1)
                plt.plot(epochs, history['train_loss'], 'o-', label=name, linewidth=2)
        
        plt.title('训练损失', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('训练损失')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 子图2: 验证RMSE
        plt.subplot(1, 3, 2)
        for name, result in self.results.items():
            if 'training_history' in result:
                history = result['training_history']
                epochs = range(1, len(history['val_rmse']) + 1)
                plt.plot(epochs, history['val_rmse'], 's-', label=name, linewidth=2)
        
        plt.title('验证RMSE', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('验证RMSE')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 子图3: 验证MAE
        plt.subplot(1, 3, 3)
        for name, result in self.results.items():
            if 'training_history' in result:
                history = result['training_history']
                epochs = range(1, len(history['val_mae']) + 1)
                plt.plot(epochs, history['val_mae'], '^-', label=name, linewidth=2)
        
        plt.title('验证MAE', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('验证MAE')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, output_file: str):
        """生成综合对比报告"""
        metrics_comparison = self.compare_metrics()
        
        report_content = []
        report_content.append("# LayerwiseAdapter vs Ensemble Baseline 综合性能对比报告\n")
        report_content.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_content.append("=" * 80 + "\n")
        
        # 执行总结
        report_content.append("## 执行总结\n")
        if len(self.results) >= 2:
            model_names = list(self.results.keys())
            baseline = model_names[0]
            enhanced = model_names[1]
            
            # 计算关键改进
            rmse_baseline = self.results[baseline]['test_metrics']['rmse']
            rmse_enhanced = self.results[enhanced]['test_metrics']['rmse']
            rmse_improvement = (rmse_baseline - rmse_enhanced) / rmse_baseline * 100
            
            param_baseline = self.results[baseline]['model_info']['total_parameters']
            param_enhanced = self.results[enhanced]['model_info']['total_parameters']
            complexity_ratio = param_enhanced / param_baseline
            
            report_content.append(f"本报告对比了{baseline}与{enhanced}在MovieLens 1M数据集上的性能表现。\n")
            report_content.append(f"主要发现：\n")
            report_content.append(f"- RMSE改进: {rmse_improvement:+.2f}%\n")
            report_content.append(f"- 模型复杂度比: {complexity_ratio:.2f}x\n")
            
            if rmse_improvement > 0:
                report_content.append(f"- 结论: {enhanced}在预测精度上优于{baseline}\n")
            
        report_content.append("\n## 详细性能对比\n")
        
        # 详细指标表格
        report_content.append("### 核心评估指标\n")
        report_content.append("| 指标 | " + " | ".join(self.results.keys()) + " | 改进率 |\n")
        report_content.append("|------|" + "|".join(["----"] * len(self.results)) + "|------|\n")
        
        for metric in ['rmse', 'mae', 'precision_at_10', 'ndcg_at_10', 'coverage', 'diversity']:
            if metrics_comparison and metric in metrics_comparison:
                values = metrics_comparison[metric]
                row = f"| {metric} |"
                metric_values = []
                for name in self.results.keys():
                    value = values.get(name)
                    if value is not None:
                        row += f" {value:.4f} |"
                        metric_values.append(value)
                    else:
                        row += " N/A |"
                
                # 计算改进率
                if len(metric_values) >= 2:
                    if metric in ['rmse', 'mae']:
                        improvement = (metric_values[0] - metric_values[1]) / metric_values[0] * 100
                    else:
                        improvement = (metric_values[1] - metric_values[0]) / metric_values[0] * 100
                    row += f" {improvement:+.2f}% |"
                else:
                    row += " N/A |"
                
                report_content.append(row + "\n")
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(report_content)
        
        print(f"📝 综合对比报告已保存到: {output_file}")

def main():
    """主函数"""
    print("=" * 80)
    print("🔍 LayerwiseAdapter vs Ensemble Baseline 性能对比分析")
    print("=" * 80)
    
    comparator = PerformanceComparator()
    
    # 加载结果文件
    baseline_results = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/baselines/simple_ensemble_baseline_results.json"
    layerwise_results = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/layerwise_adapter/full_training_results.json"
    
    # 尝试加载可用的结果
    if os.path.exists(baseline_results):
        comparator.load_results("Ensemble Baseline", baseline_results)
    else:
        print(f"⚠️ 基准结果文件不存在: {baseline_results}")
        print("请先运行 ensemble_baseline_1m.py 生成基准结果")
    
    if os.path.exists(layerwise_results):
        comparator.load_results("LayerwiseAdapter Enhanced", layerwise_results)
    else:
        print(f"⚠️ LayerwiseAdapter结果文件不存在: {layerwise_results}")
        print("请先运行 full_training_evaluation_corrected.py 生成结果")
    
    # 如果有结果，进行对比分析
    if len(comparator.results) >= 2:
        # 执行对比分析
        comparator.compare_metrics()
        comparator.compare_model_complexity()
        comparator.analyze_training_efficiency()
        
        # 生成可视化图表
        viz_dir = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/comparative_analysis/visualizations"
        comparator.generate_visual_comparison(viz_dir)
        
        # 生成综合报告
        report_file = "/home/coder-gw/7Projects_in_7Days/online-inference-system/layerwise_adapter_enhanced/experiments/comparative_analysis/comprehensive_comparison_report.md"
        comparator.generate_comprehensive_report(report_file)
        
        print("\n✅ 性能对比分析完成!")
    else:
        print("❌ 缺少必要的结果文件，无法进行对比分析")
        print("请确保已运行相关训练脚本生成结果文件")

if __name__ == "__main__":
    import time
    main()
