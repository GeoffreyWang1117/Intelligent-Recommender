"""
评估结果索引文件
提供快速访问所有评估结果的接口
"""

import os
import json

class EvaluationResultsIndex:
    """评估结果索引"""
    
    def __init__(self, results_dir="evaluation_results"):
        self.results_dir = results_dir
        
    def list_all_files(self):
        """列出所有评估结果文件"""
        files = []
        if os.path.exists(self.results_dir):
            for file in os.listdir(self.results_dir):
                file_path = os.path.join(self.results_dir, file)
                files.append({
                    'name': file,
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'type': self._get_file_type(file)
                })
        return files
    
    def _get_file_type(self, filename):
        """获取文件类型"""
        if filename.endswith('.png'):
            return 'Image/Chart'
        elif filename.endswith('.csv'):
            return 'Data/CSV'
        elif filename.endswith('.json'):
            return 'Data/JSON'
        elif filename.endswith('.html'):
            return 'Report/HTML'
        elif filename.endswith('.md'):
            return 'Report/Markdown'
        else:
            return 'Other'
    
    def get_model_ranking(self):
        """获取模型排名"""
        json_file = os.path.join(self.results_dir, "model_evaluation_summary.json")
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 按性能评分排序
            ranking = []
            for model, metrics in data.items():
                ranking.append({
                    'model': model,
                    'rmse': metrics.get('RMSE', 0),
                    'mae': metrics.get('MAE', 0),
                    'grade': metrics.get('Performance_Grade', 'N/A'),
                    'training_time': metrics.get('Training_Time', 0)
                })
            
            ranking.sort(key=lambda x: x['rmse'])
            return ranking
        return []
    
    def print_summary(self):
        """打印评估结果总结"""
        print("\n" + "="*60)
        print("📊 SOTA推荐模型评估结果总结")
        print("="*60)
        
        # 文件列表
        files = self.list_all_files()
        print(f"\n📁 生成文件 ({len(files)}个):")
        for file_info in files:
            print(f"  📄 {file_info['name']} ({file_info['type']}) - {file_info['size']} bytes")
        
        # 模型排名
        ranking = self.get_model_ranking()
        if ranking:
            print(f"\n🏆 模型性能排名:")
            for i, model_info in enumerate(ranking):
                print(f"  {i+1}. {model_info['model']} - Grade {model_info['grade']} "
                      f"(RMSE: {model_info['rmse']:.4f}, Training: {model_info['training_time']:.1f}s)")
        
        print(f"\n✅ 评估完成！最佳模型: {ranking[0]['model'] if ranking else 'N/A'}")
        print(f"📊 查看详细报告: {self.results_dir}/model_evaluation_summary.html")
        print(f"📈 可视化图表: {self.results_dir}/*.png")
        print(f"📋 详细报告: {self.results_dir}/EVALUATION_REPORT.md")


if __name__ == "__main__":
    index = EvaluationResultsIndex()
    index.print_summary()
