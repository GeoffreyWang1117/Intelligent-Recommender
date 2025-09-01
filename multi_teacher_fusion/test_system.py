"""
Multi-Teacher Fusion System - Quick Test
快速测试整个系统的基本功能
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from multi_teacher_fusion.config.base_config import SystemConfig, DatasetConfig
from multi_teacher_fusion.utils.data_processor import MovieLensProcessor
from multi_teacher_fusion.utils.evaluator import RecommendationEvaluator
from multi_teacher_fusion.teachers.ensemble_teacher import EnsembleTeacher

def test_data_processing():
    """测试数据处理功能"""
    print("🔄 测试数据处理...")
    
    try:
        # 检查数据路径
        data_path = project_root / "data" / "movielens"
        if not data_path.exists():
            print(f"❌ 数据路径不存在: {data_path}")
            print("请先下载MovieLens数据集")
            return False
        
        # 创建数据处理器
        processor = MovieLensProcessor(
            data_path=str(data_path),
            min_ratings_per_user=10,  # 降低阈值以便测试
            min_ratings_per_item=5
        )
        
        # 测试加载原始数据
        if not processor.load_raw_data():
            print("❌ 数据加载失败")
            return False
        
        print(f"✅ 数据加载成功: {len(processor.ratings_df)} 条评分")
        
        # 测试基本统计
        if processor.ratings_df is not None:
            print(f"   用户数: {processor.ratings_df['userId'].nunique()}")
            print(f"   物品数: {processor.ratings_df['movieId'].nunique()}")
            print(f"   评分范围: {processor.ratings_df['rating'].min()}-{processor.ratings_df['rating'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据处理测试失败: {e}")
        return False

def test_ensemble_teacher():
    """测试Ensemble Teacher"""
    print("🤖 测试Ensemble Teacher...")
    
    try:
        from multi_teacher_fusion.config.teacher_config import EnsembleConfig
        
        # 创建配置
        config = EnsembleConfig(
            svd_weight=0.5,
            xdeepfm_weight=0.3,
            autoint_weight=0.2,
            num_factors=20,  # 减小参数以便快速测试
            num_epochs=5,
            learning_rate=0.01
        )
        
        # 创建Ensemble Teacher
        teacher = EnsembleTeacher(config=config)
        
        print(f"✅ Ensemble Teacher 创建成功")
        print(f"   配置: SVD({config.svd_weight}), xDeepFM({config.xdeepfm_weight}), AutoInt({config.autoint_weight})")
        
        # 测试基本接口
        if hasattr(teacher, 'get_user_profile'):
            print("✅ 用户画像接口可用")
        
        if hasattr(teacher, 'get_item_profile'):
            print("✅ 物品画像接口可用")
        
        if hasattr(teacher, 'generate_recommendations'):
            print("✅ 推荐生成接口可用")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble Teacher 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluator():
    """测试评估器"""
    print("📊 测试评估器...")
    
    try:
        evaluator = RecommendationEvaluator(k_values=[5, 10])
        
        # 创建测试数据
        import numpy as np
        np.random.seed(42)
        
        y_true = np.random.uniform(1, 5, 100)
        y_pred = y_true + np.random.normal(0, 0.5, 100)
        
        # 测试RMSE
        rmse = evaluator.rmse(y_true, y_pred)
        mae = evaluator.mae(y_true, y_pred)
        
        print(f"✅ 评估器工作正常")
        print(f"   测试RMSE: {rmse:.4f}")
        print(f"   测试MAE: {mae:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估器测试失败: {e}")
        return False

def test_config_system():
    """测试配置系统"""
    print("⚙️ 测试配置系统...")
    
    try:
        # 测试系统配置
        system_config = SystemConfig()
        print(f"✅ 系统配置加载成功")
        print(f"   日志级别: {system_config.log_level}")
        print(f"   使用CUDA: {system_config.use_cuda}")
        
        # 测试数据集配置
        dataset_config = DatasetConfig()
        print(f"✅ 数据集配置加载成功")
        print(f"   数据集名称: {dataset_config.name}")
        print(f"   批次大小: {dataset_config.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("🧪 Multi-Teacher Fusion System - Quick Test")
    print("="*60)
    
    tests = [
        ("配置系统", test_config_system),
        ("数据处理", test_data_processing),
        ("评估器", test_evaluator),
        ("Ensemble Teacher", test_ensemble_teacher),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}测试:")
        print("-" * 40)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {e}")
            results[test_name] = False
    
    print("\n" + "="*60)
    print("📋 测试结果汇总:")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有测试通过! 系统基础功能正常")
    else:
        print("⚠️ 部分测试失败，请检查相关模块")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.WARNING)  # 减少日志输出
    
    # 运行测试
    success = run_all_tests()
    
    if success:
        print("\n🚀 系统准备就绪，可以开始实验!")
        print("下一步建议:")
        print("1. 确保MovieLens数据集已下载")
        print("2. 运行完整的训练实验")
        print("3. 查看实验结果和评估报告")
    else:
        print("\n🔧 请先修复上述问题再继续")
    
    exit(0 if success else 1)
