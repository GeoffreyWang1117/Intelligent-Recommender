#!/usr/bin/env python3
"""
测试Optimized Ensemble Teacher的基本功能
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from models.optimized_ensemble_teacher import OptimizedEnsembleTeacher
    print("✅ 成功导入 OptimizedEnsembleTeacher")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_ensemble_initialization():
    """测试集成教师模型初始化"""
    print("\n🔧 测试1: 初始化...")
    
    try:
        teacher = OptimizedEnsembleTeacher()
        print("✅ 初始化成功")
        
        # 检查配置
        print(f"📊 算法配置: {list(teacher.algorithm_config.keys())}")
        print(f"📊 权重分配: SVD={teacher.algorithm_config['svd']['weight']}, "
              f"xDeepFM={teacher.algorithm_config['xdeepfm']['weight']}, "
              f"AutoInt={teacher.algorithm_config['autoint']['weight']}")
        
        return teacher
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return None

def test_model_loading(teacher):
    """测试模型加载"""
    print("\n📥 测试2: 模型加载...")
    
    try:
        # 检查模型文件是否存在
        model_files = {
            'svd': 'models/saved/SVD_real_movielens.pkl',
            'xdeepfm': 'models/saved/xDeepFM_real_movielens.pkl', 
            'autoint': 'models/saved/AutoInt_real_movielens.pkl'
        }
        
        for algo, path in model_files.items():
            if os.path.exists(path):
                print(f"✅ {algo}模型文件存在: {path}")
            else:
                print(f"❌ {algo}模型文件不存在: {path}")
        
        # 尝试加载SVD模型（最稳定的）
        svd_path = 'models/saved/SVD_real_movielens.pkl'
        if os.path.exists(svd_path):
            with open(svd_path, 'rb') as f:
                svd_model = pickle.load(f)
            print(f"✅ SVD模型加载成功，类型: {type(svd_model)}")
            return True
        else:
            print("❌ SVD模型文件不存在")
            return False
            
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return False

def test_performance_summary(teacher):
    """测试性能摘要"""
    print("\n📊 测试3: 性能摘要...")
    
    try:
        summary = teacher.get_model_performance_summary()
        print("✅ 性能摘要生成成功")
        
        print("📈 集成策略:")
        for key, value in summary['ensemble_strategy'].items():
            print(f"   {key}: {value}")
            
        print("📈 预期性能:")
        for key, value in summary['expected_performance'].items():
            print(f"   {key}: {value}")
            
        return True
    except Exception as e:
        print(f"❌ 性能摘要失败: {e}")
        return False

def test_ensemble_weights(teacher):
    """测试集成权重设计"""
    print("\n⚖️ 测试4: 集成权重分析...")
    
    try:
        total_weight = sum(
            teacher.algorithm_config[algo]['weight'] 
            for algo in ['svd', 'xdeepfm', 'autoint']
        )
        print(f"✅ 总权重: {total_weight} (应该等于1.0)")
        
        if abs(total_weight - 1.0) < 0.001:
            print("✅ 权重分配正确")
        else:
            print("⚠️ 权重分配可能有问题")
            
        # 分析权重合理性
        print("📊 权重分配理由:")
        for algo in ['svd', 'xdeepfm', 'autoint']:
            config = teacher.algorithm_config[algo]
            print(f"   {algo.upper()}: {config['weight']} - {config['primary_task']}")
            if 'performance' in config:
                perf = config['performance']
                print(f"      性能: Recall@10={perf.get('recall_10', 'N/A')}, "
                      f"RMSE={perf.get('rmse', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ 权重分析失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 开始测试 Optimized Ensemble Teacher")
    print("="*60)
    
    # 测试1: 初始化
    teacher = test_ensemble_initialization()
    if not teacher:
        print("❌ 初始化失败，终止测试")
        return
    
    # 测试2: 模型加载
    model_loading_success = test_model_loading(teacher)
    
    # 测试3: 性能摘要
    summary_success = test_performance_summary(teacher)
    
    # 测试4: 权重分析
    weights_success = test_ensemble_weights(teacher)
    
    # 总结
    print("\n" + "="*60)
    print("🎯 测试总结:")
    print(f"   初始化: {'✅' if teacher else '❌'}")
    print(f"   模型加载: {'✅' if model_loading_success else '❌'}")
    print(f"   性能摘要: {'✅' if summary_success else '❌'}")
    print(f"   权重分析: {'✅' if weights_success else '❌'}")
    
    if all([teacher, model_loading_success, summary_success, weights_success]):
        print("\n🎉 所有测试通过！Ensemble Teacher 基础功能正常")
        print("📝 下一步: 可以进行Fisher Information分析和PAKD实验")
    else:
        print("\n⚠️ 部分测试失败，需要进一步调试")

if __name__ == "__main__":
    main()
