#!/usr/bin/env python3
"""
推荐系统算法测试套件

测试所有推荐算法的基本功能，包括训练、预测、推荐等核心功能。
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from typing import List, Dict

from models import (
    create_recommender, 
    get_available_algorithms,
    ExplainableRecommender
)


class TestRecommenderAlgorithms(unittest.TestCase):
    """推荐算法测试类"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试数据"""
        # 创建测试数据
        np.random.seed(42)
        cls.test_data = pd.DataFrame({
            'user_id': np.random.randint(1, 50, 200),
            'item_id': np.random.randint(1, 30, 200),
            'rating': np.random.uniform(1, 5, 200)
        })
        
        # 创建小规模测试数据（用于快速测试）
        cls.small_test_data = pd.DataFrame({
            'user_id': np.random.randint(1, 10, 50),
            'item_id': np.random.randint(1, 15, 50),
            'rating': np.random.uniform(1, 5, 50)
        })
    
    def test_available_algorithms(self):
        """测试算法工厂是否正确注册了所有算法"""
        algorithms = get_available_algorithms()
        
        # 验证核心算法都已注册
        expected_algorithms = [
            'svd', 'deepfm', 'simple_deepfm', 
            'autoint', 'simple_autoint',
            'din', 'simple_din'
        ]
        
        for algo in expected_algorithms:
            self.assertIn(algo, algorithms, f"算法 {algo} 未正确注册")
        
        # 验证至少有基本数量的算法
        self.assertGreaterEqual(len(algorithms), 7, "注册的算法数量不足")
    
    def test_svd_algorithm(self):
        """测试SVD算法"""
        model = create_recommender('svd', n_components=10)
        self.assertIsNotNone(model, "SVD模型创建失败")
        
        # 测试训练
        model.fit(self.small_test_data)
        self.assertTrue(model.is_trained, "SVD模型训练失败")
        
        # 测试预测
        prediction = model.predict(1, 1)
        self.assertIsInstance(prediction, (int, float), "SVD预测结果类型错误")
        self.assertGreaterEqual(prediction, 0, "SVD预测结果不应为负数")
        
        # 测试推荐
        recommendations = model.get_user_recommendations(1, top_k=5)
        self.assertIsInstance(recommendations, list, "SVD推荐结果应为列表")
        self.assertLessEqual(len(recommendations), 5, "SVD推荐数量超过限制")
    
    def test_simple_deepfm_algorithm(self):
        """测试简化版DeepFM算法"""
        model = create_recommender('simple_deepfm')
        self.assertIsNotNone(model, "SimpleDeepFM模型创建失败")
        
        # 测试训练
        model.fit(self.small_test_data)
        self.assertTrue(model.is_trained, "SimpleDeepFM模型训练失败")
        
        # 测试预测
        prediction = model.predict(1, 1)
        self.assertIsInstance(prediction, (int, float), "SimpleDeepFM预测结果类型错误")
        
        # 测试推荐
        recommendations = model.get_user_recommendations(1, top_k=3)
        self.assertIsInstance(recommendations, list, "SimpleDeepFM推荐结果应为列表")
        
        if len(recommendations) > 0:
            rec = recommendations[0]
            self.assertIn('item_id', rec, "推荐结果缺少item_id")
            self.assertIn('score', rec, "推荐结果缺少score")
    
    def test_pytorch_algorithms(self):
        """测试PyTorch版本算法"""
        pytorch_algorithms = ['deepfm', 'autoint', 'din']
        
        for algo_name in pytorch_algorithms:
            with self.subTest(algorithm=algo_name):
                try:
                    # 使用较小的参数进行快速测试
                    model = create_recommender(
                        algo_name, 
                        embedding_dim=4, 
                        epochs=1, 
                        batch_size=16
                    )
                    self.assertIsNotNone(model, f"{algo_name}模型创建失败")
                    
                    # 训练模型
                    model.fit(self.small_test_data)
                    self.assertTrue(model.is_trained, f"{algo_name}模型训练失败")
                    
                    # 测试预测
                    prediction = model.predict(1, 1)
                    self.assertIsInstance(prediction, (int, float), f"{algo_name}预测结果类型错误")
                    
                except ImportError:
                    self.skipTest(f"PyTorch不可用，跳过{algo_name}测试")
                except Exception as e:
                    self.fail(f"{algo_name}算法测试失败: {str(e)}")
    
    def test_ensemble_recommender(self):
        """测试集成推荐器"""
        # 使用简化版本算法进行测试（更稳定）
        algorithms = ['simple_deepfm', 'simple_autoint']
        
        ensemble = ExplainableRecommender(algorithms)
        self.assertEqual(ensemble.algorithm_names, algorithms, "集成推荐器算法列表不正确")
        
        # 测试训练
        ensemble.fit(self.small_test_data)
        self.assertTrue(ensemble.is_trained, "集成推荐器训练失败")
        self.assertGreater(len(ensemble.models), 0, "集成推荐器未成功训练任何模型")
        
        # 测试预测
        prediction = ensemble.predict(1, 1)
        self.assertIsInstance(prediction, (int, float), "集成推荐器预测结果类型错误")
        
        # 测试推荐
        recommendations = ensemble.get_user_recommendations(1, top_k=3)
        self.assertIsInstance(recommendations, list, "集成推荐器推荐结果应为列表")
        
        if len(recommendations) > 0:
            rec = recommendations[0]
            # 验证推荐结果格式
            self.assertIn('item_id', rec, "推荐结果缺少item_id")
            self.assertIn('score', rec, "推荐结果缺少score")
            self.assertIn('explanation', rec, "推荐结果缺少explanation")
            self.assertIn('ensemble_details', rec, "推荐结果缺少ensemble_details")
    
    def test_model_persistence(self):
        """测试模型保存和加载"""
        model = create_recommender('simple_deepfm')
        model.fit(self.small_test_data)
        
        # 测试保存
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            save_path = tmp_file.name
        
        try:
            success = model.save_model(save_path)
            self.assertTrue(success, "模型保存失败")
            self.assertTrue(os.path.exists(save_path), "模型文件未创建")
            
            # 测试加载
            new_model = create_recommender('simple_deepfm')
            load_success = new_model.load_model(save_path)
            self.assertTrue(load_success, "模型加载失败")
            
            # 验证加载后的模型功能
            original_pred = model.predict(1, 1)
            loaded_pred = new_model.predict(1, 1)
            self.assertAlmostEqual(original_pred, loaded_pred, places=2, 
                                 msg="加载后的模型预测结果不一致")
            
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
    
    def test_algorithm_parameters(self):
        """测试算法参数设置"""
        # 测试DeepFM参数
        model1 = create_recommender('simple_deepfm', learning_rate=0.01)
        model2 = create_recommender('simple_deepfm', learning_rate=0.1)
        
        # 参数应该正确设置
        self.assertEqual(model1.learning_rate, 0.01)
        self.assertEqual(model2.learning_rate, 0.1)
    
    def test_invalid_algorithm(self):
        """测试无效算法名称"""
        with self.assertRaises(ValueError):
            create_recommender('invalid_algorithm')
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        empty_data = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
        
        model = create_recommender('simple_deepfm')
        
        # 空数据应该抛出异常或返回合理的错误
        with self.assertRaises((ValueError, IndexError)):
            model.fit(empty_data)
    
    def test_recommendation_format(self):
        """测试推荐结果格式"""
        model = create_recommender('simple_deepfm')
        model.fit(self.small_test_data)
        
        recommendations = model.get_user_recommendations(1, top_k=5)
        
        for rec in recommendations:
            # 验证必需字段
            self.assertIn('item_id', rec, "推荐结果缺少item_id字段")
            self.assertIn('score', rec, "推荐结果缺少score字段")
            
            # 验证数据类型
            self.assertIsInstance(rec['item_id'], (int, np.integer), "item_id应为整数")
            self.assertIsInstance(rec['score'], (int, float, np.number), "score应为数值")
            
            # 验证数据范围
            self.assertGreater(rec['item_id'], 0, "item_id应为正数")
    
    def test_concurrent_training(self):
        """测试并发训练安全性"""
        models = []
        
        # 创建多个模型实例
        for i in range(3):
            model = create_recommender('simple_deepfm')
            models.append(model)
        
        # 同时训练（模拟并发）
        for model in models:
            model.fit(self.small_test_data)
        
        # 验证所有模型都训练成功
        for i, model in enumerate(models):
            self.assertTrue(model.is_trained, f"模型{i}训练失败")
            
            # 验证各模型独立性
            prediction = model.predict(1, 1)
            self.assertIsInstance(prediction, (int, float), f"模型{i}预测失败")


class TestAlgorithmPerformance(unittest.TestCase):
    """算法性能测试类"""
    
    @classmethod
    def setUpClass(cls):
        """设置性能测试数据"""
        np.random.seed(42)
        cls.large_test_data = pd.DataFrame({
            'user_id': np.random.randint(1, 100, 1000),
            'item_id': np.random.randint(1, 200, 1000),
            'rating': np.random.uniform(1, 5, 1000)
        })
    
    def test_training_speed(self):
        """测试训练速度"""
        import time
        
        algorithms_to_test = ['simple_deepfm', 'simple_autoint']
        
        for algo_name in algorithms_to_test:
            with self.subTest(algorithm=algo_name):
                model = create_recommender(algo_name)
                
                start_time = time.time()
                model.fit(self.large_test_data)
                training_time = time.time() - start_time
                
                # 训练时间不应超过合理限制（30秒）
                self.assertLess(training_time, 30, f"{algo_name}训练时间过长: {training_time:.2f}秒")
    
    def test_prediction_speed(self):
        """测试预测速度"""
        import time
        
        model = create_recommender('simple_deepfm')
        model.fit(self.large_test_data)
        
        # 测试单次预测速度
        start_time = time.time()
        for _ in range(100):
            model.predict(1, 1)
        prediction_time = (time.time() - start_time) / 100
        
        # 单次预测时间不应超过10ms
        self.assertLess(prediction_time, 0.01, f"预测速度过慢: {prediction_time*1000:.2f}ms")
    
    def test_memory_usage(self):
        """测试内存使用"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 训练多个模型
            models = []
            for _ in range(3):
                model = create_recommender('simple_deepfm')
                model.fit(self.large_test_data)
                models.append(model)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # 内存增长不应超过500MB
            self.assertLess(memory_increase, 500, f"内存使用过多: {memory_increase:.2f}MB")
        except ImportError:
            self.skipTest("psutil不可用，跳过内存测试")


def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_suite.addTest(unittest.makeSuite(TestRecommenderAlgorithms))
    test_suite.addTest(unittest.makeSuite(TestAlgorithmPerformance))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # 直接运行测试
    success = run_all_tests()
    exit(0 if success else 1)
