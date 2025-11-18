#!/usr/bin/env python3
"""
集成测试套件

测试系统各组件的集成和端到端功能。
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_recommender, ExplainableRecommender
from services.cache import CacheService


class TestEndToEndRecommendation(unittest.TestCase):
    """端到端推荐测试"""

    @classmethod
    def setUpClass(cls):
        """准备测试数据"""
        np.random.seed(42)
        cls.ratings_data = pd.DataFrame({
            'user_id': np.random.randint(1, 20, 100),
            'item_id': np.random.randint(1, 30, 100),
            'rating': np.random.uniform(1, 5, 100)
        })

    def test_single_model_workflow(self):
        """测试单模型完整工作流"""
        # 1. 创建模型
        model = create_recommender('simple_deepfm', learning_rate=0.05)
        self.assertIsNotNone(model)

        # 2. 训练模型
        model.fit(self.ratings_data)
        self.assertTrue(model.is_trained)

        # 3. 预测评分
        prediction = model.predict(user_id=1, item_id=1)
        self.assertIsInstance(prediction, (int, float))
        self.assertGreater(prediction, 0)
        self.assertLess(prediction, 10)

        # 4. 获取推荐
        recommendations = model.get_user_recommendations(user_id=1, top_k=5)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # 5. 验证推荐格式
        for rec in recommendations:
            self.assertIn('item_id', rec)
            self.assertIn('score', rec)

        # 6. 保存模型
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            save_path = tmp.name

        try:
            save_success = model.save_model(save_path)
            self.assertTrue(save_success)
            self.assertTrue(os.path.exists(save_path))

            # 7. 加载模型
            new_model = create_recommender('simple_deepfm')
            load_success = new_model.load_model(save_path)
            self.assertTrue(load_success)

            # 8. 验证加载后的模型
            new_prediction = new_model.predict(user_id=1, item_id=1)
            self.assertAlmostEqual(prediction, new_prediction, places=2)

        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)

    def test_ensemble_workflow(self):
        """测试集成推荐器完整工作流"""
        # 1. 创建集成推荐器
        algorithms = ['simple_deepfm', 'simple_autoint']
        ensemble = ExplainableRecommender(algorithms)

        # 2. 训练
        ensemble.fit(self.ratings_data)
        self.assertTrue(ensemble.is_trained)

        # 3. 验证所有模型都训练成功
        self.assertEqual(len(ensemble.models), len(algorithms))

        # 4. 获取推荐
        recommendations = ensemble.get_user_recommendations(user_id=1, top_k=5)
        self.assertIsInstance(recommendations, list)

        # 5. 验证推荐包含解释信息
        if len(recommendations) > 0:
            rec = recommendations[0]
            self.assertIn('explanation', rec)
            self.assertIn('ensemble_details', rec)

    def test_multiple_users_recommendations(self):
        """测试多用户推荐"""
        model = create_recommender('simple_deepfm')
        model.fit(self.ratings_data)

        user_ids = [1, 2, 3, 5, 10]

        for user_id in user_ids:
            with self.subTest(user_id=user_id):
                recommendations = model.get_user_recommendations(user_id, top_k=3)
                self.assertIsInstance(recommendations, list)

    def test_cold_start_handling(self):
        """测试冷启动处理"""
        model = create_recommender('simple_deepfm')
        model.fit(self.ratings_data)

        # 测试新用户（系统中不存在）
        new_user_id = 9999
        recommendations = model.get_user_recommendations(new_user_id, top_k=5)

        # 应该返回默认推荐或空列表
        self.assertIsInstance(recommendations, list)


class TestDataPipeline(unittest.TestCase):
    """数据处理流程测试"""

    def test_data_validation(self):
        """测试数据验证"""
        # 正常数据
        valid_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [1, 2, 3],
            'rating': [4.0, 3.5, 5.0]
        })

        model = create_recommender('simple_deepfm')
        # 应该成功训练
        model.fit(valid_data)
        self.assertTrue(model.is_trained)

    def test_missing_columns_handling(self):
        """测试缺失列处理"""
        # 缺少rating列
        invalid_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [1, 2, 3]
        })

        model = create_recommender('simple_deepfm')

        # 应该抛出异常
        with self.assertRaises((ValueError, KeyError)):
            model.fit(invalid_data)

    def test_duplicate_ratings_handling(self):
        """测试重复评分处理"""
        # 包含重复的用户-物品对
        data = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'item_id': [1, 1, 2, 2],
            'rating': [4.0, 5.0, 3.0, 4.0]
        })

        model = create_recommender('simple_deepfm')
        # 应该能够处理（可能去重或平均）
        model.fit(data)
        self.assertTrue(model.is_trained)


class TestModelInteraction(unittest.TestCase):
    """模型交互测试"""

    @classmethod
    def setUpClass(cls):
        """准备测试数据"""
        np.random.seed(42)
        cls.train_data = pd.DataFrame({
            'user_id': np.random.randint(1, 15, 80),
            'item_id': np.random.randint(1, 25, 80),
            'rating': np.random.uniform(1, 5, 80)
        })

        cls.test_data = pd.DataFrame({
            'user_id': np.random.randint(1, 15, 20),
            'item_id': np.random.randint(1, 25, 20),
            'rating': np.random.uniform(1, 5, 20)
        })

    def test_prediction_consistency(self):
        """测试预测一致性"""
        model = create_recommender('simple_deepfm')
        model.fit(self.train_data)

        # 多次预测同一对应该得到相同结果
        pred1 = model.predict(1, 1)
        pred2 = model.predict(1, 1)
        pred3 = model.predict(1, 1)

        self.assertAlmostEqual(pred1, pred2, places=5)
        self.assertAlmostEqual(pred2, pred3, places=5)

    def test_recommendation_diversity(self):
        """测试推荐多样性"""
        model = create_recommender('simple_deepfm')
        model.fit(self.train_data)

        recommendations = model.get_user_recommendations(1, top_k=10)

        # 推荐的物品应该不重复
        item_ids = [rec['item_id'] for rec in recommendations]
        self.assertEqual(len(item_ids), len(set(item_ids)))

    def test_different_top_k_values(self):
        """测试不同的top_k值"""
        model = create_recommender('simple_deepfm')
        model.fit(self.train_data)

        for k in [1, 3, 5, 10]:
            with self.subTest(top_k=k):
                recommendations = model.get_user_recommendations(1, top_k=k)
                self.assertLessEqual(len(recommendations), k)


class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""

    @classmethod
    def setUpClass(cls):
        """准备测试数据"""
        np.random.seed(42)
        cls.data = pd.DataFrame({
            'user_id': np.random.randint(1, 10, 50),
            'item_id': np.random.randint(1, 15, 50),
            'rating': np.random.uniform(1, 5, 50)
        })

    def test_model_cache_integration(self):
        """测试模型与缓存集成"""
        # 创建缓存服务
        cache = CacheService(host='localhost', port=6379, db=15)

        if not cache.is_connected():
            self.skipTest("Redis未运行，跳过缓存集成测试")

        try:
            # 清空测试数据库
            cache.flush_db()

            # 训练模型
            model = create_recommender('simple_deepfm')
            model.fit(self.data)

            user_id = 1
            top_k = 5

            # 生成推荐
            recommendations = model.get_user_recommendations(user_id, top_k=top_k)

            # 缓存推荐结果
            cache_key = cache.generate_recommendation_key(user_id, top_k)
            cache.set(cache_key, recommendations, ttl=300)

            # 从缓存读取
            cached_recommendations = cache.get(cache_key)

            # 验证缓存一致性
            self.assertEqual(len(recommendations), len(cached_recommendations))

            for orig, cached in zip(recommendations, cached_recommendations):
                self.assertEqual(orig['item_id'], cached['item_id'])
                self.assertAlmostEqual(orig['score'], cached['score'], places=5)

        finally:
            if cache.is_connected():
                cache.flush_db()

    def test_multiple_models_coexistence(self):
        """测试多模型共存"""
        models = {}

        # 创建多个不同的模型
        model_configs = [
            ('simple_deepfm', {'learning_rate': 0.05}),
            ('simple_autoint', {'learning_rate': 0.03}),
            ('svd', {'n_components': 10})
        ]

        for model_name, config in model_configs:
            model = create_recommender(model_name, **config)
            model.fit(self.data)
            models[model_name] = model

        # 验证所有模型都能正常工作
        for model_name, model in models.items():
            with self.subTest(model=model_name):
                self.assertTrue(model.is_trained)
                prediction = model.predict(1, 1)
                self.assertIsInstance(prediction, (int, float))


class TestErrorHandling(unittest.TestCase):
    """错误处理测试"""

    def test_invalid_user_id(self):
        """测试无效用户ID"""
        model = create_recommender('simple_deepfm')

        data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [1, 2, 3],
            'rating': [4.0, 3.5, 5.0]
        })

        model.fit(data)

        # 负数用户ID
        with self.assertRaises((ValueError, KeyError)):
            model.predict(-1, 1)

    def test_invalid_item_id(self):
        """测试无效物品ID"""
        model = create_recommender('simple_deepfm')

        data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [1, 2, 3],
            'rating': [4.0, 3.5, 5.0]
        })

        model.fit(data)

        # 负数物品ID
        with self.assertRaises((ValueError, KeyError)):
            model.predict(1, -1)

    def test_model_not_trained(self):
        """测试未训练的模型"""
        model = create_recommender('simple_deepfm')

        # 未训练就预测应该抛出异常或返回默认值
        try:
            prediction = model.predict(1, 1)
            # 如果没抛出异常，应该返回合理的默认值
            self.assertIsInstance(prediction, (int, float))
        except (ValueError, AttributeError):
            # 或者抛出适当的异常
            pass


def run_integration_tests():
    """运行所有集成测试"""
    test_suite = unittest.TestSuite()

    # 添加所有测试类
    test_suite.addTest(unittest.makeSuite(TestEndToEndRecommendation))
    test_suite.addTest(unittest.makeSuite(TestDataPipeline))
    test_suite.addTest(unittest.makeSuite(TestModelInteraction))
    test_suite.addTest(unittest.makeSuite(TestSystemIntegration))
    test_suite.addTest(unittest.makeSuite(TestErrorHandling))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)
