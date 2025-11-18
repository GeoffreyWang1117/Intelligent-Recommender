#!/usr/bin/env python3
"""
缓存服务测试套件

测试Redis缓存服务的所有核心功能，包括连接、存储、检索、过期等。
"""

import unittest
import json
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.cache import CacheService


class TestCacheService(unittest.TestCase):
    """缓存服务测试类"""

    def setUp(self):
        """测试前准备"""
        # 使用Mock Redis客户端进行测试
        self.mock_redis = MagicMock()
        self.cache_service = CacheService(
            host='localhost',
            port=6379,
            db=0,
            default_ttl=3600
        )

    @patch('services.cache.redis.Redis')
    def test_connection_success(self, mock_redis_class):
        """测试Redis成功连接"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client

        cache = CacheService()

        # 验证连接成功
        self.assertIsNotNone(cache.redis_client)
        mock_client.ping.assert_called_once()

    @patch('services.cache.redis.Redis')
    def test_connection_failure(self, mock_redis_class):
        """测试Redis连接失败"""
        mock_redis_class.side_effect = Exception("Connection failed")

        cache = CacheService()

        # 验证连接失败后的状态
        self.assertIsNone(cache.redis_client)

    def test_is_connected_when_client_none(self):
        """测试客户端为None时的连接状态"""
        cache = CacheService()
        cache.redis_client = None

        self.assertFalse(cache.is_connected())

    @patch('services.cache.redis.Redis')
    def test_is_connected_when_ping_fails(self, mock_redis_class):
        """测试ping失败时的连接状态"""
        import redis as redis_module

        mock_client = MagicMock()
        mock_client.ping.side_effect = redis_module.ConnectionError("Connection lost")
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        self.assertFalse(cache.is_connected())

    @patch('services.cache.redis.Redis')
    def test_set_value_success(self, mock_redis_class):
        """测试成功设置缓存值"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.setex.return_value = True
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        # 设置缓存
        result = cache.set('test_key', {'data': 'test_value'}, ttl=600)

        self.assertTrue(result)
        mock_client.setex.assert_called_once()

        # 验证调用参数
        call_args = mock_client.setex.call_args
        self.assertEqual(call_args[0][0], 'test_key')
        self.assertEqual(call_args[0][1], 600)
        # 验证JSON序列化
        stored_value = json.loads(call_args[0][2])
        self.assertEqual(stored_value, {'data': 'test_value'})

    @patch('services.cache.redis.Redis')
    def test_set_value_with_default_ttl(self, mock_redis_class):
        """测试使用默认TTL设置缓存"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.setex.return_value = True
        mock_redis_class.return_value = mock_client

        cache = CacheService(default_ttl=1800)
        cache.redis_client = mock_client

        cache.set('key', 'value')

        # 验证使用了默认TTL
        call_args = mock_client.setex.call_args
        self.assertEqual(call_args[0][1], 1800)

    @patch('services.cache.redis.Redis')
    def test_get_value_success(self, mock_redis_class):
        """测试成功获取缓存值"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = json.dumps({'data': 'cached_value'})
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        result = cache.get('test_key')

        self.assertEqual(result, {'data': 'cached_value'})
        mock_client.get.assert_called_once_with('test_key')

    @patch('services.cache.redis.Redis')
    def test_get_value_not_found(self, mock_redis_class):
        """测试获取不存在的缓存"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        result = cache.get('nonexistent_key')

        self.assertIsNone(result)

    @patch('services.cache.redis.Redis')
    def test_delete_value_success(self, mock_redis_class):
        """测试成功删除缓存"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.delete.return_value = 1
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        result = cache.delete('test_key')

        self.assertTrue(result)
        mock_client.delete.assert_called_once_with('test_key')

    @patch('services.cache.redis.Redis')
    def test_delete_value_not_found(self, mock_redis_class):
        """测试删除不存在的缓存"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.delete.return_value = 0
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        result = cache.delete('nonexistent_key')

        self.assertFalse(result)

    @patch('services.cache.redis.Redis')
    def test_exists_key_true(self, mock_redis_class):
        """测试检查存在的键"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.exists.return_value = 1
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        result = cache.exists('existing_key')

        self.assertTrue(result)

    @patch('services.cache.redis.Redis')
    def test_exists_key_false(self, mock_redis_class):
        """测试检查不存在的键"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.exists.return_value = 0
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        result = cache.exists('nonexistent_key')

        self.assertFalse(result)

    @patch('services.cache.redis.Redis')
    def test_expire_key(self, mock_redis_class):
        """测试设置键过期时间"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.expire.return_value = True
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        result = cache.expire('test_key', 300)

        self.assertTrue(result)
        mock_client.expire.assert_called_once_with('test_key', 300)

    @patch('services.cache.redis.Redis')
    def test_ttl_key(self, mock_redis_class):
        """测试获取键剩余时间"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.ttl.return_value = 600
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        result = cache.ttl('test_key')

        self.assertEqual(result, 600)

    @patch('services.cache.redis.Redis')
    def test_flush_db(self, mock_redis_class):
        """测试清空数据库"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.flushdb.return_value = True
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        result = cache.flush_db()

        self.assertTrue(result)
        mock_client.flushdb.assert_called_once()

    @patch('services.cache.redis.Redis')
    def test_keys_pattern(self, mock_redis_class):
        """测试获取匹配的键"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.keys.return_value = ['key1', 'key2', 'key3']
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        result = cache.keys('key*')

        self.assertEqual(result, ['key1', 'key2', 'key3'])
        mock_client.keys.assert_called_once_with('key*')

    def test_generate_recommendation_key(self):
        """测试生成推荐缓存键"""
        cache = CacheService()

        key = cache.generate_recommendation_key(user_id=123, top_k=10)

        self.assertEqual(key, 'rec:user:123:top_k:10')

    def test_generate_similar_key(self):
        """测试生成相似物品缓存键"""
        cache = CacheService()

        key = cache.generate_similar_key(item_id=456, top_k=20)

        self.assertEqual(key, 'sim:item:456:top_k:20')

    def test_generate_user_profile_key(self):
        """测试生成用户画像缓存键"""
        cache = CacheService()

        key = cache.generate_user_profile_key(user_id=789)

        self.assertEqual(key, 'profile:user:789')

    def test_generate_item_features_key(self):
        """测试生成物品特征缓存键"""
        cache = CacheService()

        key = cache.generate_item_features_key(item_id=321)

        self.assertEqual(key, 'features:item:321')

    @patch('services.cache.redis.Redis')
    def test_set_when_not_connected(self, mock_redis_class):
        """测试未连接时设置缓存"""
        mock_client = MagicMock()
        mock_client.ping.side_effect = Exception("Not connected")
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = None

        result = cache.set('key', 'value')

        self.assertFalse(result)

    @patch('services.cache.redis.Redis')
    def test_get_when_not_connected(self, mock_redis_class):
        """测试未连接时获取缓存"""
        cache = CacheService()
        cache.redis_client = None

        result = cache.get('key')

        self.assertIsNone(result)

    @patch('services.cache.redis.Redis')
    def test_complex_data_serialization(self, mock_redis_class):
        """测试复杂数据序列化"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.setex.return_value = True
        mock_redis_class.return_value = mock_client

        cache = CacheService()
        cache.redis_client = mock_client

        # 复杂数据结构
        complex_data = {
            'users': [1, 2, 3],
            'items': {'item1': 'data1', 'item2': 'data2'},
            'scores': [0.1, 0.2, 0.3],
            'metadata': {
                'version': '1.0',
                'timestamp': '2025-01-01'
            }
        }

        result = cache.set('complex_key', complex_data)

        self.assertTrue(result)

        # 验证序列化正确
        call_args = mock_client.setex.call_args
        stored_value = json.loads(call_args[0][2])
        self.assertEqual(stored_value, complex_data)


class TestCacheServiceIntegration(unittest.TestCase):
    """缓存服务集成测试（需要真实Redis）"""

    def setUp(self):
        """测试前准备"""
        # 尝试连接真实Redis
        self.cache = CacheService(host='localhost', port=6379, db=15)  # 使用测试数据库

        if not self.cache.is_connected():
            self.skipTest("Redis未运行，跳过集成测试")

        # 清空测试数据库
        self.cache.flush_db()

    def tearDown(self):
        """测试后清理"""
        if self.cache and self.cache.is_connected():
            self.cache.flush_db()

    def test_set_and_get_integration(self):
        """测试设置和获取（集成）"""
        key = 'integration_test_key'
        value = {'test': 'data', 'number': 42}

        # 设置
        set_result = self.cache.set(key, value, ttl=60)
        self.assertTrue(set_result)

        # 获取
        get_result = self.cache.get(key)
        self.assertEqual(get_result, value)

    def test_expiration_integration(self):
        """测试过期功能（集成）"""
        key = 'expiring_key'
        value = 'temporary_data'

        # 设置1秒过期
        self.cache.set(key, value, ttl=1)

        # 立即获取应该存在
        self.assertEqual(self.cache.get(key), value)

        # 等待2秒后应该过期
        time.sleep(2)
        self.assertIsNone(self.cache.get(key))

    def test_delete_integration(self):
        """测试删除（集成）"""
        key = 'delete_test_key'
        value = 'delete_me'

        # 设置
        self.cache.set(key, value)
        self.assertEqual(self.cache.get(key), value)

        # 删除
        delete_result = self.cache.delete(key)
        self.assertTrue(delete_result)

        # 验证已删除
        self.assertIsNone(self.cache.get(key))


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加单元测试
    test_suite.addTest(unittest.makeSuite(TestCacheService))

    # 添加集成测试
    test_suite.addTest(unittest.makeSuite(TestCacheServiceIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
