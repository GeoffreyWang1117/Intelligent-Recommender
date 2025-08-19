#!/usr/bin/env python3
"""
API测试脚本
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITester:
    """API测试类"""
    
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self):
        """测试健康检查接口"""
        logger.info("测试健康检查接口...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            logger.info(f"健康检查响应: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"健康状态: {data.get('status')}")
                return True
            else:
                logger.error(f"健康检查失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"健康检查请求失败: {e}")
            return False
    
    def test_recommendations(self, user_id=1, top_k=5):
        """测试推荐接口"""
        logger.info(f"测试用户 {user_id} 的推荐接口...")
        
        try:
            params = {'user_id': user_id, 'top_k': top_k}
            response = self.session.get(f"{self.base_url}/recommend", params=params)
            
            logger.info(f"推荐接口响应: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"推荐数量: {len(data.get('recommendations', []))}")
                logger.info(f"是否来自缓存: {data.get('cached', False)}")
                
                # 打印前3个推荐
                recommendations = data.get('recommendations', [])[:3]
                for i, rec in enumerate(recommendations, 1):
                    logger.info(f"  推荐{i}: 物品ID={rec.get('item_id')}, 评分={rec.get('score', 0):.4f}")
                
                return True
            else:
                logger.error(f"推荐接口失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"推荐接口请求失败: {e}")
            return False
    
    def test_similar_items(self, item_id=1, top_k=5):
        """测试相似物品接口"""
        logger.info(f"测试物品 {item_id} 的相似物品接口...")
        
        try:
            params = {'item_id': item_id, 'top_k': top_k}
            response = self.session.get(f"{self.base_url}/similar", params=params)
            
            logger.info(f"相似物品接口响应: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                similar_items = data.get('similar_items', [])
                logger.info(f"相似物品数量: {len(similar_items)}")
                logger.info(f"是否来自缓存: {data.get('cached', False)}")
                
                # 打印前3个相似物品
                for i, item in enumerate(similar_items[:3], 1):
                    logger.info(f"  相似物品{i}: 物品ID={item.get('item_id')}, 相似度={item.get('similarity', 0):.4f}")
                
                return True
            else:
                logger.error(f"相似物品接口失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"相似物品接口请求失败: {e}")
            return False
    
    def test_update_rating(self, user_id=999, item_id=1, rating=4.5):
        """测试评分更新接口"""
        logger.info(f"测试评分更新接口...")
        
        try:
            data = {
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating
            }
            
            response = self.session.post(
                f"{self.base_url}/rating",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            logger.info(f"评分更新接口响应: {response.status_code}")
            
            if response.status_code == 200:
                resp_data = response.json()
                logger.info(f"评分更新成功: {resp_data.get('message')}")
                return True
            else:
                logger.error(f"评分更新失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"评分更新请求失败: {e}")
            return False
    
    def test_model_info(self):
        """测试模型信息接口"""
        logger.info("测试模型信息接口...")
        
        try:
            response = self.session.get(f"{self.base_url}/info")
            logger.info(f"模型信息接口响应: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("模型信息:")
                for key, value in data.items():
                    if key != 'timestamp':
                        logger.info(f"  {key}: {value}")
                return True
            else:
                logger.error(f"模型信息接口失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"模型信息接口请求失败: {e}")
            return False
    
    def test_cache_performance(self, user_id=1, top_k=10, iterations=3):
        """测试缓存性能"""
        logger.info("测试缓存性能...")
        
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            
            params = {'user_id': user_id, 'top_k': top_k}
            response = self.session.get(f"{self.base_url}/recommend", params=params)
            
            end_time = time.time()
            request_time = end_time - start_time
            times.append(request_time)
            
            if response.status_code == 200:
                data = response.json()
                cached = data.get('cached', False)
                logger.info(f"  请求{i+1}: {request_time:.3f}s, 缓存: {cached}")
            else:
                logger.error(f"  请求{i+1}失败: {response.status_code}")
        
        if times:
            avg_time = sum(times) / len(times)
            logger.info(f"平均响应时间: {avg_time:.3f}s")
        
        return len(times) == iterations
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始API测试...")
        
        tests = [
            self.test_health_check,
            self.test_model_info,
            lambda: self.test_recommendations(user_id=1),
            lambda: self.test_similar_items(item_id=1),
            lambda: self.test_update_rating(user_id=999, item_id=1, rating=4.5),
            lambda: self.test_cache_performance(user_id=1)
        ]
        
        passed = 0
        total = len(tests)
        
        for i, test in enumerate(tests, 1):
            logger.info(f"\n--- 测试 {i}/{total} ---")
            try:
                if test():
                    passed += 1
                    logger.info("✓ 测试通过")
                else:
                    logger.error("✗ 测试失败")
            except Exception as e:
                logger.error(f"✗ 测试异常: {e}")
        
        logger.info(f"\n=== 测试总结 ===")
        logger.info(f"通过: {passed}/{total}")
        logger.info(f"成功率: {passed/total*100:.1f}%")
        
        return passed == total

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='API测试工具')
    parser.add_argument('--url', default='http://localhost:5000', help='API基础URL')
    parser.add_argument('--test', choices=['health', 'recommend', 'similar', 'rating', 'info', 'cache', 'all'], 
                       default='all', help='要运行的测试')
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.test == 'health':
        tester.test_health_check()
    elif args.test == 'recommend':
        tester.test_recommendations()
    elif args.test == 'similar':
        tester.test_similar_items()
    elif args.test == 'rating':
        tester.test_update_rating()
    elif args.test == 'info':
        tester.test_model_info()
    elif args.test == 'cache':
        tester.test_cache_performance()
    elif args.test == 'all':
        tester.run_all_tests()

if __name__ == '__main__':
    main()
