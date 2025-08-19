#!/usr/bin/env python3
"""
在线推荐系统集成测试脚本
测试所有高级功能的集成工作情况
"""

import requests
import json
import time
import random
from typing import Dict, Any

class IntegrationTester:
    def __init__(self, base_url: str = 'http://localhost:5000'):
        self.base_url = base_url
        self.test_user_id = 1
        self.test_item_id = 100
        
    def test_health_check(self) -> bool:
        """测试系统健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✓ 系统健康检查通过: {health_data['status']}")
                print(f"  - 模型已加载: {health_data['model_loaded']}")
                print(f"  - Redis已连接: {health_data['redis_connected']}")
                return True
            else:
                print(f"✗ 健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ 健康检查异常: {e}")
            return False
    
    def test_basic_recommendation(self) -> bool:
        """测试基础推荐功能"""
        try:
            response = requests.get(
                f"{self.base_url}/recommend",
                params={'user_id': self.test_user_id, 'top_k': 5}
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✓ 基础推荐测试通过: 获得 {len(data['recommendations'])} 个推荐")
                return True
            else:
                print(f"✗ 基础推荐测试失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ 基础推荐测试异常: {e}")
            return False
    
    def test_enhanced_recommendation(self) -> bool:
        """测试增强推荐功能"""
        try:
            response = requests.get(
                f"{self.base_url}/recommend",
                params={
                    'user_id': self.test_user_id,
                    'top_k': 5,
                    'explanation': True,
                    'multi_objective': True,
                    'diversify': True,
                    'ab_test': True
                }
            )
            if response.status_code == 200:
                data = response.json()
                recommendations = data['recommendations']
                features = data.get('features_enabled', {})
                
                print(f"✓ 增强推荐测试通过: 获得 {len(recommendations)} 个推荐")
                print(f"  - 启用特性: {list(features.keys())}")
                
                # 检查第一个推荐是否有解释
                if recommendations and 'explanation' in recommendations[0]:
                    print(f"  - 推荐解释: ✓")
                
                # 检查多目标评分
                if recommendations and 'multi_objective_score' in recommendations[0]:
                    print(f"  - 多目标优化: ✓")
                
                return True
            else:
                print(f"✗ 增强推荐测试失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ 增强推荐测试异常: {e}")
            return False
    
    def test_explanation(self) -> bool:
        """测试推荐解释功能"""
        try:
            response = requests.get(
                f"{self.base_url}/explain",
                params={
                    'user_id': self.test_user_id,
                    'item_id': self.test_item_id
                }
            )
            if response.status_code == 200:
                data = response.json()
                explanation = data.get('explanation', {})
                
                print(f"✓ 推荐解释测试通过")
                print(f"  - 协同过滤解释: {'✓' if 'collaborative' in explanation else '✗'}")
                print(f"  - 内容相似解释: {'✓' if 'content' in explanation else '✗'}")
                print(f"  - 流行度解释: {'✓' if 'popularity' in explanation else '✗'}")
                print(f"  - 多样性解释: {'✓' if 'diversity' in explanation else '✗'}")
                
                if 'summary' in explanation:
                    print(f"  - 解释摘要: {explanation['summary'][:50]}...")
                
                return True
            else:
                print(f"✗ 推荐解释测试失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ 推荐解释测试异常: {e}")
            return False
    
    def test_interaction_logging(self) -> bool:
        """测试用户交互记录"""
        try:
            interaction_data = {
                'user_id': self.test_user_id,
                'item_id': self.test_item_id,
                'interaction_type': 'click',
                'experiment_name': 'integration_test'
            }
            
            response = requests.post(
                f"{self.base_url}/interaction",
                json=interaction_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ 用户交互记录测试通过: {data['message']}")
                return True
            else:
                print(f"✗ 用户交互记录测试失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ 用户交互记录测试异常: {e}")
            return False
    
    def test_ab_testing(self) -> bool:
        """测试A/B测试功能"""
        try:
            # 先记录一些交互数据
            for i in range(5):
                interaction_data = {
                    'user_id': self.test_user_id + i,
                    'item_id': self.test_item_id + i,
                    'interaction_type': random.choice(['click', 'view', 'purchase']),
                    'experiment_name': 'integration_test'
                }
                requests.post(f"{self.base_url}/interaction", json=interaction_data)
            
            # 获取A/B测试结果
            response = requests.get(f"{self.base_url}/ab-test/results/integration_test")
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', {})
                print(f"✓ A/B测试功能测试通过")
                print(f"  - 实验状态: {results.get('status', 'unknown')}")
                print(f"  - 变体数量: {len(results.get('variants', {}))}")
                return True
            else:
                print(f"✗ A/B测试功能测试失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ A/B测试功能测试异常: {e}")
            return False
    
    def test_rating_update(self) -> bool:
        """测试评分更新功能"""
        try:
            rating_data = {
                'user_id': self.test_user_id,
                'item_id': self.test_item_id,
                'rating': 4.5
            }
            
            response = requests.post(
                f"{self.base_url}/rating",
                json=rating_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ 评分更新测试通过: {data['message']}")
                return True
            else:
                print(f"✗ 评分更新测试失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ 评分更新测试异常: {e}")
            return False
    
    def test_similar_items(self) -> bool:
        """测试相似物品推荐"""
        try:
            response = requests.get(
                f"{self.base_url}/similar",
                params={'item_id': self.test_item_id, 'top_k': 5}
            )
            
            if response.status_code == 200:
                data = response.json()
                similar_items = data.get('similar_items', [])
                print(f"✓ 相似物品推荐测试通过: 获得 {len(similar_items)} 个相似物品")
                return True
            else:
                print(f"✗ 相似物品推荐测试失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ 相似物品推荐测试异常: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有集成测试"""
        print("=" * 60)
        print("🚀 开始运行在线推荐系统集成测试")
        print("=" * 60)
        
        tests = [
            ("系统健康检查", self.test_health_check),
            ("基础推荐功能", self.test_basic_recommendation),
            ("增强推荐功能", self.test_enhanced_recommendation),
            ("推荐解释功能", self.test_explanation),
            ("用户交互记录", self.test_interaction_logging),
            ("A/B测试功能", self.test_ab_testing),
            ("评分更新功能", self.test_rating_update),
            ("相似物品推荐", self.test_similar_items),
        ]
        
        results = {}
        passed = 0
        
        for test_name, test_func in tests:
            print(f"\n🧪 测试: {test_name}")
            print("-" * 40)
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                print(f"✗ 测试执行异常: {e}")
                results[test_name] = False
        
        print("\n" + "=" * 60)
        print(f"📊 测试总结: {passed}/{len(tests)} 个测试通过")
        print("=" * 60)
        
        for test_name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")
        
        if passed == len(tests):
            print("\n🎉 所有测试通过！推荐系统已准备就绪。")
        else:
            print(f"\n⚠️  有 {len(tests) - passed} 个测试失败，请检查系统配置。")
        
        return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='在线推荐系统集成测试')
    parser.add_argument(
        '--url', 
        default='http://localhost:5000',
        help='推荐系统API的基础URL (默认: http://localhost:5000)'
    )
    parser.add_argument(
        '--wait',
        type=int,
        default=0,
        help='测试前等待时间（秒），用于系统启动'
    )
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"⏳ 等待系统启动 {args.wait} 秒...")
        time.sleep(args.wait)
    
    tester = IntegrationTester(args.url)
    results = tester.run_all_tests()
    
    # 返回适当的退出码
    all_passed = all(results.values())
    exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
