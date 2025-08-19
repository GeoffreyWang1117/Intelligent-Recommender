import random
import hashlib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class ABTestConfig:
    """A/B测试配置"""
    
    def __init__(self, name: str, variants: Dict[str, Dict], 
                 traffic_allocation: Dict[str, float],
                 start_time: datetime, end_time: datetime,
                 success_metrics: List[str] = None):
        self.name = name
        self.variants = variants  # {variant_name: config}
        self.traffic_allocation = traffic_allocation  # {variant_name: percentage}
        self.start_time = start_time
        self.end_time = end_time
        self.success_metrics = success_metrics or ['ctr', 'conversion_rate']
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置有效性"""
        # 检查流量分配总和
        total_allocation = sum(self.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.001:
            raise ValueError(f"流量分配总和必须为1.0，当前为{total_allocation}")
        
        # 检查变体是否匹配
        for variant in self.traffic_allocation.keys():
            if variant not in self.variants:
                raise ValueError(f"变体 {variant} 在variants中不存在")
    
    def is_active(self) -> bool:
        """检查实验是否激活"""
        now = datetime.utcnow()
        return self.start_time <= now <= self.end_time

class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self, cache_service=None):
        self.cache_service = cache_service
        self.experiments = {}  # {experiment_name: ABTestConfig}
        self.user_assignments = {}  # {user_id: {experiment_name: variant}}
        
    def add_experiment(self, experiment: ABTestConfig):
        """添加实验"""
        self.experiments[experiment.name] = experiment
        logger.info(f"添加A/B测试实验: {experiment.name}")
    
    def get_user_variant(self, user_id: int, experiment_name: str) -> Optional[str]:
        """获取用户所属的实验变体"""
        if experiment_name not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_name]
        
        # 检查实验是否激活
        if not experiment.is_active():
            return None
        
        # 检查缓存
        cache_key = f"ab_test:user:{user_id}:exp:{experiment_name}"
        if self.cache_service:
            cached_variant = self.cache_service.get(cache_key)
            if cached_variant:
                return cached_variant
        
        # 计算用户分组
        variant = self._assign_user_to_variant(user_id, experiment)
        
        # 缓存结果
        if self.cache_service and variant:
            # 缓存到实验结束时间
            ttl = int((experiment.end_time - datetime.utcnow()).total_seconds())
            self.cache_service.set(cache_key, variant, ttl=max(ttl, 3600))
        
        return variant
    
    def _assign_user_to_variant(self, user_id: int, experiment: ABTestConfig) -> Optional[str]:
        """将用户分配到实验变体"""
        # 使用用户ID和实验名称生成一致的哈希
        hash_input = f"{user_id}:{experiment.name}".encode('utf-8')
        hash_value = hashlib.md5(hash_input).hexdigest()
        
        # 转换为0-1之间的数值
        random_value = int(hash_value[:8], 16) / (16**8)
        
        # 根据流量分配确定变体
        cumulative_probability = 0.0
        for variant, allocation in experiment.traffic_allocation.items():
            cumulative_probability += allocation
            if random_value <= cumulative_probability:
                return variant
        
        return None
    
    def log_event(self, user_id: int, experiment_name: str, event_type: str, 
                  event_data: Dict = None):
        """记录实验事件"""
        variant = self.get_user_variant(user_id, experiment_name)
        if not variant:
            return
        
        event = {
            'user_id': user_id,
            'experiment_name': experiment_name,
            'variant': variant,
            'event_type': event_type,
            'event_data': event_data or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # 记录到缓存或数据库
        if self.cache_service:
            cache_key = f"ab_events:{experiment_name}:{variant}:{event_type}"
            events = self.cache_service.get(cache_key) or []
            events.append(event)
            self.cache_service.set(cache_key, events, ttl=86400)  # 24小时
        
        logger.info(f"记录A/B测试事件: {event}")
    
    def get_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """获取实验结果"""
        if experiment_name not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_name]
        results = {}
        
        for variant in experiment.variants.keys():
            variant_results = self._calculate_variant_metrics(experiment_name, variant)
            results[variant] = variant_results
        
        return {
            'experiment_name': experiment_name,
            'status': 'active' if experiment.is_active() else 'inactive',
            'variants': results,
            'statistical_significance': self._calculate_statistical_significance(results)
        }
    
    def _calculate_variant_metrics(self, experiment_name: str, variant: str) -> Dict:
        """计算变体指标"""
        if not self.cache_service:
            return {}
        
        # 获取各种事件数据
        impression_key = f"ab_events:{experiment_name}:{variant}:impression"
        click_key = f"ab_events:{experiment_name}:{variant}:click"
        conversion_key = f"ab_events:{experiment_name}:{variant}:conversion"
        
        impressions = self.cache_service.get(impression_key) or []
        clicks = self.cache_service.get(click_key) or []
        conversions = self.cache_service.get(conversion_key) or []
        
        # 计算指标
        impression_count = len(impressions)
        click_count = len(clicks)
        conversion_count = len(conversions)
        
        ctr = click_count / impression_count if impression_count > 0 else 0
        conversion_rate = conversion_count / impression_count if impression_count > 0 else 0
        
        return {
            'impressions': impression_count,
            'clicks': click_count,
            'conversions': conversion_count,
            'ctr': ctr,
            'conversion_rate': conversion_rate,
            'sample_size': impression_count
        }
    
    def _calculate_statistical_significance(self, results: Dict) -> Dict:
        """计算统计显著性"""
        # 简化的统计显著性计算
        # 实际项目中应该使用更严格的统计测试
        
        variants = list(results.keys())
        if len(variants) < 2:
            return {}
        
        # 比较第一个变体和其他变体
        control = variants[0]
        control_metrics = results[control]
        
        significance_results = {}
        
        for variant in variants[1:]:
            variant_metrics = results[variant]
            
            # 简单的样本大小检查
            min_sample_size = 100
            control_sample = control_metrics.get('sample_size', 0)
            variant_sample = variant_metrics.get('sample_size', 0)
            
            if control_sample < min_sample_size or variant_sample < min_sample_size:
                significance_results[f"{control}_vs_{variant}"] = {
                    'significant': False,
                    'reason': 'insufficient_sample_size',
                    'min_required': min_sample_size
                }
                continue
            
            # 比较CTR差异
            control_ctr = control_metrics.get('ctr', 0)
            variant_ctr = variant_metrics.get('ctr', 0)
            
            improvement = (variant_ctr - control_ctr) / control_ctr if control_ctr > 0 else 0
            
            significance_results[f"{control}_vs_{variant}"] = {
                'significant': abs(improvement) > 0.1,  # 10%改进阈值
                'improvement': improvement,
                'control_ctr': control_ctr,
                'variant_ctr': variant_ctr
            }
        
        return significance_results

class RecommendationABTest:
    """推荐系统专用A/B测试"""
    
    @staticmethod
    def create_algorithm_test(start_time: datetime, end_time: datetime) -> ABTestConfig:
        """创建算法对比测试"""
        return ABTestConfig(
            name="recommendation_algorithm_test",
            variants={
                "lightfm": {
                    "algorithm": "lightfm",
                    "params": {"components": 100, "learning_rate": 0.05}
                },
                "lightfm_enhanced": {
                    "algorithm": "lightfm",
                    "params": {"components": 150, "learning_rate": 0.03}
                }
            },
            traffic_allocation={
                "lightfm": 0.5,
                "lightfm_enhanced": 0.5
            },
            start_time=start_time,
            end_time=end_time,
            success_metrics=["ctr", "conversion_rate", "engagement_time"]
        )
    
    @staticmethod
    def create_topk_test(start_time: datetime, end_time: datetime) -> ABTestConfig:
        """创建推荐数量测试"""
        return ABTestConfig(
            name="recommendation_topk_test",
            variants={
                "top_5": {"top_k": 5},
                "top_10": {"top_k": 10},
                "top_15": {"top_k": 15}
            },
            traffic_allocation={
                "top_5": 0.33,
                "top_10": 0.34,
                "top_15": 0.33
            },
            start_time=start_time,
            end_time=end_time
        )
    
    @staticmethod
    def create_ranking_test(start_time: datetime, end_time: datetime) -> ABTestConfig:
        """创建排序策略测试"""
        return ABTestConfig(
            name="ranking_strategy_test",
            variants={
                "score_only": {
                    "ranking_strategy": "score",
                    "diversity_weight": 0.0
                },
                "score_with_diversity": {
                    "ranking_strategy": "score_diversity",
                    "diversity_weight": 0.2
                }
            },
            traffic_allocation={
                "score_only": 0.5,
                "score_with_diversity": 0.5
            },
            start_time=start_time,
            end_time=end_time
        )
