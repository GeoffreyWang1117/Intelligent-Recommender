#!/usr/bin/env python3
"""
推荐算法工厂模块

提供统一的接口来创建和管理不同的推荐算法，支持算法的动态切换和配置。
"""

import logging
from typing import Dict, Type, Optional, Any
from .base_recommender import BaseRecommender

logger = logging.getLogger(__name__)

# 算法注册表
ALGORITHM_REGISTRY: Dict[str, Type[BaseRecommender]] = {}


def register_algorithm(name: str, algorithm_class: Type[BaseRecommender]):
    """注册推荐算法
    
    Args:
        name: 算法名称
        algorithm_class: 算法类
    """
    ALGORITHM_REGISTRY[name] = algorithm_class
    logger.info(f"已注册推荐算法: {name}")


def get_available_algorithms() -> list:
    """获取所有可用的算法名称
    
    Returns:
        算法名称列表
    """
    return list(ALGORITHM_REGISTRY.keys())


def create_recommender(algorithm_name: str, **kwargs) -> Optional[BaseRecommender]:
    """创建推荐器实例
    
    Args:
        algorithm_name: 算法名称
        **kwargs: 算法参数
        
    Returns:
        推荐器实例
    """
    if algorithm_name not in ALGORITHM_REGISTRY:
        logger.error(f"未知的推荐算法: {algorithm_name}")
        logger.info(f"可用算法: {get_available_algorithms()}")
        raise ValueError(f"未知的推荐算法: {algorithm_name}")
    
    try:
        algorithm_class = ALGORITHM_REGISTRY[algorithm_name]
        return algorithm_class(**kwargs)
    except Exception as e:
        logger.error(f"创建推荐器失败 ({algorithm_name}): {e}")
        raise ValueError(f"创建推荐器失败 ({algorithm_name}): {e}")


def initialize_algorithms():
    """初始化并注册所有算法"""
    
    # 注册SVD算法
    try:
        from .svd_model import SVDRecommender
        register_algorithm('svd', SVDRecommender)
    except ImportError as e:
        logger.warning(f"SVD算法加载失败: {e}")
    
    # 注册DeepFM算法
    try:
        from .deepfm import DeepFMRecommender, SimpleDeepFMRecommender, create_deepfm_recommender
        register_algorithm('deepfm', DeepFMRecommender)
        register_algorithm('simple_deepfm', SimpleDeepFMRecommender)
        
        # 注册工厂函数创建的算法
        def create_deepfm_auto(**kwargs):
            return create_deepfm_recommender(use_pytorch=True, **kwargs)
        
        def create_deepfm_simple(**kwargs):
            return create_deepfm_recommender(use_pytorch=False, **kwargs)
            
        register_algorithm('deepfm_auto', lambda **kwargs: create_deepfm_auto(**kwargs))
        register_algorithm('deepfm_simple', lambda **kwargs: create_deepfm_simple(**kwargs))
        
    except ImportError as e:
        logger.warning(f"DeepFM算法加载失败: {e}")
    
    # 注册AutoInt算法
    try:
        from .autoint import AutoIntRecommender, SimpleAutoIntRecommender, create_autoint_recommender
        register_algorithm('autoint', AutoIntRecommender)
        register_algorithm('simple_autoint', SimpleAutoIntRecommender)
        
        def create_autoint_auto(**kwargs):
            return create_autoint_recommender(use_pytorch=True, **kwargs)
        
        def create_autoint_simple(**kwargs):
            return create_autoint_recommender(use_pytorch=False, **kwargs)
            
        register_algorithm('autoint_auto', lambda **kwargs: create_autoint_auto(**kwargs))
        register_algorithm('autoint_simple', lambda **kwargs: create_autoint_simple(**kwargs))
        
    except ImportError as e:
        logger.warning(f"AutoInt算法加载失败: {e}")
    
    # 注册DIN算法
    try:
        from .din import DINRecommender, SimpleDINRecommender, create_din_recommender
        register_algorithm('din', DINRecommender)
        register_algorithm('simple_din', SimpleDINRecommender)
        
        def create_din_auto(**kwargs):
            return create_din_recommender(use_pytorch=True, **kwargs)
        
        def create_din_simple(**kwargs):
            return create_din_recommender(use_pytorch=False, **kwargs)
            
        register_algorithm('din_auto', lambda **kwargs: create_din_auto(**kwargs))
        register_algorithm('din_simple', lambda **kwargs: create_din_simple(**kwargs))
        
    except ImportError as e:
        logger.warning(f"DIN算法加载失败: {e}")
    
    # 注册Transformer4Rec算法
    try:
        from .transformer4rec import Transformer4RecRecommender, create_transformer4rec_recommender
        register_algorithm('transformer4rec', Transformer4RecRecommender)
        
        def create_transformer4rec_auto(**kwargs):
            return create_transformer4rec_recommender(**kwargs)
            
        register_algorithm('transformer4rec_auto', lambda **kwargs: create_transformer4rec_auto(**kwargs))
        
    except ImportError as e:
        logger.warning(f"Transformer4Rec算法加载失败: {e}")
    
    # 注册DCNv2算法
    try:
        from .dcnv2 import DCNv2Recommender, create_dcnv2_recommender
        register_algorithm('dcnv2', DCNv2Recommender)
        
        def create_dcnv2_auto(**kwargs):
            return create_dcnv2_recommender(**kwargs)
            
        register_algorithm('dcnv2_auto', lambda **kwargs: create_dcnv2_auto(**kwargs))
        
    except ImportError as e:
        logger.warning(f"DCNv2算法加载失败: {e}")
    
    # 注册xDeepFM算法
    try:
        from .xdeepfm import xDeepFMRecommender, create_xdeepfm_recommender
        register_algorithm('xdeepfm', xDeepFMRecommender)
        
        def create_xdeepfm_auto(**kwargs):
            return create_xdeepfm_recommender(**kwargs)
            
        register_algorithm('xdeepfm_auto', lambda **kwargs: create_xdeepfm_auto(**kwargs))
        
    except ImportError as e:
        logger.warning(f"xDeepFM算法加载失败: {e}")
    
    logger.info(f"算法初始化完成，共注册 {len(ALGORITHM_REGISTRY)} 个算法")


class AlgorithmManager:
    """算法管理器
    
    提供算法的高级管理功能，包括算法切换、性能监控等。
    """
    
    def __init__(self):
        self.current_algorithms = {}
        self.algorithm_configs = {}
        
    def add_algorithm(self, name: str, algorithm_name: str, **config):
        """添加算法实例
        
        Args:
            name: 实例名称
            algorithm_name: 算法名称
            **config: 算法配置
        """
        recommender = create_recommender(algorithm_name, **config)
        if recommender:
            self.current_algorithms[name] = recommender
            self.algorithm_configs[name] = {
                'algorithm_name': algorithm_name,
                'config': config
            }
            logger.info(f"已添加算法实例: {name} ({algorithm_name})")
        else:
            logger.error(f"添加算法实例失败: {name}")
    
    def get_algorithm(self, name: str) -> Optional[BaseRecommender]:
        """获取算法实例
        
        Args:
            name: 实例名称
            
        Returns:
            算法实例
        """
        return self.current_algorithms.get(name)
    
    def remove_algorithm(self, name: str):
        """移除算法实例
        
        Args:
            name: 实例名称
        """
        if name in self.current_algorithms:
            del self.current_algorithms[name]
            del self.algorithm_configs[name]
            logger.info(f"已移除算法实例: {name}")
    
    def list_algorithms(self) -> Dict[str, str]:
        """列出所有算法实例
        
        Returns:
            实例名称到算法名称的映射
        """
        return {name: config['algorithm_name'] 
                for name, config in self.algorithm_configs.items()}
    
    def switch_algorithm(self, instance_name: str, new_algorithm_name: str, **new_config):
        """切换算法
        
        Args:
            instance_name: 实例名称
            new_algorithm_name: 新算法名称
            **new_config: 新算法配置
        """
        if instance_name in self.current_algorithms:
            old_algorithm = self.algorithm_configs[instance_name]['algorithm_name']
            
            # 创建新算法实例
            new_recommender = create_recommender(new_algorithm_name, **new_config)
            if new_recommender:
                self.current_algorithms[instance_name] = new_recommender
                self.algorithm_configs[instance_name] = {
                    'algorithm_name': new_algorithm_name,
                    'config': new_config
                }
                logger.info(f"已切换算法: {instance_name} ({old_algorithm} -> {new_algorithm_name})")
            else:
                logger.error(f"切换算法失败: {instance_name}")
        else:
            logger.error(f"算法实例不存在: {instance_name}")


# 全局算法管理器实例
algorithm_manager = AlgorithmManager()

# 初始化算法
initialize_algorithms()
