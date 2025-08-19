#!/usr/bin/env python3
"""
推荐系统模型包

本包包含了完整的推荐系统算法实现，采用模块化设计，支持多种算法的灵活组合和切换。

主要模块：
- base_recommender: 所有推荐算法的基类接口
- algorithm_factory: 算法工厂，用于动态创建和管理算法
- svd_model: SVD矩阵分解算法
- deepfm: DeepFM深度学习算法
- autoint: AutoInt自注意力算法
- din: DIN深度兴趣网络算法
- ensemble_recommender: 集成推荐器
- faiss_index: FAISS向量索引
- lightfm_model: LightFM算法

使用示例：
    from models import create_recommender, ExplainableRecommender
    
    # 创建单个算法
    model = create_recommender('deepfm', embedding_dim=8)
    
    # 创建集成推荐器
    ensemble = ExplainableRecommender(['deepfm', 'autoint', 'din'])
"""

# 导入主要类和函数
from .base_recommender import BaseRecommender
from .algorithm_factory import (
    create_recommender, 
    register_algorithm, 
    get_available_algorithms,
    AlgorithmManager
)
from .svd_model import SVDRecommender
from .deepfm import DeepFMRecommender, SimpleDeepFMRecommender
from .autoint import AutoIntRecommender, SimpleAutoIntRecommender
from .din import DINRecommender, SimpleDINRecommender
from .ensemble_recommender import ExplainableRecommender
from .optimized_ensemble import OptimizedEnsembleRecommender, create_optimized_ensemble

# 新的Teacher模型
try:
    from .transformer4rec import Transformer4RecRecommender
except ImportError:
    Transformer4RecRecommender = None

try:
    from .dcnv2 import DCNv2Recommender
except ImportError:
    DCNv2Recommender = None

try:
    from .xdeepfm import xDeepFMRecommender
except ImportError:
    xDeepFMRecommender = None

# 可选组件（可能不可用）
try:
    from .lightfm_model import LightFMRecommender
except ImportError:
    LightFMRecommender = None

try:
    from .faiss_index import FAISSIndex
except ImportError:
    FAISSIndex = None

# 版本信息
__version__ = "1.0.0"
__author__ = "Recommendation System Team"

# 导出的公共接口
__all__ = [
    # 基类
    'BaseRecommender',
    
    # 工厂模式
    'create_recommender',
    'register_algorithm', 
    'get_available_algorithms',
    'AlgorithmManager',
    
    # 具体算法
    'SVDRecommender',
    'DeepFMRecommender',
    'SimpleDeepFMRecommender',
    'AutoIntRecommender', 
    'SimpleAutoIntRecommender',
    'DINRecommender',
    'SimpleDINRecommender',
    
    # 集成推荐器
    'ExplainableRecommender',
    'OptimizedEnsembleRecommender',
    'create_optimized_ensemble',
    
    # Teacher模型
    'Transformer4RecRecommender',
    'DCNv2Recommender', 
    'xDeepFMRecommender',
]

# Teacher模型
if Transformer4RecRecommender is not None:
    __all__.append('Transformer4RecRecommender')
if DCNv2Recommender is not None:
    __all__.append('DCNv2Recommender') 
if xDeepFMRecommender is not None:
    __all__.append('xDeepFMRecommender')

# 可选组件
if LightFMRecommender is not None:
    __all__.append('LightFMRecommender')
if FAISSIndex is not None:
    __all__.append('FAISSIndex')

# 配置日志
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
