"""
Multi-Teacher Fusion System - Base Configuration
系统基础配置定义
"""

from dataclasses import dataclass
from typing import Optional
import logging

@dataclass
class SystemConfig:
    """系统配置"""
    log_level: str = "INFO"
    use_cuda: bool = True
    random_seed: int = 42
    num_workers: int = 4
    cache_size: int = 1000
    
    def __post_init__(self):
        # 配置日志级别
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {self.log_level}')

@dataclass 
class DatasetConfig:
    """数据集配置"""
    name: str = "movielens-1m"
    batch_size: int = 256
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    min_ratings_per_user: int = 20
    min_ratings_per_item: int = 10
    
    # 特征工程配置
    normalize_features: bool = True
    use_temporal_features: bool = True
    use_user_features: bool = True
    use_item_features: bool = True

from dataclasses import dataclass
from typing import List
import json
import os


@dataclass
class BaseConfig:
    """基础配置类"""
    
    # 数据集配置
    dataset_name: str = "movielens"
    dataset_path: str = "data/movielens/"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 模型通用配置
    embedding_dim: int = 64
    hidden_dim: int = 128
    dropout_rate: float = 0.1
    batch_size: int = 256
    learning_rate: float = 0.001
    
    # 训练配置
    max_epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "cuda"
    random_seed: int = 42
    
    # 评估配置
    eval_batch_size: int = 512
    top_k_list: List[int] = None
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.top_k_list is None:
            self.top_k_list = [5, 10, 20, 50]
        if self.metrics is None:
            self.metrics = ["rmse", "mae", "precision", "recall", "ndcg"]
    
    # 系统配置
    num_workers: int = 4
    pin_memory: bool = True
    cache_dir: str = "cache/"
    log_dir: str = "logs/"
    model_save_dir: str = "saved_models/"
    
    def __post_init__(self):
        if self.top_k_list is None:
            self.top_k_list = [5, 10, 20]
        if self.metrics is None:
            self.metrics = ["ndcg", "recall", "precision", "hit_rate"]
    
    def save_config(self, filepath: str) -> None:
        """保存配置到文件"""
        config_dict = self.__dict__.copy()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'BaseConfig':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def update_config(self, **kwargs) -> 'BaseConfig':
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self


@dataclass  
class MovieLensConfig(BaseConfig):
    """MovieLens数据集专用配置"""
    
    # MovieLens特定配置
    min_rating: float = 1.0
    max_rating: float = 5.0
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    
    # 电影特征配置
    use_genres: bool = True
    use_year: bool = True
    use_title_embedding: bool = True
    max_genres: int = 20
    
    # 用户特征配置
    use_user_demographics: bool = False  # MovieLens Small没有用户信息
    age_bins: int = 7
    
    # 负采样配置
    negative_sampling_ratio: float = 4.0
    negative_sampling_strategy: str = "uniform"  # uniform, popularity
    
    def __post_init__(self):
        super().__post_init__()
        # MovieLens专用数据路径
        self.dataset_path = "data/movielens/"
        self.cache_dir = "cache/movielens/"
        self.log_dir = "logs/movielens/"
        self.model_save_dir = "saved_models/movielens/"


def get_default_config(dataset: str = "movielens") -> BaseConfig:
    """获取默认配置"""
    if dataset.lower() == "movielens":
        return MovieLensConfig()
    else:
        return BaseConfig()


def load_config_from_file(filepath: str, dataset: str = "movielens") -> BaseConfig:
    """从文件加载配置"""
    if dataset.lower() == "movielens":
        return MovieLensConfig.load_config(filepath)
    else:
        return BaseConfig.load_config(filepath)


if __name__ == "__main__":
    # 测试配置功能
    system_config = SystemConfig()
    dataset_config = DatasetConfig()
    print("System Config:")
    print(f"Log level: {system_config.log_level}")
    print(f"Use CUDA: {system_config.use_cuda}")
    print(f"Dataset: {dataset_config.name}")
    print(f"Batch size: {dataset_config.batch_size}")
