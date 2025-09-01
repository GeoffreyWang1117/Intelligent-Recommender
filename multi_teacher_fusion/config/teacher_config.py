"""
Multi-Teacher Fusion System - Teacher Configuration
各种Teacher的配置定义
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EnsembleConfig:
    """Ensemble Teacher配置"""
    # 算法权重
    svd_weight: float = 0.5
    xdeepfm_weight: float = 0.3
    autoint_weight: float = 0.2
    
    # SVD参数
    num_factors: int = 50
    num_epochs: int = 100
    learning_rate: float = 0.01
    regularization: float = 0.02
    
    # 深度学习参数
    embedding_dim: int = 64
    hidden_dims: Optional[List[int]] = None
    dropout_rate: float = 0.1
    batch_size: int = 256
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]

@dataclass
class LLMConfig:
    """LLM Teacher配置"""
    model_name: str = "llama3:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: int = 30
    
    # 推荐配置
    max_history_items: int = 10
    explanation_enabled: bool = True
    batch_size: int = 1

@dataclass
class FusionConfig:
    """融合配置"""
    method: str = "weighted_average"  # weighted_average, rank_fusion, neural_fusion
    ensemble_weight: float = 0.7
    llm_weight: float = 0.3
    
    # 多样性配置
    diversification: bool = True
    diversity_weight: float = 0.1
    
    # 其他配置
    max_candidates: int = 100
    rerank_top_k: int = 50

@dataclass
class TeacherConfig:
    """全局Teacher配置"""
    ensemble_config: EnsembleConfig
    llm_config: LLMConfig
    fusion_config: FusionConfig
    
    def __init__(self, 
                 ensemble_config: Optional[EnsembleConfig] = None,
                 llm_config: Optional[LLMConfig] = None,
                 fusion_config: Optional[FusionConfig] = None):
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.llm_config = llm_config or LLMConfig()
        self.fusion_config = fusion_config or FusionConfig()

if __name__ == "__main__":
    # 测试配置
    config = TeacherConfig()
    
    print("Teacher Configurations:")
    print(f"Ensemble weights: SVD={config.ensemble_config.svd_weight}, xDeepFM={config.ensemble_config.xdeepfm_weight}, AutoInt={config.ensemble_config.autoint_weight}")
    print(f"LLM model: {config.llm_config.model_name}")
    print(f"Fusion method: {config.fusion_config.method}")
    print(f"Fusion weights: Ensemble={config.fusion_config.ensemble_weight}, LLM={config.fusion_config.llm_weight}")
