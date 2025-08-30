"""
Teacher配置管理
定义各种Teacher的配置参数
"""

from dataclasses import dataclass
from typing import List, Dict
from .base_config import BaseConfig


@dataclass
class EnsembleTeacherConfig:
    """Ensemble Teacher配置 (SVD + xDeepFM + AutoInt)"""
    
    # Ensemble组合权重
    teacher_weights: Dict[str, float] = None
    
    # SVD配置
    svd_factors: int = 100
    svd_epochs: int = 20
    svd_lr: float = 0.005
    svd_reg: float = 0.02
    
    # xDeepFM配置
    xdeepfm_embedding_dim: int = 32
    xdeepfm_hidden_dims: List[int] = None
    xdeepfm_cross_layers: int = 3
    xdeepfm_dropout: float = 0.1
    
    # AutoInt配置
    autoint_attention_dim: int = 32
    autoint_num_heads: int = 4
    autoint_num_layers: int = 3
    autoint_dropout: float = 0.1
    
    # 模型路径配置
    svd_model_path: str = "models/saved/SVD_real_movielens.pkl"
    xdeepfm_model_path: str = "models/saved/xDeepFM_model.pth"
    autoint_model_path: str = "models/saved/AutoInt_model.pth"
    
    def __post_init__(self):
        if self.teacher_weights is None:
            # 基于已验证性能的默认权重
            self.teacher_weights = {
                "svd": 0.5,      # 最佳推荐质量 (Recall@10: 0.0297)
                "xdeepfm": 0.3,  # 最佳评分预测 (RMSE: 0.491)  
                "autoint": 0.2   # 注意力机制增强多样性
            }
        if self.xdeepfm_hidden_dims is None:
            self.xdeepfm_hidden_dims = [128, 64, 32]


@dataclass
class LLMTeacherConfig:
    """LLM Teacher配置 (Llama3.1)"""
    
    # LLM基础配置
    model_name: str = "llama3.1"
    model_path: str = "models/llama3.1/"
    api_endpoint: str = "http://localhost:11434/api/generate"
    max_tokens: int = 512
    temperature: float = 0.7
    
    # MovieLens专用提示配置
    system_prompt: str = ""
    recommendation_prompt_template: str = ""
    max_recommendations: int = 20
    
    # 语义嵌入配置
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # 推理优化配置
    batch_size: int = 8
    cache_responses: bool = True
    cache_ttl: int = 3600  # 缓存1小时
    timeout: int = 30      # 30秒超时
    
    # 错误处理配置
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_enabled: bool = True
    
    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = """You are an expert movie recommendation system. 
            Based on user's movie watching history and preferences, provide personalized 
            movie recommendations with clear reasoning."""
            
        if not self.recommendation_prompt_template:
            self.recommendation_prompt_template = """
            User Profile:
            - Previous movies: {user_history}
            - Preferred genres: {preferred_genres}
            - Average rating: {avg_rating}
            
            Please recommend {num_recommendations} movies and explain why each recommendation 
            fits the user's preferences. Format as JSON with movie_id, title, and reasoning.
            """


@dataclass
class FusionConfig:
    """多Teacher融合配置"""
    
    # Fisher引导融合配置
    use_fisher_guidance: bool = True
    fisher_analysis_path: str = "analysis_unified/ensemble_fisher_analysis.json"
    fisher_weight_threshold: float = 0.01
    
    # 动态权重配置
    enable_dynamic_weighting: bool = True
    context_aware_fusion: bool = True
    user_preference_adaptation: bool = True
    
    # 融合策略配置
    fusion_strategy: str = "weighted_average"  # weighted_average, attention, stacking
    attention_heads: int = 4
    fusion_hidden_dim: int = 64
    
    # 置信度配置
    confidence_calculation: str = "teacher_agreement"  # teacher_agreement, variance, entropy
    min_confidence_threshold: float = 0.3
    
    # 性能优化配置
    enable_caching: bool = True
    parallel_inference: bool = True
    max_concurrent_teachers: int = 4
    
    # 解释性配置
    provide_explanations: bool = True
    explanation_detail_level: str = "medium"  # low, medium, high
    include_teacher_contributions: bool = True


@dataclass
class MultiTeacherConfig(BaseConfig):
    """多Teacher系统完整配置"""
    
    # Teacher子配置
    ensemble_config: EnsembleTeacherConfig = None
    llm_config: LLMTeacherConfig = None
    fusion_config: FusionConfig = None
    
    # 系统级配置
    enable_monitoring: bool = True
    performance_logging: bool = True
    debug_mode: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.ensemble_config is None:
            self.ensemble_config = EnsembleTeacherConfig()
        if self.llm_config is None:
            self.llm_config = LLMTeacherConfig()
        if self.fusion_config is None:
            self.fusion_config = FusionConfig()
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        # 验证Teacher权重
        weights = self.ensemble_config.teacher_weights
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("Ensemble teacher weights must sum to 1.0")
        
        # 验证文件路径
        import os
        if not os.path.exists(self.fusion_config.fisher_analysis_path):
            print(f"Warning: Fisher analysis file not found: {self.fusion_config.fisher_analysis_path}")
        
        return True


def get_multi_teacher_config() -> MultiTeacherConfig:
    """获取默认多Teacher配置"""
    return MultiTeacherConfig()


if __name__ == "__main__":
    # 测试配置
    config = get_multi_teacher_config()
    config.validate_config()
    
    print("Multi-Teacher Configuration:")
    print(f"Ensemble weights: {config.ensemble_config.teacher_weights}")
    print(f"LLM model: {config.llm_config.model_name}")
    print(f"Fusion strategy: {config.fusion_config.fusion_strategy}")
    print(f"Fisher guidance: {config.fusion_config.use_fisher_guidance}")
