import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Redis配置
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = int(os.getenv('REDIS_DB', '0'))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
    
    # 模型配置
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/saved/')
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'models/saved/faiss_index.bin')
    
    # 数据配置
    DATA_PATH = os.getenv('DATA_PATH', 'data/ml-1m/')
    
    # API配置
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '5000'))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # 推荐配置
    DEFAULT_TOP_K = int(os.getenv('DEFAULT_TOP_K', '10'))
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))  # 缓存时间 (秒)
    
    # LightFM模型参数
    LIGHTFM_COMPONENTS = int(os.getenv('LIGHTFM_COMPONENTS', '100'))
    LIGHTFM_EPOCHS = int(os.getenv('LIGHTFM_EPOCHS', '50'))
    LIGHTFM_LEARNING_RATE = float(os.getenv('LIGHTFM_LEARNING_RATE', '0.05'))
    
    # FAISS配置
    FAISS_NLIST = int(os.getenv('FAISS_NLIST', '100'))
    FAISS_DIMENSION = int(os.getenv('FAISS_DIMENSION', '100'))
