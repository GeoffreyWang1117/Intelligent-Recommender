import redis
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class CacheService:
    """Redis缓存服务"""
    
    def __init__(self, host='localhost', port=6379, db=0, password=None, 
                 default_ttl=3600):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """连接Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            # 测试连接
            self.redis_client.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            self.redis_client = None
    
    def is_connected(self):
        """检查Redis连接状态"""
        try:
            if self.redis_client is None:
                return False
            self.redis_client.ping()
            return True
        except:
            return False
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        try:
            if not self.is_connected():
                logger.warning("Redis未连接，跳过缓存设置")
                return False
            
            ttl = ttl or self.default_ttl
            value_str = json.dumps(value, ensure_ascii=False)
            
            result = self.redis_client.setex(key, ttl, value_str)
            return result
            
        except Exception as e:
            logger.error(f"设置缓存失败 {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            if not self.is_connected():
                return None
            
            value_str = self.redis_client.get(key)
            if value_str is None:
                return None
            
            return json.loads(value_str)
            
        except Exception as e:
            logger.error(f"获取缓存失败 {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            if not self.is_connected():
                return False
            
            result = self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"删除缓存失败 {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            if not self.is_connected():
                return False
            
            return self.redis_client.exists(key) > 0
            
        except Exception as e:
            logger.error(f"检查缓存存在性失败 {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """设置缓存过期时间"""
        try:
            if not self.is_connected():
                return False
            
            return self.redis_client.expire(key, ttl)
            
        except Exception as e:
            logger.error(f"设置缓存过期时间失败 {key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """获取缓存剩余时间"""
        try:
            if not self.is_connected():
                return -1
            
            return self.redis_client.ttl(key)
            
        except Exception as e:
            logger.error(f"获取缓存TTL失败 {key}: {e}")
            return -1
    
    def flush_db(self) -> bool:
        """清空当前数据库"""
        try:
            if not self.is_connected():
                return False
            
            self.redis_client.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            return False
    
    def keys(self, pattern: str = "*"):
        """获取匹配的键"""
        try:
            if not self.is_connected():
                return []
            
            return self.redis_client.keys(pattern)
            
        except Exception as e:
            logger.error(f"获取键列表失败: {e}")
            return []
    
    def info(self):
        """获取Redis信息"""
        try:
            if not self.is_connected():
                return None
            
            return self.redis_client.info()
            
        except Exception as e:
            logger.error(f"获取Redis信息失败: {e}")
            return None
    
    def generate_recommendation_key(self, user_id: int, top_k: int = 10) -> str:
        """生成推荐缓存键"""
        return f"rec:user:{user_id}:top_k:{top_k}"
    
    def generate_similar_key(self, item_id: int, top_k: int = 10) -> str:
        """生成相似物品缓存键"""
        return f"sim:item:{item_id}:top_k:{top_k}"
    
    def generate_user_profile_key(self, user_id: int) -> str:
        """生成用户画像缓存键"""
        return f"profile:user:{user_id}"
    
    def generate_item_features_key(self, item_id: int) -> str:
        """生成物品特征缓存键"""
        return f"features:item:{item_id}"
