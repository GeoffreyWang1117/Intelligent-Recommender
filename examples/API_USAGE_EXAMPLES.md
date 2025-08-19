# API使用示例

## 1. 启动服务

### 本地启动
```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据并训练模型
python scripts/train_model.py --download

# 启动Redis
redis-server

# 启动API服务
python app.py
```

### Docker启动
```bash
# 构建并启动
docker-compose up --build

# 后台运行
docker-compose up -d
```

## 2. API使用示例

### 健康检查
```bash
curl -X GET "http://localhost:5000/health"
```

**响应:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "redis_connected": true,
  "faiss_loaded": true,
  "timestamp": "2025-08-17T10:30:00Z"
}
```

### 获取用户推荐
```bash
curl -X GET "http://localhost:5000/recommend?user_id=123&top_k=10"
```

**响应:**
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "item_id": 1234,
      "score": 0.85,
      "title": "Toy Story (1995)",
      "genres": ["Animation", "Children's", "Comedy"],
      "cached": false
    }
  ],
  "cached": false,
  "timestamp": "2025-08-17T10:30:00Z"
}
```

### 获取相似物品
```bash
curl -X GET "http://localhost:5000/similar?item_id=1234&top_k=5"
```

**响应:**
```json
{
  "item_id": 1234,
  "similar_items": [
    {
      "item_id": 5678,
      "similarity": 0.95,
      "title": "Toy Story 2 (1999)",
      "genres": ["Animation", "Children's", "Comedy"],
      "cached": false
    }
  ],
  "cached": false,
  "timestamp": "2025-08-17T10:30:00Z"
}
```

### 更新用户评分
```bash
curl -X POST "http://localhost:5000/rating" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "item_id": 1234,
    "rating": 4.5
  }'
```

**响应:**
```json
{
  "message": "评分更新成功",
  "user_id": 123,
  "item_id": 1234,
  "rating": 4.5,
  "timestamp": "2025-08-17T10:30:00Z"
}
```

### 获取模型信息
```bash
curl -X GET "http://localhost:5000/info"
```

**响应:**
```json
{
  "lightfm_loaded": true,
  "faiss_loaded": true,
  "cache_connected": true,
  "item_metadata_count": 3883,
  "faiss_info": {
    "dimension": 100,
    "index_type": "IVFFlat",
    "total_vectors": 3883,
    "is_trained": true
  },
  "timestamp": "2025-08-17T10:30:00Z"
}
```

## 3. Python客户端示例

```python
import requests
import json

class RecommendationClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
    
    def get_recommendations(self, user_id, top_k=10):
        """获取用户推荐"""
        response = requests.get(
            f"{self.base_url}/recommend",
            params={'user_id': user_id, 'top_k': top_k}
        )
        return response.json()
    
    def get_similar_items(self, item_id, top_k=10):
        """获取相似物品"""
        response = requests.get(
            f"{self.base_url}/similar",
            params={'item_id': item_id, 'top_k': top_k}
        )
        return response.json()
    
    def update_rating(self, user_id, item_id, rating):
        """更新评分"""
        response = requests.post(
            f"{self.base_url}/rating",
            json={
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating
            }
        )
        return response.json()

# 使用示例
client = RecommendationClient()

# 获取推荐
recs = client.get_recommendations(user_id=1, top_k=5)
print(f"推荐结果: {len(recs['recommendations'])} 个物品")

# 获取相似物品
similar = client.get_similar_items(item_id=1, top_k=5)
print(f"相似物品: {len(similar['similar_items'])} 个")

# 更新评分
result = client.update_rating(user_id=1, item_id=100, rating=4.5)
print(f"评分更新: {result['message']}")
```

## 4. 性能测试

```bash
# 使用API测试脚本
python tests/test_api.py --url http://localhost:5000 --test all

# 压力测试（需要安装apache2-utils）
ab -n 1000 -c 10 "http://localhost:5000/recommend?user_id=1"
```

## 5. 监控和日志

### 查看实时日志
```bash
# Docker方式
docker-compose logs -f recommendation-api

# 本地方式
tail -f logs/app.log
```

### Redis监控
```bash
# 连接Redis
redis-cli

# 查看缓存统计
INFO stats

# 查看缓存键
KEYS rec:*
```
