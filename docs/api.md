# API 接口文档

## 概述

推荐系统提供RESTful API接口，支持推荐、训练、评估等核心功能。所有接口返回JSON格式数据。

## 基础信息

- **Base URL**: `http://localhost:5000/api/v1`
- **Content-Type**: `application/json`
- **Authentication**: 暂不需要认证（开发阶段）

## 推荐接口

### 获取用户推荐

获取指定用户的个性化推荐列表。

**请求**
```
GET /api/v1/recommendations/{user_id}
```

**参数**
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| user_id | int | 是 | - | 用户ID |
| top_k | int | 否 | 10 | 推荐数量 |
| algorithm | string | 否 | ensemble | 算法名称 |

**响应示例**
```json
{
    "user_id": 123,
    "recommendations": [
        {
            "item_id": 456,
            "score": 0.89,
            "title": "推荐电影标题",
            "genres": ["Action", "Sci-Fi"],
            "year": 2023,
            "explanation": {
                "primary_reason": "基于您喜欢的科幻类型推荐",
                "supporting_factors": ["类型偏好匹配", "高质量评分"],
                "confidence": 0.85
            },
            "algorithm_details": {
                "deepfm": {"score": 0.85, "weight": 0.4},
                "autoint": {"score": 0.93, "weight": 0.6}
            }
        }
    ],
    "total_count": 10,
    "algorithm_used": "ensemble",
    "response_time_ms": 45,
    "timestamp": "2024-08-18T10:30:00Z"
}
```

### 获取物品相似推荐

获取与指定物品相似的推荐列表。

**请求**
```
GET /api/v1/items/{item_id}/similar
```

**参数**
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| item_id | int | 是 | - | 物品ID |
| top_k | int | 否 | 10 | 推荐数量 |

**响应示例**
```json
{
    "item_id": 123,
    "similar_items": [
        {
            "item_id": 456,
            "similarity": 0.92,
            "title": "相似电影标题",
            "genres": ["Action", "Sci-Fi"],
            "year": 2023
        }
    ],
    "total_count": 10,
    "response_time_ms": 25
}
```

## 模型管理接口

### 训练模型

启动模型训练任务。

**请求**
```
POST /api/v1/models/train
```

**请求体**
```json
{
    "algorithm": "deepfm",
    "parameters": {
        "embedding_dim": 8,
        "epochs": 10,
        "batch_size": 256,
        "learning_rate": 0.001
    },
    "data_path": "data/train.csv",
    "model_name": "deepfm_v1"
}
```

**响应示例**
```json
{
    "task_id": "train_123456",
    "status": "started",
    "algorithm": "deepfm",
    "estimated_time_minutes": 30,
    "start_time": "2024-08-18T10:30:00Z"
}
```

### 查询训练状态

查询模型训练任务的状态。

**请求**
```
GET /api/v1/models/train/{task_id}
```

**响应示例**
```json
{
    "task_id": "train_123456",
    "status": "completed",
    "progress": 100,
    "algorithm": "deepfm",
    "start_time": "2024-08-18T10:30:00Z",
    "end_time": "2024-08-18T11:00:00Z",
    "duration_minutes": 30,
    "model_id": "deepfm_20240818_110000",
    "metrics": {
        "train_loss": 0.65,
        "validation_loss": 0.72
    }
}
```

### 模型评估

评估指定模型的性能。

**请求**
```
GET /api/v1/models/{model_id}/evaluate
```

**参数**
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| model_id | string | 是 | - | 模型ID |
| test_data_path | string | 否 | data/test.csv | 测试数据路径 |

**响应示例**
```json
{
    "model_id": "deepfm_20240818_110000",
    "algorithm": "deepfm",
    "metrics": {
        "rmse": 0.85,
        "mae": 0.67,
        "precision@5": 0.28,
        "precision@10": 0.23,
        "recall@5": 0.15,
        "recall@10": 0.18,
        "ndcg@5": 0.35,
        "ndcg@10": 0.31,
        "auc": 0.78
    },
    "evaluation_time": "2024-08-18T11:30:00Z",
    "test_data_size": 20000
}
```

### 获取模型列表

获取所有已训练的模型列表。

**请求**
```
GET /api/v1/models
```

**参数**
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| algorithm | string | 否 | - | 按算法筛选 |
| status | string | 否 | - | 按状态筛选 |
| limit | int | 否 | 20 | 返回数量限制 |

**响应示例**
```json
{
    "models": [
        {
            "model_id": "deepfm_20240818_110000",
            "algorithm": "deepfm",
            "status": "active",
            "created_time": "2024-08-18T11:00:00Z",
            "metrics": {
                "rmse": 0.85,
                "precision@10": 0.23
            }
        }
    ],
    "total_count": 15,
    "page": 1,
    "page_size": 20
}
```

## 配置管理接口

### 获取算法配置

获取指定算法的配置信息。

**请求**
```
GET /api/v1/config/algorithms/{algorithm_name}
```

**响应示例**
```json
{
    "algorithm": "deepfm",
    "enabled": true,
    "parameters": {
        "embedding_dim": 8,
        "hidden_dims": [256, 128, 64],
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 256
    },
    "description": "DeepFM algorithm for CTR prediction"
}
```

### 更新算法配置

更新指定算法的配置。

**请求**
```
PUT /api/v1/config/algorithms/{algorithm_name}
```

**请求体**
```json
{
    "enabled": true,
    "parameters": {
        "embedding_dim": 16,
        "epochs": 20
    }
}
```

**响应示例**
```json
{
    "message": "Algorithm configuration updated successfully",
    "algorithm": "deepfm",
    "updated_at": "2024-08-18T12:00:00Z"
}
```

## 用户管理接口

### 获取用户信息

获取指定用户的基本信息和偏好。

**请求**
```
GET /api/v1/users/{user_id}
```

**响应示例**
```json
{
    "user_id": 123,
    "profile": {
        "age_group": "25-34",
        "gender": "M",
        "occupation": "Engineer"
    },
    "preferences": {
        "favorite_genres": ["Action", "Sci-Fi", "Thriller"],
        "avg_rating": 3.8,
        "total_ratings": 150
    },
    "last_activity": "2024-08-18T10:00:00Z"
}
```

### 更新用户评分

添加或更新用户对物品的评分。

**请求**
```
POST /api/v1/users/{user_id}/ratings
```

**请求体**
```json
{
    "item_id": 456,
    "rating": 4.5,
    "timestamp": "2024-08-18T12:00:00Z"
}
```

**响应示例**
```json
{
    "message": "Rating updated successfully",
    "user_id": 123,
    "item_id": 456,
    "rating": 4.5,
    "updated_at": "2024-08-18T12:00:00Z"
}
```

## 系统监控接口

### 获取系统状态

获取系统整体运行状态。

**请求**
```
GET /api/v1/health
```

**响应示例**
```json
{
    "status": "healthy",
    "timestamp": "2024-08-18T12:00:00Z",
    "version": "1.0.0",
    "uptime_seconds": 86400,
    "services": {
        "database": "healthy",
        "redis": "healthy",
        "model_service": "healthy"
    },
    "metrics": {
        "total_requests": 10000,
        "avg_response_time_ms": 45,
        "error_rate": 0.1
    }
}
```

### 获取性能指标

获取系统性能指标。

**请求**
```
GET /api/v1/metrics
```

**参数**
| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| start_time | string | 否 | 1h ago | 开始时间 |
| end_time | string | 否 | now | 结束时间 |
| interval | string | 否 | 5m | 时间间隔 |

**响应示例**
```json
{
    "time_range": {
        "start": "2024-08-18T11:00:00Z",
        "end": "2024-08-18T12:00:00Z"
    },
    "metrics": [
        {
            "timestamp": "2024-08-18T11:00:00Z",
            "requests_per_second": 50,
            "avg_response_time_ms": 45,
            "error_rate": 0.1,
            "cpu_usage": 65.2,
            "memory_usage": 78.5
        }
    ]
}
```

## 错误响应

所有接口在发生错误时返回统一格式的错误响应。

### 错误响应格式

```json
{
    "error": {
        "code": "INVALID_PARAMETER",
        "message": "Invalid user_id parameter",
        "details": "user_id must be a positive integer",
        "timestamp": "2024-08-18T12:00:00Z"
    }
}
```

### 常见错误码

| 错误码 | HTTP状态码 | 说明 |
|--------|------------|------|
| INVALID_PARAMETER | 400 | 请求参数无效 |
| USER_NOT_FOUND | 404 | 用户不存在 |
| MODEL_NOT_FOUND | 404 | 模型不存在 |
| ALGORITHM_NOT_SUPPORTED | 400 | 不支持的算法 |
| TRAINING_IN_PROGRESS | 409 | 训练任务进行中 |
| INTERNAL_ERROR | 500 | 内部服务器错误 |
| SERVICE_UNAVAILABLE | 503 | 服务暂时不可用 |

## 限流和配额

### 请求限制

- **推荐接口**: 每用户每分钟最多100次请求
- **训练接口**: 每小时最多10次训练任务
- **评估接口**: 每小时最多50次评估请求

### 配额响应头

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1692360000
```

## SDK和示例

### Python SDK示例

```python
import requests

class RecommenderClient:
    def __init__(self, base_url="http://localhost:5000/api/v1"):
        self.base_url = base_url
    
    def get_recommendations(self, user_id, top_k=10, algorithm="ensemble"):
        url = f"{self.base_url}/recommendations/{user_id}"
        params = {"top_k": top_k, "algorithm": algorithm}
        response = requests.get(url, params=params)
        return response.json()
    
    def train_model(self, algorithm, parameters, data_path):
        url = f"{self.base_url}/models/train"
        data = {
            "algorithm": algorithm,
            "parameters": parameters,
            "data_path": data_path
        }
        response = requests.post(url, json=data)
        return response.json()

# 使用示例
client = RecommenderClient()

# 获取推荐
recommendations = client.get_recommendations(user_id=123, top_k=5)

# 训练模型
task = client.train_model(
    algorithm="deepfm",
    parameters={"embedding_dim": 8, "epochs": 10},
    data_path="data/train.csv"
)
```

### JavaScript SDK示例

```javascript
class RecommenderClient {
    constructor(baseUrl = "http://localhost:5000/api/v1") {
        this.baseUrl = baseUrl;
    }
    
    async getRecommendations(userId, topK = 10, algorithm = "ensemble") {
        const url = `${this.baseUrl}/recommendations/${userId}`;
        const params = new URLSearchParams({
            top_k: topK,
            algorithm: algorithm
        });
        
        const response = await fetch(`${url}?${params}`);
        return await response.json();
    }
    
    async trainModel(algorithm, parameters, dataPath) {
        const url = `${this.baseUrl}/models/train`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                algorithm: algorithm,
                parameters: parameters,
                data_path: dataPath
            })
        });
        return await response.json();
    }
}

// 使用示例
const client = new RecommenderClient();

// 获取推荐
client.getRecommendations(123, 5).then(recommendations => {
    console.log(recommendations);
});
```

## 版本信息

- **API版本**: v1
- **最后更新**: 2024-08-18
- **维护者**: 推荐系统开发团队

---

更多详细信息请参考 [系统架构文档](../ARCHITECTURE.md) 和 [开发指南](development.md)。
