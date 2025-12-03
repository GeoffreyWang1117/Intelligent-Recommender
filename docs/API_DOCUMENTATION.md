# API Documentation

## 📚 Overview

The Intelligent Recommender System provides a RESTful API for generating personalized recommendations.

**Base URL**: `http://localhost:5000`

**Version**: 1.0.0

**License**: CC BY-NC-SA 4.0

---

## 🔑 Authentication

Currently, the API does not require authentication. Future versions will implement API key-based authentication.

---

## 📊 Rate Limiting

- **Rate**: 100 requests per minute per IP address
- **Headers**: Rate limit information will be included in response headers

---

## 🚀 Endpoints

### Health Check

#### GET /health

Check the health status of the API service.

**Parameters**: None

**Response** (200 OK):
```json
{
  "status": "healthy",
  "services": {
    "svd_model": true,
    "faiss_index": true
  },
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**Response** (503 Service Unavailable):
```json
{
  "status": "unhealthy",
  "message": "Service not initialized",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**Example**:
```bash
curl http://localhost:5000/health
```

---

### Get Recommendations

#### GET /recommend

Get personalized recommendations for a user.

**Parameters**:
- `user_id` (required): Integer - User ID for whom to generate recommendations
- `top_k` (optional): Integer (1-100, default 10) - Number of recommendations to return

**Response** (200 OK):
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "item_id": 456,
      "score": 4.5,
      "title": "The Matrix",
      "genres": "Action|Sci-Fi",
      "year": 1999,
      "genre_list": ["Action", "Sci-Fi"]
    },
    {
      "item_id": 789,
      "score": 4.2,
      "title": "Inception",
      "genres": "Action|Sci-Fi|Thriller",
      "year": 2010,
      "genre_list": ["Action", "Sci-Fi", "Thriller"]
    }
  ],
  "count": 2,
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**Response** (400 Bad Request):
```json
{
  "error": "user_id parameter must be provided",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**Examples**:
```bash
# Get 10 recommendations for user 1
curl "http://localhost:5000/recommend?user_id=1"

# Get 5 recommendations for user 100
curl "http://localhost:5000/recommend?user_id=100&top_k=5"
```

---

### Get Similar Items

#### GET /similar

Find items similar to a given item.

**Parameters**:
- `item_id` (required): Integer - Item ID for which to find similar items
- `top_k` (optional): Integer (1-100, default 10) - Number of similar items to return

**Response** (200 OK):
```json
{
  "item_id": 456,
  "item_info": {
    "title": "The Matrix",
    "genres": "Action|Sci-Fi"
  },
  "similar_items": [
    {
      "item_id": 789,
      "score": 0.95,
      "title": "Inception",
      "genres": "Action|Sci-Fi|Thriller",
      "year": 2010
    },
    {
      "item_id": 101,
      "score": 0.89,
      "title": "Blade Runner",
      "genres": "Sci-Fi|Thriller",
      "year": 1982
    }
  ],
  "count": 2,
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**Response** (400 Bad Request):
```json
{
  "error": "item_id parameter must be provided",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**Examples**:
```bash
# Get 10 similar items for item 1
curl "http://localhost:5000/similar?item_id=1"

# Get 5 similar items for item 50
curl "http://localhost:5000/similar?item_id=50&top_k=5"
```

---

### Get Model Information

#### GET /info

Get information about the recommendation model.

**Parameters**: None

**Response** (200 OK):
```json
{
  "model_type": "SVD",
  "n_components": 50,
  "n_users": 943,
  "n_items": 1682,
  "has_faiss_index": true,
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**Response** (503 Service Unavailable):
```json
{
  "error": "Model not initialized",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**Example**:
```bash
curl http://localhost:5000/info
```

---

## 🔢 Status Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 400  | Bad Request - Invalid parameters |
| 404  | Not Found - Endpoint does not exist |
| 405  | Method Not Allowed - Wrong HTTP method |
| 500  | Internal Server Error |
| 503  | Service Unavailable - Service not initialized |

---

## 📝 Error Responses

All error responses follow this format:

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

For 404 errors, additional information is provided:

```json
{
  "error": "Endpoint not found",
  "message": "Please check the request URL",
  "available_endpoints": [
    "/ [GET] - Main page",
    "/health [GET] - Health check",
    "/recommend [GET] - User recommendations",
    "/similar [GET] - Similar items",
    "/info [GET] - Model information"
  ],
  "timestamp": "2025-01-01T12:00:00Z"
}
```

---

## 🌐 OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **File**: `docs/openapi.yaml`
- **Swagger UI**: Coming soon

To view the OpenAPI spec:
```bash
# Using swagger-ui-express (Node.js)
npm install -g swagger-ui-express
swagger-ui docs/openapi.yaml

# Using online editor
# Visit: https://editor.swagger.io/
# Then import docs/openapi.yaml
```

---

## 🧪 Testing

### Using cURL

```bash
# Health check
curl -i http://localhost:5000/health

# Get recommendations
curl -i "http://localhost:5000/recommend?user_id=1&top_k=5"

# Get similar items
curl -i "http://localhost:5000/similar?item_id=1&top_k=5"

# Get model info
curl -i http://localhost:5000/info
```

### Using Python

```python
import requests

# Base URL
BASE_URL = "http://localhost:5000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Get recommendations
params = {'user_id': 1, 'top_k': 10}
response = requests.get(f"{BASE_URL}/recommend", params=params)
recommendations = response.json()
print(f"Got {len(recommendations['recommendations'])} recommendations")

# Get similar items
params = {'item_id': 1, 'top_k': 5}
response = requests.get(f"{BASE_URL}/similar", params=params)
similar_items = response.json()
print(f"Got {len(similar_items['similar_items'])} similar items")
```

### Using the Test Suite

```bash
# Run API tests
make test-api

# Or directly
python tests/test_api.py --url http://localhost:5000
```

---

## 🚦 Performance

Expected response times:
- Health check: < 10ms
- Recommendations: < 100ms (P95)
- Similar items: < 100ms (P95)
- Model info: < 10ms

Expected throughput:
- QPS: > 1000 queries per second

---

## 🔮 Future Enhancements

Planned features:
- [ ] API key authentication
- [ ] Rate limiting headers
- [ ] Batch recommendations
- [ ] Real-time model updates
- [ ] A/B testing support
- [ ] Recommendation explanations
- [ ] User feedback endpoints
- [ ] WebSocket support for real-time updates

---

## 📞 Support

For issues or questions:
- **GitHub Issues**: https://github.com/GeoffreyWang1117/Intelligent-Recommender/issues
- **Documentation**: Check project README and docs/

---

*Last updated: 2025-11-18*
