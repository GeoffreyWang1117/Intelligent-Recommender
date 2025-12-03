# Monitoring Configuration

## Overview

This directory contains monitoring and observability configurations for the Intelligent Recommender System.

## Components

- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Alert Manager**: Alert routing and management

## Quick Start

### Using Docker Compose

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access Grafana
open http://localhost:3000
# Default credentials: admin/admin

# Access Prometheus
open http://localhost:9090
```

### Kubernetes Deployment

```bash
# Install Prometheus Operator
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Deploy monitoring stack
kubectl apply -f monitoring/k8s/
```

## Metrics

### Application Metrics

- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request duration histogram
- `cache_hits_total`: Cache hit counter
- `cache_misses_total`: Cache miss counter
- `model_inference_duration_seconds`: Model inference time
- `recommendations_generated_total`: Total recommendations generated

### System Metrics

- CPU usage
- Memory usage
- Network I/O
- Disk I/O

## Dashboards

Pre-configured Grafana dashboards:
- API Performance Dashboard
- Cache Performance Dashboard
- Model Performance Dashboard
- System Resources Dashboard

## Alerts

Configured alerts:
- High error rate (> 5%)
- High latency (P95 > 200ms)
- Low cache hit rate (< 70%)
- High memory usage (> 90%)

## Configuration Files

- `prometheus.yml`: Prometheus configuration
- `grafana-dashboard.json`: Grafana dashboard template
- `alertmanager.yml`: Alert routing configuration
