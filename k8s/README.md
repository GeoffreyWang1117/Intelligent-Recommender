# Kubernetes Deployment Guide

## 📋 Overview

This directory contains Kubernetes manifests for deploying the Intelligent Recommender System.

## 📁 Files

- `deployment.yaml` - Main application deployment
- `service.yaml` - Kubernetes services
- `configmap.yaml` - Configuration data
- `secrets.yaml` - Sensitive data (template)
- `hpa.yaml` - Horizontal Pod Autoscaler
- `ingress.yaml` - Ingress controller configuration
- `redis-deployment.yaml` - Redis cache deployment

## 🚀 Quick Start

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Ingress controller (nginx recommended)
- StorageClass available

### Deploy All Resources

```bash
# 1. Create namespace
kubectl create namespace recommender

# 2. Apply configurations
kubectl apply -f k8s/ -n recommender

# 3. Check deployment status
kubectl get all -n recommender

# 4. Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=intelligent-recommender -n recommender --timeout=300s
```

### Step-by-Step Deployment

```bash
# 1. Deploy Redis
kubectl apply -f k8s/redis-deployment.yaml -n recommender

# 2. Create ConfigMap
kubectl apply -f k8s/configmap.yaml -n recommender

# 3. Create Secrets (customize first!)
kubectl apply -f k8s/secrets.yaml -n recommender

# 4. Deploy application
kubectl apply -f k8s/deployment.yaml -n recommender

# 5. Create services
kubectl apply -f k8s/service.yaml -n recommender

# 6. Setup autoscaling
kubectl apply -f k8s/hpa.yaml -n recommender

# 7. Configure ingress
kubectl apply -f k8s/ingress.yaml -n recommender
```

## ⚙️ Configuration

### Environment Variables

Edit `configmap.yaml` to customize:

- `redis_host`: Redis hostname
- `redis_port`: Redis port
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `default_top_k`: Default number of recommendations
- `cache_ttl`: Cache TTL in seconds

### Secrets

**Important**: Update `secrets.yaml` with your actual credentials before deploying!

```bash
# Create secrets from command line (recommended)
kubectl create secret generic recommender-secrets \
  --from-literal=redis_password='your-redis-password' \
  --from-literal=api_key='your-api-key' \
  -n recommender
```

### Resource Limits

Adjust in `deployment.yaml`:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Autoscaling

Modify `hpa.yaml` to adjust scaling behavior:

- `minReplicas`: Minimum number of pods (default: 3)
- `maxReplicas`: Maximum number of pods (default: 10)
- `targetCPUUtilizationPercentage`: CPU threshold (default: 70%)

## 🔍 Monitoring

### Check Pod Status

```bash
# List all pods
kubectl get pods -n recommender

# Describe pod
kubectl describe pod <pod-name> -n recommender

# View logs
kubectl logs -f <pod-name> -n recommender

# View logs for all pods
kubectl logs -f -l app=intelligent-recommender -n recommender
```

### Check HPA Status

```bash
# View HPA status
kubectl get hpa -n recommender

# Describe HPA
kubectl describe hpa intelligent-recommender-hpa -n recommender
```

### Check Services

```bash
# List services
kubectl get svc -n recommender

# Test service internally
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -n recommender \
  -- curl http://intelligent-recommender/health
```

## 🧪 Testing

### Port Forward for Local Testing

```bash
# Forward API port
kubectl port-forward -n recommender svc/intelligent-recommender 5000:80

# Test API
curl http://localhost:5000/health
```

### Execute Commands in Pod

```bash
# Get shell in pod
kubectl exec -it <pod-name> -n recommender -- /bin/bash

# Run tests inside pod
kubectl exec -it <pod-name> -n recommender -- python -m pytest tests/
```

## 📊 Scaling

### Manual Scaling

```bash
# Scale deployment
kubectl scale deployment intelligent-recommender --replicas=5 -n recommender

# Verify scaling
kubectl get pods -n recommender
```

### Autoscaling Configuration

The HPA automatically scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)

Scale up/down behavior:
- Scale up: Quick (30s stabilization)
- Scale down: Gradual (300s stabilization)

## 🔄 Updates and Rollbacks

### Rolling Update

```bash
# Update image
kubectl set image deployment/intelligent-recommender \
  recommender-api=intelligent-recommender:v1.1 \
  -n recommender

# Check rollout status
kubectl rollout status deployment/intelligent-recommender -n recommender
```

### Rollback

```bash
# View rollout history
kubectl rollout history deployment/intelligent-recommender -n recommender

# Rollback to previous version
kubectl rollout undo deployment/intelligent-recommender -n recommender

# Rollback to specific revision
kubectl rollout undo deployment/intelligent-recommender --to-revision=2 -n recommender
```

## 🗑️ Cleanup

### Delete All Resources

```bash
# Delete all resources
kubectl delete -f k8s/ -n recommender

# Delete namespace (removes everything)
kubectl delete namespace recommender
```

### Delete Specific Resources

```bash
# Delete deployment only
kubectl delete deployment intelligent-recommender -n recommender

# Delete service
kubectl delete service intelligent-recommender -n recommender
```

## 🔐 Security Best Practices

1. **Secrets Management**
   - Use external secret management (e.g., Sealed Secrets, Vault)
   - Never commit real secrets to git
   - Rotate secrets regularly

2. **Network Policies**
   - Implement network policies to restrict traffic
   - Use service mesh for advanced traffic management

3. **Resource Limits**
   - Always set resource requests and limits
   - Prevents resource starvation

4. **RBAC**
   - Create service accounts with minimal permissions
   - Implement RBAC policies

5. **Image Security**
   - Use specific image tags (not :latest)
   - Scan images for vulnerabilities
   - Use minimal base images

## 🌐 Ingress Configuration

### Prerequisites

Install nginx ingress controller:

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
```

### TLS/SSL

Install cert-manager for automatic TLS:

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Custom Domain

Update `ingress.yaml` with your domain:

```yaml
spec:
  rules:
  - host: recommender.yourdomain.com  # Change this
```

## 📈 Performance Tuning

### Redis Optimization

```yaml
# In redis-deployment.yaml, add:
args:
  - redis-server
  - --maxmemory
  - 512mb
  - --maxmemory-policy
  - allkeys-lru
```

### API Optimization

```yaml
# In deployment.yaml, adjust:
- name: MAX_WORKERS
  value: "4"  # Adjust based on CPU
```

## 🐛 Troubleshooting

### Common Issues

**Pods not starting**
```bash
kubectl describe pod <pod-name> -n recommender
kubectl logs <pod-name> -n recommender
```

**Service not accessible**
```bash
kubectl get endpoints -n recommender
kubectl describe svc intelligent-recommender -n recommender
```

**HPA not scaling**
```bash
# Check metrics server
kubectl get apiservice v1beta1.metrics.k8s.io -o yaml

# View HPA events
kubectl get events -n recommender | grep HPA
```

**Image pull errors**
```bash
# Check image pull secrets
kubectl get secrets -n recommender

# Describe pod for more details
kubectl describe pod <pod-name> -n recommender
```

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Ingress Controllers](https://kubernetes.io/docs/concepts/services-networking/ingress-controllers/)

## 🆘 Support

For issues or questions:
- Check logs: `kubectl logs -f -l app=intelligent-recommender -n recommender`
- Describe resources: `kubectl describe <resource> <name> -n recommender`
- Check events: `kubectl get events -n recommender --sort-by='.lastTimestamp'`

---

*Last updated: 2025-11-18*
