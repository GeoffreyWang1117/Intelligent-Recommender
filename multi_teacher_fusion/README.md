# 🚀 多Teacher融合系统 (Ensemble + Llama3.1)

> **项目目标**: 实现SVD+xDeepFM+AutoInt Ensemble与Llama3.1的全新融合架构  
> **创建时间**: 2024年8月30日  
> **开发重点**: 全新设计的多Teacher融合推荐系统

## 📋 项目概述

本目录包含全新设计的多Teacher融合推荐系统，专门针对MovieLens数据集优化，实现：
- **Ensemble Teacher**: SVD+xDeepFM+AutoInt三算法集成
- **LLM Teacher**: Llama3.1语义理解推荐
- **Fisher引导融合**: 基于已完成Fisher分析的智能权重分配
- **生产级API**: FastAPI + Redis + Docker部署架构

## 🏗️ 系统架构

```
multi_teacher_fusion/
├── README.md                          # 本文档
├── __init__.py
├── config/                             # 配置管理
│   ├── __init__.py
│   ├── base_config.py                  # 基础配置
│   ├── teacher_config.py               # Teacher配置
│   └── fusion_config.py                # 融合策略配置
├── teachers/                           # Teacher实现
│   ├── __init__.py
│   ├── base_teacher.py                 # Teacher基类
│   ├── ensemble_teacher.py             # 三算法Ensemble Teacher
│   ├── llama_teacher.py                # Llama3.1 Teacher
│   └── teacher_manager.py              # Teacher管理器
├── fusion/                             # 融合算法
│   ├── __init__.py
│   ├── fisher_guided_fusion.py         # Fisher引导融合核心
│   ├── weight_calculator.py            # 动态权重计算
│   └── fusion_strategies.py            # 融合策略集合
├── models/                             # 数据模型
│   ├── __init__.py
│   ├── user_profile.py                 # 用户画像
│   ├── item_profile.py                 # 物品画像
│   └── recommendation.py               # 推荐结果
├── api/                                # 生产API
│   ├── __init__.py
│   ├── main.py                         # FastAPI应用入口
│   ├── endpoints/                      # API端点
│   │   ├── __init__.py
│   │   ├── recommendation.py           # 推荐接口
│   │   └── health.py                   # 健康检查
│   └── middleware/                     # 中间件
│       ├── __init__.py
│       ├── cache.py                    # Redis缓存
│       └── logging.py                  # 日志中间件
├── utils/                              # 工具函数
│   ├── __init__.py
│   ├── data_loader.py                  # MovieLens数据加载
│   ├── metrics.py                      # 评估指标
│   └── visualization.py                # 结果可视化
├── experiments/                        # 实验脚本
│   ├── __init__.py
│   ├── benchmark_fusion.py             # 融合基准测试
│   ├── ablation_study.py               # 消融实验
│   └── performance_analysis.py         # 性能分析
├── tests/                              # 单元测试
│   ├── __init__.py
│   ├── test_teachers.py                # Teacher测试
│   ├── test_fusion.py                  # 融合算法测试
│   └── test_api.py                     # API测试
├── docker/                             # 容器化部署
│   ├── Dockerfile                      # Docker镜像
│   ├── docker-compose.yml              # 多容器编排
│   └── requirements.txt                # Python依赖
└── docs/                               # 文档
    ├── architecture.md                 # 架构设计文档
    ├── api_documentation.md            # API文档
    └── deployment_guide.md             # 部署指南
```

## 🎯 核心特性

### 1. 双Teacher架构
- **Ensemble Teacher**: 基于已验证的SVD+xDeepFM+AutoInt最优组合
- **LLM Teacher**: Llama3.1语义理解，支持电影推荐对话
- **统一接口**: 标准化Teacher接口，支持扩展

### 2. Fisher引导融合
- **智能权重**: 基于已完成Fisher分析的参数重要性权重
- **动态选择**: 根据查询特性动态调整Teacher权重
- **上下文感知**: 用户历史和偏好的个性化融合策略

### 3. 生产级架构
- **高性能API**: FastAPI异步框架，支持高并发
- **智能缓存**: Redis缓存常用推荐结果
- **容器化部署**: Docker + Docker Compose一键部署
- **监控告警**: 完整的系统监控和性能指标

## 🚀 快速开始

### 环境准备
```bash
cd multi_teacher_fusion
pip install -r docker/requirements.txt
```

### 运行测试
```bash
python -m pytest tests/ -v
```

### 启动API服务
```bash
python api/main.py
```

### Docker部署
```bash
cd docker
docker-compose up -d
```

## 📈 预期性能目标

- **准确性提升**: 相比最佳单Teacher提升5-10%
- **响应时间**: <100ms电影推荐响应
- **并发处理**: 支持1000+ QPS
- **系统可用性**: 99.9%服务可用性

## 🔬 实验验证

### 基准对比
- SVD单独 vs xDeepFM单独 vs AutoInt单独
- Ensemble Teacher vs LLM Teacher
- 多Teacher融合 vs 所有基准方法

### 评估指标
- **准确性**: NDCG@10, Recall@20, Precision@10
- **多样性**: 流派多样性，长尾推荐能力
- **效率**: 推理时间，内存占用
- **可解释性**: 推荐原因和来源

---

**开发状态**: 🚧 初始化完成，准备开始核心开发  
**负责人**: AI Development Team  
**预计完成**: 2024年9月20日
