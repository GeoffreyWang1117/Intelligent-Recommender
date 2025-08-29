# 🎯 教师模型优化计划
# Teacher Model Optimization Plan

**目标**: 为Fisher Information per-layer + pruning-aware蒸馏准备高质量教师模型

---

## 📋 第一阶段：增强现有Teacher模型

### 🔥 优先级1: Traditional Teachers集成优化

#### 1.1 Ensemble教师模型构建
```python
# 目标文件: teachers/traditional_teachers/ensemble_teacher.py
class EnsembleTeacher:
    """6个SOTA算法的集成教师模型"""
    def __init__(self):
        self.models = {
            'dcnv2': DCNv2Model(weight=0.25),      # 最优模型，权重最高
            'din': DINModel(weight=0.20),          # 次优模型
            'xdeepfm': xDeepFMModel(weight=0.18),  # 第三名
            'deepfm': DeepFMModel(weight=0.15),    # 基础模型
            'autoint': AutoIntModel(weight=0.12),  # 注意力模型
            'transformer4rec': T4RModel(weight=0.10)  # 序列模型
        }
    
    def get_ensemble_prediction(self, user_id, candidates):
        """加权集成预测"""
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(user_id, candidates)
            predictions[name] = pred * model.weight
        return self._weighted_combine(predictions)
```

#### 1.2 性能提升要点
- **权重优化**: 基于Phase 1结果动态调整模型权重
- **特征融合**: 不同模型的特征表示层融合
- **预测校准**: Platt Scaling/Isotonic Regression校准预测

### 🤖 优先级2: LLM Teachers增强

#### 2.1 双语协同优化
```python
# 目标文件: teachers/llm_teachers/enhanced_dual_llm.py
class EnhancedDualLLMTeacher:
    """增强版双语LLM教师"""
    def __init__(self):
        self.llama3 = LlamaTeacher()    # 英文主力
        self.qwen3 = QwenTeacher()      # 中文辅助
        self.consensus_threshold = 0.7   # 一致性阈值
    
    def get_consensus_recommendation(self, user_profile, candidates):
        """双模型一致性推荐"""
        llama_recs = self.llama3.recommend(user_profile, candidates)
        qwen_recs = self.qwen3.recommend(user_profile, candidates)
        
        # 计算交集和一致性
        consensus = self._calculate_consensus(llama_recs, qwen_recs)
        return self._merge_recommendations(llama_recs, qwen_recs, consensus)
```

#### 2.2 LLM优化要点
- **提示词工程**: 针对推荐任务的专门提示词模板
- **输出格式化**: 稳定的JSON输出格式确保
- **响应速度优化**: 批量推理 + 缓存机制

---

## 📋 第二阶段：跨数据集泛化能力

### 🎬 MovieLens优化
- **冷启动处理**: 新用户/新物品的推荐策略
- **时序建模**: 用户偏好随时间变化的建模
- **隐式反馈**: 除评分外的行为信号整合

### 🛒 Amazon Reviews优化  
- **多模态融合**: 文本评论 + 评分 + 商品特征
- **层次分类**: 商品类别的层次化建模
- **季节性模式**: 购买行为的时间模式

---

## 📋 第三阶段：为蒸馏准备的特殊优化

### 🔍 Fisher Information友好设计
```python
# 目标: 每层参数重要性可计算
class FisherFriendlyTeacher:
    """为Fisher Information计算优化的教师模型"""
    def __init__(self, base_model):
        self.base_model = base_model
        self.layer_hooks = {}  # 注册梯度钩子
        
    def register_fisher_hooks(self):
        """注册Fisher信息计算钩子"""
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                param.register_hook(self._fisher_hook(name))
    
    def compute_layer_fisher(self, dataloader):
        """计算每层Fisher信息"""
        fisher_info = {}
        for batch in dataloader:
            # 前向传播 + 反向传播
            # 累积梯度平方
        return fisher_info
```

### ✂️ Pruning-aware预处理
- **结构化剪枝**: 整个神经元/通道的剪枝
- **非结构化剪枝**: 个别权重的剪枝
- **渐进式剪枝**: 训练过程中逐步剪枝

---

## 🚀 实施时间线

### Week 1: Traditional Ensemble
- [ ] 实现加权集成Teacher
- [ ] 在MovieLens+Amazon上验证
- [ ] 性能基准建立

### Week 2: LLM Enhancement  
- [ ] 双语一致性机制
- [ ] 响应速度优化
- [ ] 输出格式标准化

### Week 3: Fisher-friendly Adaptation
- [ ] 梯度钩子注册
- [ ] Fisher信息计算模块
- [ ] Pruning预处理流程

### Week 4: Integration Testing
- [ ] 端到端测试
- [ ] 蒸馏流程对接
- [ ] 性能评估

---

## 💡 技术要点

### 代码组织
```
teachers/
├── ensemble_teachers/           # 新增：集成教师模块
│   ├── traditional_ensemble.py
│   ├── llm_ensemble.py
│   └── cross_modal_ensemble.py
├── fisher_utils/               # 新增：Fisher信息计算
│   ├── fisher_calculator.py
│   ├── gradient_hooks.py
│   └── layer_importance.py
└── pruning_prep/              # 新增：剪枝预处理
    ├── structured_pruning.py
    ├── unstructured_pruning.py
    └── progressive_pruning.py
```

### 性能目标
- **Traditional Ensemble**: 综合得分 > 0.45 (vs 当前最好0.3676)
- **LLM Teachers**: 成功率 > 95%, 响应时间 < 2秒
- **Fisher计算**: 梯度计算开销 < 20%
- **内存效率**: 同时加载所有教师模型

---

## 🎯 成功标准

1. **可靠性**: 所有教师模型100%推荐成功率
2. **性能**: Ensemble效果显著优于单模型
3. **效率**: Fisher信息计算在合理时间内完成
4. **兼容性**: 无缝对接后续蒸馏流程

这个计划确保我们有强大而可靠的教师模型，为高质量的Fisher Information + Pruning-aware蒸馏奠定基础。
