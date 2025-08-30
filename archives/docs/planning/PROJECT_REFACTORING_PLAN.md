# 🏗️ 项目重构计划 - 在线推理系统整理

**日期**: 2025-08-29  
**目标**: 全面整理项目结构，明确组件关系，统一相同目标的工作  

## 📋 当前状态分析

### 1. Ensemble模型现状
**真实组成**: 6个子模型
- ✅ SVD (已有Fisher分析)
- ✅ DeepFM 
- ✅ xDeepFM (已有Fisher分析)
- ✅ AutoInt (已有Fisher分析)
- ✅ DIN
- ✅ DCNv2 (🥇 最佳性能: 0.3676)

**Fisher分析缺失**:
- ❌ DeepFM Fisher分析
- ❌ DIN Fisher分析  
- ❌ DCNv2 Fisher分析 (最重要!)

### 2. 目录结构问题
```
❌ 分散的相似功能:
- models/ (根目录算法)
- teachers/traditional_teachers/models/ (重复算法)
- layerwise_adapter/models/ (新架构)

❌ 分析结果散布:
- analysis_results/ (根目录分析)
- layerwise_adapter/architecture_analysis/ (架构分析)
- evaluation_results/ (性能评估)

❌ 实验脚本重复:
- 多个相似的teacher实验文件
- 重复的评估脚本
```

## 🎯 重构目标

### 目标1: 统一模型管理
```
📁 models/  (统一模型库)
├── algorithms/          # 核心算法实现
│   ├── svd.py
│   ├── deepfm.py  
│   ├── xdeepfm.py
│   ├── autoint.py
│   ├── din.py
│   └── dcnv2.py
├── ensemble/           # 集成模型
│   ├── traditional_ensemble.py
│   └── optimized_ensemble.py
├── llm_teachers/       # LLM教师模型
│   ├── llama3_teacher.py
│   └── real_llm_teacher.py
└── layerwise_adapter/  # 新架构学生模型
    ├── adapters.py
    └── distillation.py
```

### 目标2: 集中分析体系
```
📁 analysis/  (统一分析中心)
├── fisher_analysis/    # Fisher信息分析
│   ├── ensemble_fisher/
│   ├── llm_fisher/
│   └── student_fisher/
├── pakd_experiments/   # PAKD实验
│   ├── traditional_pakd/
│   └── multi_teacher_pakd/
├── performance/        # 性能评估
│   ├── benchmarks/
│   └── comparative_analysis/
└── reports/           # 分析报告
    ├── ensemble_analysis.md
    ├── llm_analysis.md
    └── fusion_analysis.md
```

### 目标3: 清理实验代码
```
📁 experiments/  (统一实验管理)
├── training/          # 模型训练
│   ├── traditional_training.py
│   ├── llm_training.py
│   └── student_training.py
├── evaluation/        # 模型评估
│   ├── benchmark_evaluation.py
│   └── comparative_evaluation.py
├── distillation/      # 知识蒸馏
│   ├── single_teacher_distillation.py
│   └── multi_teacher_distillation.py
└── archived/         # 历史实验
    └── deprecated_experiments/
```

## 📊 具体重构步骤

### 阶段1: 模型库整理 (立即执行)
1. 创建统一的models/algorithms/目录
2. 移动重复的算法实现到统一位置
3. 清理teachers/traditional_teachers/models/重复代码
4. 更新所有import路径

### 阶段2: 分析结果整合 (立即执行)  
1. 创建analysis/目录结构
2. 移动analysis_results/到analysis/
3. 整合layerwise_adapter/architecture_analysis/
4. 统一evaluation_results/到analysis/performance/

### 阶段3: 补全Fisher分析 (下一步工作)
1. 为DeepFM, DIN, DCNv2进行Fisher分析
2. 完成6模型完整Fisher对比
3. 更新ensemble_fisher_analysis.md

### 阶段4: README更新 (完成后)
1. 更新英文README.md
2. 更新中文README_CN.md  
3. 详细说明当前成果和不足

## 🔄 执行计划

**立即执行**: 目录重构 + 代码整理
**1天内完成**: Fisher分析补全
**2天内完成**: README全面更新
**最终目标**: 清晰的项目结构 + 完整的分析体系
