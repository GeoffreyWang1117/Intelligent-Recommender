
# LLM Teachers Experiment Results - Visual Analysis Summary

## 📊 Generated Visualizations

### 1. **LLM Recommendation Performance** (`llm_recommendation_performance.png`)
- **Success Rate Comparison**: Llama3 vs Qwen3 recommendation success rates
- **Response Time Analysis**: Average inference time per recommendation
- **Recommendation Distribution**: Number of recommendations per user
- **Dataset Overview**: MovieLens dataset statistics

### 2. **Fisher Information Analysis** (`llm_fisher_analysis.png`)
- **Information Content**: Fisher information content across different embedding components
- **Condition Numbers**: Matrix stability analysis on logarithmic scale
- **Eigenvalue Statistics**: Mean and standard deviation of Fisher matrix eigenvalues
- **Matrix Traces**: Overall magnitude of Fisher information matrices

### 3. **PAKD Experiment Results** (`llm_pakd_experiment.png`)
- **Loss Components**: Distillation, alignment, and total loss breakdown
- **Transfer Efficiency**: Correlation and information retention metrics
- **Teacher vs Student**: Output statistics comparison
- **Feature Importance**: Top features after pruning analysis

### 4. **Comprehensive Comparison** (`llm_comprehensive_analysis.png`)
- **Performance Radar**: Multi-dimensional model comparison
- **Experiment Configuration**: Key experimental parameters
- **Time Distribution**: Computational time allocation
- **Data Sparsity**: Dataset density visualization

## 🔍 Key Findings

### Performance Insights

- **Llama3 Success Rate**: 90.0% (27/30 users)
- **Qwen3 Success Rate**: 100.0% (30/30 users)
- **Fisher Information**: Recommendation embeddings show highest information content
- **PAKD Efficiency**: 0.996 information retention rate

### Technical Insights
- **Dataset Sparsity**: 98.30% sparse (typical for recommendation systems)
- **Experiment Duration**: 49.6 minutes total runtime
- **Feature Pruning**: Successful identification of key features through PAKD

---
*Generated on: 2025-08-29 15:35:59*
*Environment: SysDesign-Recommender (Conda)*
