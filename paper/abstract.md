# Abstract

## Layerwise Knowledge Distillation for Efficient Recommender Systems: A Multi-Teacher Fusion Approach

Knowledge distillation has emerged as a powerful technique for compressing large neural networks while preserving their performance. In the domain of recommender systems, this approach is particularly valuable given the computational constraints of real-time applications. We present LayerwiseAdapter, a novel three-layer progressive knowledge distillation architecture that effectively combines knowledge from both traditional collaborative filtering algorithms and large language models (LLMs). 

Our approach introduces a hierarchical distillation strategy with three specialized adapters: (1) an Embedding Adapter for feature alignment, (2) an Interaction Adapter for user-item interaction modeling, and (3) a Reasoning Adapter for recommendation decision making. Through extensive experiments on MovieLens datasets, we demonstrate that our LayerwiseAdapter achieves a 9,215× speedup compared to LLM teachers while maintaining a 2.09% improvement in RMSE (from 1.226 to 1.201). 

The model achieves production-ready performance with only 686K parameters (4.2MB), 0.000137s inference time, and superior accuracy compared to traditional ensemble methods. Our Fisher Information analysis reveals optimal pruning strategies that further enhance efficiency without significant performance degradation. The approach represents a significant advancement in deploying intelligent recommendation systems at scale, bridging the gap between the semantic understanding of LLMs and the efficiency requirements of real-world applications.

**Keywords:** Knowledge Distillation, Recommender Systems, Large Language Models, Multi-Teacher Learning, Fisher Information Analysis
