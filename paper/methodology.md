# Methodology

## 3. Methodology

### 3.1 Problem Formulation

Given a set of user-item interactions $\mathcal{D} = \{(u_i, v_j, r_{ij})\}$ where $u_i \in \mathcal{U}$ represents users, $v_j \in \mathcal{V}$ represents items, and $r_{ij}$ represents ratings, our goal is to learn a compact student model $f_S$ that can effectively predict user preferences while leveraging knowledge from multiple heterogeneous teacher models.

Let $\mathcal{T} = \{T_1, T_2, ..., T_K\}$ be a set of $K$ teacher models, where each teacher $T_k$ can be either a traditional collaborative filtering algorithm or a large language model. Our objective is to train a student model $f_S$ with parameters $\theta_S$ that minimizes:

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{task} + \beta \sum_{k=1}^K w_k \mathcal{L}_{distill}^k + \gamma \mathcal{L}_{reg}$$

where $\mathcal{L}_{task}$ is the primary task loss, $\mathcal{L}_{distill}^k$ is the distillation loss from teacher $T_k$, $w_k$ are teacher-specific weights, and $\mathcal{L}_{reg}$ is a regularization term.

### 3.2 LayerwiseAdapter Architecture

Our LayerwiseAdapter consists of three specialized layers, each designed to capture different aspects of the recommendation process:

#### 3.2.1 Embedding Adapter (Layer 1)

The Embedding Adapter focuses on learning aligned representations between different teacher models:

$$\mathbf{h}_1^{emb} = \text{EmbeddingAdapter}(\mathbf{u}, \mathbf{v})$$

where $\mathbf{u}$ and $\mathbf{v}$ are user and item embeddings respectively. This layer employs multi-head attention to capture feature interactions:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

The embedding adapter aligns features from different teachers through learned projection matrices:

$$\mathbf{h}_1^{aligned} = \sum_{k=1}^K w_k \mathbf{W}_k^{proj} \mathbf{h}_k^{teacher}$$

#### 3.2.2 Interaction Adapter (Layer 2)

The Interaction Adapter models complex user-item interactions using cross-attention mechanisms:

$$\mathbf{h}_2^{int} = \text{CrossAttention}(\mathbf{h}_1^{emb}, \mathbf{h}_1^{emb}) + \text{FFN}(\mathbf{h}_1^{emb})$$

This layer captures both explicit and implicit feature interactions through:

1. **Cross-Attention**: Models inter-feature dependencies
2. **Feed-Forward Networks**: Learns non-linear transformations
3. **Residual Connections**: Preserves information flow

#### 3.2.3 Reasoning Adapter (Layer 3)

The Reasoning Adapter performs final recommendation decisions with adaptive teacher weighting:

$$\mathbf{h}_3^{reason} = \text{ReasoningAdapter}(\mathbf{h}_2^{int})$$

$$\hat{r}_{ij} = \sigma(\mathbf{W}_{out} \mathbf{h}_3^{reason} + \mathbf{b}_{out})$$

The adaptive weighting mechanism dynamically adjusts teacher contributions based on input characteristics:

$$w_k^{adaptive} = \text{softmax}(\mathbf{W}_{gate} \mathbf{h}_2^{int})_k$$

### 3.3 Multi-Teacher Fusion Strategy

#### 3.3.1 Teacher Selection and Preparation

We select complementary teacher models representing different paradigms:

1. **Traditional Teachers**: SVD, xDeepFM, AutoInt (efficient, specialized)
2. **LLM Teachers**: Llama3 via Ollama API (semantic understanding)

Each teacher provides both predictions and intermediate representations:

$$\mathcal{O}_k = \{pred_k, \mathbf{h}_k^{emb}, \mathbf{h}_k^{int}, \mathbf{h}_k^{reason}\}$$

#### 3.3.2 Knowledge Distillation Loss

For each teacher $T_k$, we define layer-specific distillation losses:

**Prediction Distillation**:
$$\mathcal{L}_{pred}^k = \text{KL}(\sigma(z_S/\tau) || \sigma(z_k/\tau))$$

where $z_S$ and $z_k$ are student and teacher logits, $\tau$ is temperature.

**Feature Distillation**:
$$\mathcal{L}_{feat}^k = ||\mathbf{h}_S^l - \mathbf{W}_k^l \mathbf{h}_k^l||_2^2$$

**Attention Distillation**:
$$\mathcal{L}_{att}^k = ||\mathbf{A}_S - \mathbf{A}_k||_F^2$$

where $\mathbf{A}_S$ and $\mathbf{A}_k$ are attention matrices.

### 3.4 Fisher Information-Guided Optimization

#### 3.4.1 Fisher Information Computation

We compute the Fisher Information Matrix to identify critical parameters:

$$\mathbf{F}_{ii} = \mathbb{E}\left[\left(\frac{\partial \log p(y|\mathbf{x}, \theta)}{\partial \theta_i}\right)^2\right]$$

For each layer $l$ and teacher $k$, we compute:

$$\mathbf{F}^{l,k} = \mathbb{E}_{\mathcal{D}}\left[\nabla_{\theta^l} \log p(T_k(\mathbf{x})) \nabla_{\theta^l} \log p(T_k(\mathbf{x}))^T\right]$$

#### 3.4.2 Fisher-Guided Distillation

We weight the distillation loss based on Fisher importance:

$$\mathcal{L}_{distill}^k = \sum_i F_{ii}^k \cdot \mathcal{L}_i^k$$

where $F_{ii}^k$ represents the Fisher importance of parameter $i$ for teacher $k$.

#### 3.4.3 Fisher-Guided Pruning

For model compression, we use Fisher information to guide structured pruning:

$$\text{Importance}(\theta_i) = F_{ii} \cdot |\theta_i|^2$$

Parameters with low Fisher importance are candidates for pruning.

### 3.5 Training Procedure

#### 3.5.1 Progressive Training Strategy

We employ a three-stage training procedure:

**Stage 1: Individual Layer Pre-training**
- Train each adapter layer independently
- Focus on teacher-specific knowledge transfer

**Stage 2: End-to-End Fine-tuning**
- Joint optimization of all layers
- Balance task loss and distillation losses

**Stage 3: Fisher-Guided Compression**
- Apply Fisher-guided pruning
- Fine-tune compressed model

#### 3.5.2 Dynamic Weight Scheduling

Teacher weights are dynamically adjusted during training:

$$w_k(t) = w_k^{init} \cdot \exp(-\lambda_k \cdot t)$$

where $t$ is the training step and $\lambda_k$ controls the decay rate for teacher $k$.

### 3.6 Inference Optimization

#### 3.6.1 Model Quantization

We apply post-training quantization to reduce model size:
- 16-bit precision for activations
- 8-bit precision for weights
- Dynamic quantization for attention layers

#### 3.6.2 Architectural Optimizations

- **Attention Pruning**: Remove low-importance attention heads
- **Layer Fusion**: Combine consecutive linear layers
- **Knowledge Distillation Caching**: Cache teacher outputs for common patterns

### 3.7 Computational Complexity

The computational complexity of LayerwiseAdapter is:

$$\mathcal{O}(d^2 \cdot h + d \cdot n \cdot h + d \cdot k)$$

where $d$ is embedding dimension, $h$ is hidden dimension, $n$ is sequence length, and $k$ is number of items.

This represents a significant reduction compared to LLM teachers which have complexity $\mathcal{O}(L \cdot d^2 \cdot n^2)$ where $L$ is the number of layers and $n$ is sequence length.

### 3.8 Theoretical Guarantees

We provide theoretical analysis showing that under certain conditions, LayerwiseAdapter can achieve:

1. **Approximation Bound**: $|\mathcal{R}(f_S) - \mathcal{R}(f_T)| \leq \epsilon$ with high probability
2. **Generalization Bound**: Based on Rademacher complexity analysis
3. **Compression Ratio**: Theoretical lower bounds on achievable compression

The detailed proofs are provided in the supplementary material.
