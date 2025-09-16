# Related Work

## 2. Related Work

### 2.1 Knowledge Distillation in Neural Networks

Knowledge distillation, first introduced by Hinton et al. [1], has become a fundamental technique for model compression and acceleration. The core idea involves training a smaller "student" network to mimic the behavior of a larger, more complex "teacher" network. Traditional distillation approaches focus on matching the output distributions or intermediate feature representations.

Recent advances in knowledge distillation include attention transfer [2], feature-based distillation [3], and relational knowledge distillation [4]. However, most existing work focuses on computer vision or natural language processing tasks, with limited exploration in recommender systems.

### 2.2 Multi-Teacher Knowledge Distillation

Multi-teacher distillation extends the traditional single-teacher paradigm by leveraging knowledge from multiple teacher models. Zhang et al. [5] proposed ensemble distillation for image classification, while You et al. [6] introduced gate-based multi-teacher approaches. However, these methods primarily focus on homogeneous teachers with similar architectures.

The challenge of heterogeneous multi-teacher distillation—combining fundamentally different model types—remains largely unexplored, particularly in the context of recommender systems where traditional collaborative filtering and modern LLMs represent vastly different paradigms.

### 2.3 Recommender Systems

#### 2.3.1 Traditional Approaches
Classical collaborative filtering methods include memory-based approaches (user-based and item-based CF) and model-based methods such as matrix factorization [7]. Singular Value Decomposition (SVD) and its variants have been widely successful for explicit feedback scenarios.

#### 2.3.2 Deep Learning Methods
The advent of deep learning revolutionized recommender systems. Neural Collaborative Filtering [8] introduced neural networks to collaborative filtering. Subsequently, more sophisticated architectures emerged:

- **DeepFM** [9]: Combines factorization machines with deep neural networks
- **xDeepFM** [10]: Introduces compressed interaction networks for feature crossing
- **AutoInt** [11]: Employs self-attention for automatic feature interaction
- **DIN** [12]: Uses attention mechanisms for sequential recommendation
- **DCNv2** [13]: Improves deep cross networks for feature crossing

#### 2.3.3 Large Language Models for Recommendation
Recent work has explored LLMs for recommendation tasks. InstructRec [14] uses instruction tuning for recommendation. P5 [15] formulates recommendation as a text-to-text generation problem. However, the computational cost of LLMs makes them impractical for real-time scenarios.

### 2.4 Efficiency in Recommender Systems

Efficiency in recommender systems has been addressed through various approaches:

- **Model Pruning**: Removing redundant parameters to reduce model size [16]
- **Quantization**: Reducing precision of weights and activations [17]
- **Matrix Factorization Variants**: Developing more efficient factorization techniques [18]
- **Hashing Techniques**: Using locality-sensitive hashing for efficient similarity computation [19]

### 2.5 Fisher Information in Neural Networks

Fisher Information has been used in neural networks for various purposes:

- **Catastrophic Forgetting**: Elastic Weight Consolidation (EWC) uses Fisher Information to identify important parameters [20]
- **Neural Network Pruning**: Fisher-based pruning identifies critical weights [21]
- **Knowledge Distillation**: Fisher Information guides the distillation process [22]

However, the application of Fisher Information to multi-teacher distillation in recommender systems remains unexplored.

### 2.6 Gap Analysis

Despite significant progress in both knowledge distillation and recommender systems, several gaps remain:

1. **Heterogeneous Multi-Teacher Distillation**: No existing work effectively combines traditional CF methods with LLMs in a unified distillation framework.

2. **Architecture-Aware Distillation**: Most distillation methods ignore the architectural differences between teacher and student models, particularly important in recommender systems.

3. **Production-Oriented Design**: Academic solutions often ignore real-world constraints such as latency, memory usage, and deployment complexity.

4. **Comprehensive Evaluation**: Many studies lack thorough evaluation against multiple SOTA baselines and fail to consider multiple evaluation metrics.

Our LayerwiseAdapter framework addresses these gaps by providing a principled approach to multi-teacher distillation specifically designed for production-ready recommender systems.

---

**References:**
[1] Hinton, G., et al. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).
[2] Zagoruyko, S., et al. "Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer." ICLR 2017.
[3] Romero, A., et al. "Fitnets: Hints for thin deep nets." ICLR 2015.
[4] Park, W., et al. "Relational knowledge distillation." CVPR 2019.
[5] Zhang, D., et al. "Deep mutual learning." CVPR 2018.
[6] You, S., et al. "Learning from multiple teacher networks." KDD 2017.
[7] Koren, Y., et al. "Matrix factorization techniques for recommender systems." Computer 2009.
[8] He, X., et al. "Neural collaborative filtering." WWW 2017.
[9] Guo, H., et al. "DeepFM: a factorization-machine based neural network for CTR prediction." IJCAI 2017.
[10] Lian, J., et al. "xDeepFM: Combining explicit and implicit feature interactions for recommender systems." KDD 2018.
[11] Song, W., et al. "AutoInt: Automatic feature interaction learning via self-attentive neural networks." CIKM 2019.
[12] Zhou, G., et al. "Deep interest network for click-through rate prediction." KDD 2018.
[13] Wang, R., et al. "DCN V2: Improved deep & cross network for feature cross learning in web-scale LTR systems." WWW 2021.
[14] Zhang, J., et al. "InstructRec: Instruction tuning for large language models in recommender systems." arXiv preprint (2023).
[15] Geng, S., et al. "Recommendation as language processing (RLP): A unified pretrain, personalize, and predict paradigm (P5)." RecSys 2022.
[16] Han, S., et al. "Learning both weights and connections for efficient neural network." NIPS 2015.
[17] Jacob, B., et al. "Quantization and training of neural networks for efficient integer-arithmetic-only inference." CVPR 2018.
[18] Rendle, S., et al. "BPR: Bayesian personalized ranking from implicit feedback." UAI 2009.
[19] Shrivastava, A., et al. "Asymmetric LSH (ALSH) for sublinear time maximum inner product search (MIPS)." NIPS 2014.
[20] Kirkpatrick, J., et al. "Overcoming catastrophic forgetting in neural networks." PNAS 2017.
[21] Theis, L., et al. "Faster gaze prediction with dense networks and Fisher pruning." arXiv preprint (2018).
[22] Srinivas, S., et al. "Knowledge transfer with Jacobian matching." ICML 2018.
