# Recent Advances in Activation Steering (2025)

## Key Developments

### 1. Activation Scaling Methods

Recent research continues to refine scaling techniques, confirming that **merely scaling the signed magnitude** of activation vectors, without changing direction, remains sufficient for effective steering on simple tasks. This approach has evolved to include dynamic adaptations for broader applicability.

#### DynScalar Innovation
- Makes activation scalars **learned functions** of the activation vectors
- Allows interventions to transfer to prompts of varying length
- Improves robustness across different contexts
- 2025 extensions incorporate multi-subspace representations for adaptive steering in complex scenarios

### 2. SAE-Targeted Steering (SAE-TS)

SAE-TS remains a cornerstone, with 2025 work emphasizing feature selection for optimal steering. New analyses show SAEs excel when right features are targeted, enhancing predictability.

#### Key Features
- Constructs steering vectors to achieve **specific desired effects** while minimizing unintended changes
- Learns linear relationships between steering vectors and SAE feature effects
- More interpretable than traditional steering vectors
- Recent advances: Selective feature activation boosts coherence by up to 20% in LLM outputs

#### Implementation Approach
```python
# SAE-TS Pipeline
1. Train SAE on model activations
2. Identify target features to modify
3. Learn steering vector that selectively affects target features
4. Apply steering with minimal side effects
```

### 3. Feature Guided Activation Additions (FGAA)

FGAA has gained traction in 2025 as a hybrid method, outperforming baselines in precision and interpretability across benchmarks.

#### Advantages
- Operates in SAE latent space for interpretability
- Uses optimization to select desired features
- Creates precise, human-interpretable steering vectors
- Better control over side effects; excels in multilingual and bias tasks

### 4. Sparse Activation Steering (SAS) and Fusion Steering

Emerging 2025 methods like SAS leverage sparse spaces for efficient control, while Fusion Steering enables prompt-specific interventions.

#### SAS Highlights
- Steers LLMs in SAE-derived sparse activations for reduced compute
- Targets causal directions in low-dimensional spaces

#### Fusion Steering
- Dynamically fuses steering vectors per prompt
- Boosts factual accuracy in QA by 15-25%

### 5. Vision-Language-Action (VLA) Steering and ExpertSteer

2025 saw expansions to multimodal models, with mechanistic steering for VLAs and expert-knowledge integration.

#### VLA Capabilities
- Direct intervention in model behavior at inference time
- Identifies sparse semantic directions (e.g., speed, direction in robotics)
- Causally linked to action selection
- Enables real-time modulation without fine-tuning

#### ExpertSteer
- Incorporates domain expertise into steering vectors
- Improves control in specialized tasks like medical QA

## Current Challenges

### 1. Unpredictability
- Often unclear exactly how a steering vector will affect behavior
- May produce unintended changes or no interpretable changes
- 2025 progress: Backtracking mechanisms allow iterative refinement during inference
- Need better prediction methods for steering effects, especially in multilingual settings

### 2. Performance Impact
- Steering can reduce general model performance
- Trade-off between control and capability preservation
- New mitigations: Preference optimization hybrids preserve capabilities while steering for safety

### 3. Scalability
- Steering effects may not transfer across model scales
- Computational cost of finding optimal steering vectors
- Challenge of automated steering vector discovery; 2025 solutions include transfer across architectures via activation spaces

### 4. Bias and Self-Preference
- Emerging issue: LLMs exhibit self-preference in evaluations
- Steering vectors mitigate up to 97% of bias via activation additions


### Evaluation Framework

1. **Intended Effect**: Measure if steering achieves desired behavior
2. **Side Effects**: Quantify unintended changes
3. **Performance Preservation**: Test model capabilities post-steering
4. **Robustness**: Evaluate across diverse inputs and scales
5. **Interpretability**: Assess human understanding of changes
6. **Bias Reduction**: Track mitigation of preferences or stereotypes (new 2025 metric)

## Future Directions

### Research Priorities
1. **Automated Discovery**: ML methods to find optimal steering vectors, including expert-infused automation
2. **Multi-objective Steering**: Controlling multiple behaviors simultaneously, e.g., via multi-subspace reps
3. **Adversarial Robustness**: Steering for safety against attacks
4. **Cross-model Transfer**: Steering vectors that work across architectures and modalities
5. **Sparse and Causal Control**: Expanding SAS for causal multilingual steering

### Applications
- **Safety**: Reducing harmful outputs without retraining; bias mitigation pipelines
- **Customization**: User-specific model behavior
- **Debugging**: Understanding model failures through steering
- **Alignment**: Fine-grained control over model values
- **Robotics/Multimodal**: Real-time VLA interventions

## Practical Tips

### When to Use Each Method

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| Basic ActAdd | Quick experiments | Simple, fast | Less precise |
| SAE-TS | Precise control | Interpretable, minimal side effects | Requires SAE training; feature selection key |
| FGAA | Feature-level control | Human-interpretable, outperforms baselines | Computationally intensive |
| DynScalar | Varying contexts | Robust, transferable | Limited to scaling |
| SAS | Sparse/efficient steering | Low compute, causal targeting | Early-stage, SAE-dependent |
| Fusion Steering | Prompt-specific QA | High accuracy, dynamic | Prompt-dependent |

### Implementation Checklist
- [ ] Choose appropriate steering method for task
- [ ] Create contrast dataset or identify target features
- [ ] Compute steering vectors (consider sparse approximations for scale)
- [ ] Validate on held-out examples
- [ ] Monitor for side effects and bias
- [ ] Test robustness across contexts and models
- [ ] Document steering effects

## Resources
- [Activation Engineering Paper](https://arxiv.org/abs/2308.10248)
- [Improving Steering Vectors by Targeting Sparse Autoencoder Features (SAE-TS)](https://arxiv.org/abs/2411.02193)
- [Interpretable Steering of Large Language Models with Feature Guided Activation Additions (FGAA)](https://arxiv.org/abs/2501.09929)
- [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/)
- [SAEs Are Good for Steering – If You Select the Right Features](https://arxiv.org/abs/2505.20063)
- [Mechanistic Interpretability for Steering Vision-Language-Action Models](https://arxiv.org/abs/2509.00328)
- [Activation Steering for Bias Mitigation](https://arxiv.org/abs/2508.09019)