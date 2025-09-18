# Activation Steering

## Overview
Activation steering is a mechanistic interpretability technique that modifies model behavior by adding concept vectors to internal activations during inference. This allows for precise, targeted control of model outputs without retraining.

## Who Proposed It
- **Turner et al. (2023)** - "Activation Addition: Steering Language Models Without Optimization" (ActAdd baseline method) - arXiv:2308.10248
- **Zou et al. (2023)** - "Representation Engineering: A Top-Down Approach to AI Transparency" (RepE framework) - arXiv:2310.01405
- **Li et al. (2023)** - "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" (ITI for truthfulness) - arXiv:2306.03341
- **Subramani et al. (2022)** - "Extracting Latent Steering Vectors from Pretrained Language Models" (Early foundational work) - arXiv:2205.05124

## What It Does
Activation steering works by:
1. **Computing concept vectors**: Extract vectors representing desired behaviors or attributes via contrast pairs (positive/negative examples) (Subramani et al., 2022)
2. **Adding vectors to activations**: Apply these vectors to model activations at specific layers during inference (Turner et al., 2023; ActAdd)
3. **Steering model outputs**: Control outputs toward or away from particular concepts, e.g., via scalar multiples for strength adjustment

The technique can control various aspects of model behavior including:
- Sentiment and emotional tone
- Truthfulness and honesty
- Helpfulness vs. harmfulness
- Specific knowledge domains or topics

## Relevant Deductions
- **Linear Representation Hypothesis**: Many concepts are represented as linear directions in activation space (Zou et al., 2023; RepE)
- **Activation Additivity**: Simple vector addition can effectively modify complex behaviors (Turner et al., 2023; core ActAdd finding)
- **Layer-Specific Effects**: Different layers encode different types of information - early layers: syntax/grammar, later layers: semantics/concepts (Li et al., 2023; ITI layer ablations)
- **Compositional Control**: Multiple concept vectors can be combined for nuanced steering (Zou et al., 2023; multi-vector combinations)

## Potential Extensions
1. **Probe Different Layers**: Test steering effectiveness across all model layers to understand the hierarchy of representations
2. **Analyze Vector Geometry**: Study the geometric relationships between concept vectors (Subramani et al., 2022; geometric probes)
3. **Cross-Model Generalization**: Test whether concept vectors transfer between different model architectures (Zou et al., 2023; multi-LLM tests)
4. **Ablation Studies**: Remove or modify components to understand which parts are crucial
5. **Intervention Cascades**: Study how steering one concept affects others (Subramani et al., 2022; interaction analysis)

## Key Papers and Citations
- Turner, A.M., Thiergart, L., Udell, D., Leech, G., Mini, U., et al. (2023). "Activation Addition: Steering Language Models Without Optimization" arXiv:2308.10248. DOI:10.48550/arXiv.2308.10248
- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Hasan, M.I., et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency" arXiv:2310.01405. DOI:10.48550/arXiv.2310.01405
- Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" arXiv:2306.03341. DOI:10.48550/arXiv.2306.03341
- Subramani, N., Suresh, N., & Peters, M.E. (2022). "Extracting Latent Steering Vectors from Pretrained Language Models" arXiv:2205.05124. DOI:10.48550/arXiv.2205.05124

## Example Use Cases

### Safety Applications
- Reducing harmful or toxic outputs (Turner et al., 2023; sentiment steering)
- Increasing truthfulness and reducing hallucinations (Li et al., 2023; ITI method)
- Steering away from biased responses (Zou et al., 2023; bias mitigation via RepE)

### Research Applications
- Testing model capabilities and limitations
- Understanding internal representations
- Probing knowledge organization

### Practical Applications
- Customizing chatbot personality
- Domain-specific fine-tuning without retraining
- A/B testing different model behaviors

## Implementation Notes
- **Compute vectors**: Via contrast pairs (positive/negative examples) or probing methods
- **Model size**: Works best with larger models (>1B parameters)
- **Layer selection**: Effectiveness varies by layer and concept; mid-to-late layers typically most effective
- **Validation required**: Careful testing needed to avoid unintended side effects
- **Integration**: Can be combined with other techniques (e.g., SAE-targeted steering for 2025 advances)
- **Strength control**: Adjust via scalar multiples for fine-grained control