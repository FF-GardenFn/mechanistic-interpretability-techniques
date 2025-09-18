# Logit Lens

## Overview

The logit lens is a mechanistic interpretability technique that projects intermediate representations from any layer of a transformer through the final unembedding matrix to reveal what tokens the model is "thinking about" at each layer. This provides insights into how the model's understanding evolves throughout the forward pass.

## Who Proposed It

The logit lens was:
- **Initially proposed by**: nostalgebraist (2020) in "Interpreting GPT: The Logit Lens" - LessWrong post
- **Formalized by**: Geva et al. (2021) in "Transformer Feed-Forward Layers Are Key-Value Memories" (applied to FFN analysis)
- **Extended by**: Belrose et al. (2023) with the Tuned Lens (layer-specific transformations)
- **Further developed by**: Elhage et al. (2021) at Anthropic for circuit analysis

## What It Does

The logit lens works by:
1. Taking intermediate activations from any layer in the transformer
2. Applying the final layer normalization (if present)
3. Projecting through the unembedding matrix (W_U) to get logits
4. Converting logits to probability distributions over the vocabulary
5. Examining which tokens have the highest probability at each layer (e.g., top-k tokens for focused analysis)

The key insight is that if the residual stream maintains consistent semantics throughout the network, intermediate representations should be interpretable when projected through the output space.

## Relevant Deductions

The logit lens reveals:
- **Token Predictions at Each Layer**: What tokens the model is considering at intermediate stages (nostalgebraist, 2020)
- **Information Processing Flow**: How the model's "thoughts" evolve from layer to layer (Elhage et al., 2021)
- **Convergence Patterns**: Whether and when the model settles on its final prediction (Belrose et al., 2023)
- **Attention vs. MLP Contributions**: Different roles of attention and feed-forward components (Geva et al., 2021; Elhage et al., 2021)
- **Semantic Development**: How semantic understanding builds up progressively (Dar et al., 2022)

Common patterns observed (as demonstrated by nostalgebraist, 2020):
- Early layers often predict frequent/generic tokens
- Middle layers develop more specific semantic predictions
- Later layers fine-tune and finalize the prediction
- Some layers may temporarily consider incorrect tokens before correction

##  Potential Extensions

To gain deeper insights:
1. **Component-wise Analysis**: Apply logit lens after attention vs. MLP sublayers separately (Elhage et al., 2021)
2. **Token Position Analysis**: Track how predictions vary across sequence positions (Dar et al., 2022)
3. **Comparative Studies**: Compare patterns across different model sizes and architectures
4. **Intervention Experiments**: Modify intermediate activations and observe changes in logit lens outputs (Meng et al., 2022)
5. **Attention Pattern Correlation**: Correlate logit lens predictions with attention patterns (Wang et al., 2023)
6. **Error Analysis**: Analyze where and why the model changes its predictions between layers (Wang et al., 2023; logit diff analysis)

## Key Papers and Citations

### Foundational Papers
- nostalgebraist (2020). "Interpreting GPT: The Logit Lens" *LessWrong*. URL: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). "Transformer Feed-Forward Layers Are Key-Value Memories" *Proceedings of EMNLP*. arXiv:2012.14913. DOI:10.18653/v1/2021.emnlp-main.446

### Related Work
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits" *Anthropic*. URL: https://transformer-circuits.pub/2021/framework/index.html
- Dar, G., Geva, M., Gupta, A., et al. (2022). "Analyzing Transformer Dynamics as Movement Through Embedding Space" arXiv:2206.02654. DOI:10.48550/arXiv.2206.02654
- Belrose, N., Furman, Z., et al. (2023). "Eliciting Latent Predictions from Transformers with the Tuned Lens" arXiv:2303.08112. DOI:10.48550/arXiv.2303.08112

### Applications and Extensions
- Meng, K., Bau, D., et al. (2022). "Locating and Editing Factual Associations in GPT" *Advances in Neural Information Processing Systems*. arXiv:2202.05262. DOI:10.48550/arXiv.2202.05262
- Wang, K., Variengien, A., et al. (2023). "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small" *ICLR*. arXiv:2211.00593. DOI:10.48550/arXiv.2211.00593

## Example Use Cases

### Language Modeling Analysis
- **Next Token Prediction**: Observe how confidence in the correct next token builds up
- **Uncertainty Tracking**: Identify layers where the model is most uncertain
- **Error Correction**: See how models recover from early incorrect predictions

### Factual Knowledge Probing
- **Fact Recall**: Track when factual information becomes accessible (e.g., "The capital of France is ___") (Meng et al., 2022)
- **Knowledge Integration**: Observe how multiple pieces of information combine
- **Contradiction Detection**: See how models handle conflicting information

### Syntax and Semantics
- **Grammatical Structure**: Track when syntactic constraints influence predictions
- **Semantic Coherence**: Observe semantic consistency across layers
- **Context Integration**: See how context modifies predictions over layers

### Model Comparison
- **Architecture Differences**: Compare how different architectures process information
- **Scale Effects**: Understand how model size affects information processing
- **Training Effects**: Analyze how different training procedures affect internal processing

### Debugging and Analysis
- **Failure Mode Analysis**: Understand where and why models make mistakes (Wang et al., 2023)
- **Bias Detection**: Identify layers where problematic biases emerge
- **Robustness Testing**: Analyze how perturbations affect internal processing

## Technical Considerations

### Limitations
- **Assumes Linear Readout**: The technique assumes information is linearly accessible (critiqued in Belrose et al., 2023; tuned lens mitigates)
- **Final Layer Bias**: Projects through final layer parameters, which may not be optimal for intermediate layers (Belrose et al., 2023)
- **Layer Norm Effects**: Layer normalization can significantly affect results
- **Vocabulary Bias**: Results may be biased toward frequent tokens (Geva et al., 2021; memorized tokens)

### Best Practices
- Always apply appropriate layer normalization before unembedding
- Consider both raw logits and probability distributions
- Analyze multiple examples to identify consistent patterns
- Compare with other interpretability techniques for validation

## Getting Started

See `technique.py` for a practical implementation that demonstrates how to apply the logit lens to analyze transformer models and interpret the results.