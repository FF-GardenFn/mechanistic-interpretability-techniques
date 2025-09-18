# Linear Probing

## Overview

Linear probing is a fundamental mechanistic interpretability technique that trains linear classifiers on internal neural network representations to detect what information is linearly accessible at different layers. This method provides insights into what linguistic, semantic, and factual knowledge is encoded in the model's internal representations.

## Who Proposed It

Linear probing was pioneered by:
- **Belinkov & Glass (2019)** - "Analysis Methods in Neural Language Processing: A Survey" (Comprehensive survey) - arXiv:1812.08951
- **Tenney et al. (2019)** - "BERT Rediscovers the Classical NLP Pipeline" (Seminal application to BERT) - arXiv:1905.05950
- **Tenney et al. (2019)** - "What do you learn from context? Probing for sentence structure in contextualized word representations" (Edge-probing precursor) - arXiv:1905.06316

## What It Does

Linear probing works by:
1. Extracting internal representations from specific layers of a neural network
2. Training simple linear classifiers (logistic regression, linear SVM) on these representations (Belinkov & Glass, 2019)
3. Evaluating how well the classifier can predict target labels (e.g., part-of-speech tags, syntactic roles) (Tenney et al., 2019)
4. Measuring the linear separability of different types of information across layers

## Relevant Deductions

Linear probing reveals:
- **Information Accessibility**: What information is linearly accessible without complex transformations (Belinkov & Glass, 2019)
- **Layer-wise Information Flow**: How different types of information emerge and evolve across layers (Tenney et al., 2019)
- **Hierarchical Processing**: Lower layers often encode syntax, higher layers encode semantics - as shown by Tenney et al. (2019) in "BERT Rediscovers the Classical NLP Pipeline"
- **Information Localization**: Which layers contain peak performance for specific tasks (Tenney et al., 2019)

##  Potential Extensions

To gain deeper insights:
1. **Multi-task Probing**: Probe for multiple linguistic phenomena simultaneously to understand information organization (Voita & Titov, 2020; MDL approach)
2. **Cross-layer Analysis**: Track how the same information changes representation across layers (Hewitt & Manning, 2019; structural probes)
3. **Comparative Studies**: Compare probing results across different model architectures and sizes (Rogers et al., 2016)
4. **Causal Analysis**: Combine with intervention methods to understand if detected information is actually used (inspired by Voita & Titov, 2020)
5. **Feature Interaction**: Investigate how different linguistic features interact in the representation space (Hewitt & Manning, 2019)

## Key Papers and Citations

### Foundational Papers
- Belinkov, Y., & Glass, J. (2019). "Analysis Methods in Neural Language Processing: A Survey" *Transactions of the Association for Computational Linguistics*, 7, 49-72. arXiv:1812.08951. DOI:10.1162/tacl_a_00254
- Tenney, I., Das, D., & Pavlick, E. (2019). "BERT Rediscovers the Classical NLP Pipeline" *Proceedings of ACL*. arXiv:1905.05950. DOI:10.18653/v1/P19-1452
- Tenney, I., Xia, P., Chen, B., et al. (2019). "What do you learn from context? Probing for sentence structure in contextualized word representations" arXiv:1905.06316

### Important Follow-ups
- Rogers, A., Kovaleva, O., & Rumshisky, A. (2016). "A Primer on Neural Network Models for Natural Language Processing" *Journal of Artificial Intelligence Research*, 57, 615-686. arXiv:1510.00726
- Hewitt, J., & Manning, C. D. (2019). "A Structural Probe for Finding Syntax in Word Representations" *Proceedings of NAACL*. DOI:10.18653/v1/N19-1419
- Voita, E., & Titov, I. (2020). "Information-Theoretic Probing with Minimum Description Length" *Proceedings of EMNLP*. arXiv:2003.12298. DOI:10.18653/v1/2020.emnlp-main.14

## Example Use Cases

### Syntactic Analysis
- **Part-of-speech tagging**: Probe for grammatical categories (Tenney et al., 2019)
- **Dependency parsing**: Detect syntactic relationships between words (Hewitt & Manning, 2019)
- **Constituency parsing**: Identify phrase structure information

### Semantic Analysis
- **Word sense disambiguation**: Detect which meaning of polysemous words is encoded
- **Semantic role labeling**: Identify argument structure of predicates
- **Named entity recognition**: Detect entity types and boundaries

### Factual Knowledge
- **Relation extraction**: Probe for factual relationships (e.g., "Paris is the capital of France") (Tenney et al., 2019)
- **World knowledge**: Test for encoded factual information about entities (extends to later work like Dai et al., 2022)
- **Temporal reasoning**: Detect understanding of temporal relationships

### Cross-linguistic Studies
- **Language identification**: Detect which language is being processed in multilingual models
- **Transfer learning**: Understand how knowledge transfers across languages
- **Typological features**: Probe for linguistic universals and language-specific features

## Getting Started

See `technique.py` for a practical implementation that you can extend and customize for your specific probing experiments.