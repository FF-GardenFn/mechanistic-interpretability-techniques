# Concept Editing

## Overview
Concept editing is a mechanistic interpretability technique that modifies stored knowledge, associations, or concepts within trained language models without retraining. These methods enable precise updates to model knowledge and behavior by directly editing internal representations and weights.

## Who Proposed It
- **De Cao et al. (2021)** - "Editing Factual Knowledge in Language Models" (Early foundational work) - arXiv:2104.00164
- **Meng et al. (2022)** - "Locating and Editing Factual Associations in GPT" (ROME - Rank-One Model Editing) - arXiv:2202.05262
- **Meng et al. (2022/2023)** - "Mass-Editing Memory in a Transformer" (MEMIT) - arXiv:2210.07229, ICLR 2023
- **Mitchell et al. (2021/2022)** - "Fast Model Editing at Scale" (SERAC - Semi-Parametric Editing) - arXiv:2110.11309, ICLR 2022
- **Yao et al. (2023)** - "Editing Large Language Models: Problems, Methods, and Opportunities" (Comprehensive survey) - arXiv:2305.13172

## What It Does
Concept editing works by:
1. **Locating Knowledge**: Identifying where specific facts or concepts are stored in the model (Meng et al., 2022; Dai et al., 2022)
2. **Targeted Modification**: Precisely editing weights or representations to change stored knowledge (De Cao et al., 2021)
3. **Preserving Locality**: Maintaining model performance on unrelated tasks and knowledge (Mitchell et al., 2022; SERAC)
4. **Efficient Updates**: Avoiding the need for expensive retraining or fine-tuning (Meng et al., 2023; MEMIT)

Key capabilities include:
- **Factual Updates**: Correcting outdated or incorrect information
- **Concept Associations**: Modifying relationships between concepts
- **Bias Mitigation**: Reducing harmful associations and stereotypes
- **Knowledge Injection**: Adding new facts or capabilities to existing models

## Relevant Deductions
- **Knowledge Localization**: Factual knowledge is stored in specific, identifiable locations, particularly in MLP layers (Meng et al., 2022; ROME)
- **Distributed Representations**: Knowledge can be both localized and distributed across layers (Dai et al., 2022; Geva et al., 2021)
- **Causal Mechanisms**: How knowledge storage relates to generation mechanisms (De Cao et al., 2021; Hase et al., 2023)
- **Memory Architecture**: How transformer models organize and retrieve stored information as key-value memories (Geva et al., 2021)
- **Plasticity**: Models can be efficiently modified post-training while preserving capabilities (Mitchell et al., 2022; SERAC)

##  Potential Extensions
1. **Knowledge Mapping**: Create comprehensive maps of where different types of knowledge are stored (Dai et al., 2022)
2. **Interaction Analysis**: Study how editing one concept affects related knowledge (Hase et al., 2023; surprising differences in localization)
3. **Temporal Dynamics**: Examine how knowledge representations change during generation (MEMIT generation studies)
4. **Cross-Model Studies**: Compare knowledge storage across different model architectures (Yao et al., 2023 survey)
5. **Hierarchical Analysis**: Understand how abstract concepts relate to specific facts
6. **Robustness Testing**: Evaluate how well edits persist under various conditions (Hase et al., 2023)

## Key Papers and Citations

### Foundational Papers
- De Cao, N., Aziz, W., & Titov, I. (2021). "Editing Factual Knowledge in Language Models" EMNLP 2021. DOI:10.18653/v1/2021.emnlp-main.522
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). "Locating and Editing Factual Associations in GPT" (ROME) arXiv:2202.05262. DOI:10.48550/arXiv.2202.05262
- Meng, K., Sen Sharma, A., Andonian, A., Belinkov, Y., & Bau, D. (2023). "Mass-Editing Memory in a Transformer" (MEMIT) arXiv:2210.07229. DOI:10.48550/arXiv.2210.07229
- Mitchell, E., Lin, C., Bosselut, A., Finn, C., & Manning, C.D. (2022). "Fast Model Editing at Scale" (SERAC) arXiv:2110.11309. DOI:10.48550/arXiv.2110.11309

### Survey and Analysis
- Yao, Y., Wang, P., Tian, B., et al. (2023). "Editing Large Language Models: Problems, Methods, and Opportunities" arXiv:2305.13172. DOI:10.18653/v1/2023.emnlp-main.632
- Hase, P., Bansal, M., et al. (2023). "Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models" arXiv:2301.08585. DOI:10.48550/arXiv.2301.08585

### Related Work
- Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2022). "Knowledge Neurons in Pretrained Transformers" arXiv:2104.08696. DOI:10.18653/v1/2022.acl-long.581
- Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). "Transformer Feed-Forward Layers Are Key-Value Memories" arXiv:2012.14913. DOI:10.18653/v1/2021.emnlp-main.446

## Example Use Cases

### Factual Correction
- Updating outdated information (e.g., current presidents, recent events)
- Correcting historical inaccuracies or misconceptions
- Fixing model hallucinations and false facts

### Bias Mitigation
- Reducing gender, racial, or cultural biases in model outputs (Yao et al., 2023)
- Modifying harmful stereotypes and associations
- Promoting more inclusive and fair representations

### Knowledge Updates
- Adding new scientific discoveries or technological developments
- Incorporating recent world events and changes
- Updating domain-specific knowledge

### Research Applications
- Testing theories about knowledge representation
- Understanding causal relationships in model behavior
- Developing better training and alignment methods

## Types of Concept Editing

### ROME (Rank-One Model Editing)
- Identifies specific neurons responsible for factual associations
- Uses rank-one updates to modify MLP weights
- Preserves model performance on unrelated tasks

### MEMIT (Mass Editing Memory in a Transformer)
- Enables simultaneous editing of multiple facts
- Uses covariance statistics to optimize edit locations
- Scales better than single-fact editing methods

### SERAC (Semi-Parametric Editing with External Memory)
- Uses external memory module to store edits (semi-parametric approach)
- Classifies whether to use original model or edited knowledge
- Allows for complex, conditional edits
- Hypernetwork-based fast adaptation at scale

### Fine-Tuning Approaches
- Hypernetwork-based editing
- Adapter-based modifications
- Constrained optimization methods

## Evaluation Criteria

### Efficacy
- Does the edit successfully change the target behavior?
- How reliably does the model produce the desired output?

### Locality
- Are unrelated model capabilities preserved?
- Does the edit avoid unintended side effects?

### Generalization
- Does the edit apply to paraphrases and related queries?
- Can the model reason with the new knowledge?

### Portability
- Does the edit transfer to related facts and contexts? (SERAC evaluation)
- How well does it integrate with existing knowledge?

## Technical Challenges
- Balancing specificity with generalization (Yao et al., 2023)
- Maintaining model coherence after multiple edits (MEMIT scaling challenges)
- Scaling to large-scale knowledge updates (Yao et al., 2023)
- Handling contradictory or complex knowledge relationships (Hase et al., 2023)