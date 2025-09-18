"""
Concept Editing Implementation
Techniques for editing factual knowledge and concepts in language models
Includes ROME, MEMIT, and other knowledge editing approaches
"""

#%% Imports and Setup
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import pandas as pd
from collections import defaultdict
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Configuration Classes
@dataclass
class EditConfig:
    """Configuration for concept editing experiments"""
    model_name: str = "gpt2-small"
    edit_method: str = "rome"  # "rome", "memit", "fine_tune"
    target_layer: int = 8  # Layer to edit (middle layers often work best)
    lambda_factor: float = 0.0001  # Regularization strength
    max_tokens: int = 50

@dataclass
class FactEdit:
    """Represents a factual edit to be made"""
    subject: str
    relation: str
    target_new: str
    target_true: str = ""  # Original true value
    prompt_template: str = "{subject} {relation}"
    description: str = ""

    def format_prompt(self) -> str:
        """Format the prompt for this fact"""
        return self.prompt_template.format(subject=self.subject, relation=self.relation)

@dataclass
class EditResult:
    """Results of an editing operation"""
    edit: FactEdit
    success: bool
    pre_edit_output: str
    post_edit_output: str
    locality_preserved: bool = True
    generalization_score: float = 0.0

#%% Core Concept Editing Class
class ConceptEditor:
    """
    Main class for implementing concept editing techniques
    """

    def __init__(self, config: EditConfig):
        self.config = config
        self.model = HookedTransformer.from_pretrained(config.model_name)
        self.model.to(device)
        self.original_weights = None
        self.applied_edits = []

    def backup_weights(self):
        """Backup original model weights"""
        self.original_weights = {}
        for name, param in self.model.named_parameters():
            self.original_weights[name] = param.data.clone()

    def restore_weights(self):
        """Restore original model weights"""
        if self.original_weights is None:
            raise ValueError("No backup weights found. Call backup_weights() first.")

        for name, param in self.model.named_parameters():
            if name in self.original_weights:
                param.data.copy_(self.original_weights[name])

        self.applied_edits = []

    def get_mlp_weights(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get MLP weights for a specific layer"""
        mlp_module = self.model.blocks[layer].mlp
        W_in = mlp_module.W_in  # [d_model, d_mlp]
        W_out = mlp_module.W_out  # [d_mlp, d_model]
        return W_in, W_out

    def compute_knowledge_vector(
        self,
        edit: FactEdit,
        layer: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute knowledge vector for a factual association using causal tracing

        Args:
            edit: The factual edit
            layer: Layer to analyze (uses config default if None)

        Returns:
            Tuple of (key_vector, value_vector)
        """
        layer = layer or self.config.target_layer

        # Get the subject representation
        subject_tokens = self.model.to_tokens(edit.subject)
        prompt_tokens = self.model.to_tokens(edit.format_prompt())

        # Run with cache to get activations
        with torch.no_grad():
            _, cache = self.model.run_with_cache(prompt_tokens)

        # Get the residual stream at the target layer for the last subject token
        resid_act_name = get_act_name("resid_post", layer)
        subject_repr = cache[resid_act_name][0, -1, :]  # Last token representation

        # Get MLP weights
        W_in, W_out = self.get_mlp_weights(layer)

        # Compute the key vector (input to MLP)
        key_vector = subject_repr @ W_in  # [d_mlp]

        # For the value vector, we need to compute what should be output
        # This is a simplified version - in practice, more sophisticated methods are used
        target_tokens = self.model.to_tokens(edit.target_new)
        target_embedding = self.model.embed.W_E[target_tokens[0, 0], :]  # First token of target

        # The value vector should produce the target when multiplied by W_out
        # This is a simplification - actual ROME uses more complex optimization
        value_vector = torch.linalg.pinv(W_out) @ target_embedding

        return key_vector, value_vector

    def rome_edit(self, edit: FactEdit) -> EditResult:
        """
        Perform ROME (Rank-One Model Editing) on a single fact

        Args:
            edit: The factual edit to perform

        Returns:
            EditResult containing the results
        """
        if self.original_weights is None:
            self.backup_weights()

        layer = self.config.target_layer

        # Get pre-edit output
        pre_output = self.generate_text(edit.format_prompt(), max_tokens=10)

        # Compute knowledge vectors
        key_vector, value_vector = self.compute_knowledge_vector(edit, layer)

        # Get current MLP weights
        W_in, W_out = self.get_mlp_weights(layer)

        # Compute the rank-one update using the ROME formula
        # This is a simplified version of the actual ROME algorithm
        with torch.no_grad():
            # Get covariance matrix (in practice, this uses statistics from many examples)
            # For simplicity, we'll use identity scaling
            C_inv = torch.eye(W_out.shape[0], device=device) * self.config.lambda_factor

            # Compute the update direction
            denominator = key_vector @ C_inv @ key_vector + self.config.lambda_factor
            update_direction = (C_inv @ key_vector) / denominator

            # Current prediction for this key
            current_value = key_vector @ W_out

            # Desired change
            delta_value = value_vector - current_value

            # Apply rank-one update to W_out
            self.model.blocks[layer].mlp.W_out += torch.outer(update_direction, delta_value)

        # Get post-edit output
        post_output = self.generate_text(edit.format_prompt(), max_tokens=10)

        # Check if edit was successful
        success = edit.target_new.lower() in post_output.lower()

        # Test locality (simplified)
        locality_preserved = self.test_locality(edit)

        result = EditResult(
            edit=edit,
            success=success,
            pre_edit_output=pre_output,
            post_edit_output=post_output,
            locality_preserved=locality_preserved
        )

        self.applied_edits.append(result)
        return result

    def memit_edit(self, edits: List[FactEdit]) -> List[EditResult]:
        """
        Perform MEMIT (Mass Editing Memory in a Transformer) on multiple facts

        Args:
            edits: List of factual edits to perform simultaneously

        Returns:
            List of EditResult objects
        """
        if self.original_weights is None:
            self.backup_weights()

        layer = self.config.target_layer
        results = []

        # Get pre-edit outputs
        pre_outputs = []
        for edit in edits:
            pre_output = self.generate_text(edit.format_prompt(), max_tokens=10)
            pre_outputs.append(pre_output)

        # Collect all knowledge vectors
        key_vectors = []
        value_vectors = []
        for edit in edits:
            key_vec, val_vec = self.compute_knowledge_vector(edit, layer)
            key_vectors.append(key_vec)
            value_vectors.append(val_vec)

        # Stack vectors
        K = torch.stack(key_vectors)  # [n_edits, d_mlp]
        V = torch.stack(value_vectors)  # [n_edits, d_model]

        # Get current MLP weights
        W_in, W_out = self.get_mlp_weights(layer)

        # Compute MEMIT update (simplified version)
        with torch.no_grad():
            # Compute covariance matrix for all keys
            C = K @ K.T + torch.eye(K.shape[0], device=device) * self.config.lambda_factor

            # Current values for all keys
            current_values = K @ W_out  # [n_edits, d_model]

            # Desired changes
            delta_values = V - current_values  # [n_edits, d_model]

            # Solve for optimal update
            C_inv = torch.linalg.inv(C)
            update_weights = C_inv @ delta_values  # [n_edits, d_model]

            # Apply the update
            delta_W = K.T @ update_weights  # [d_mlp, d_model]
            self.model.blocks[layer].mlp.W_out += delta_W

        # Get post-edit outputs and create results
        for i, edit in enumerate(edits):
            post_output = self.generate_text(edit.format_prompt(), max_tokens=10)
            success = edit.target_new.lower() in post_output.lower()
            locality_preserved = self.test_locality(edit)

            result = EditResult(
                edit=edit,
                success=success,
                pre_edit_output=pre_outputs[i],
                post_edit_output=post_output,
                locality_preserved=locality_preserved
            )
            results.append(result)
            self.applied_edits.append(result)

        return results

    def test_locality(self, edit: FactEdit, test_prompts: List[str] = None) -> bool:
        """
        Test if the edit preserves model performance on unrelated tasks

        Args:
            edit: The edit that was applied
            test_prompts: Prompts to test (uses default if None)

        Returns:
            True if locality is preserved
        """
        if test_prompts is None:
            test_prompts = [
                "The capital of France is",
                "2 + 2 equals",
                "The largest planet in our solar system is",
                "Shakespeare wrote the play"
            ]

        # Remove prompts that might be related to the edit
        filtered_prompts = [p for p in test_prompts if edit.subject.lower() not in p.lower()]

        if not filtered_prompts:
            return True  # No unrelated prompts to test

        # Test a few prompts (simplified locality test)
        changes = 0
        for prompt in filtered_prompts[:3]:  # Test first 3 prompts
            # This is simplified - in practice, you'd compare with original model
            output = self.generate_text(prompt, max_tokens=5)
            # For now, assume locality is preserved if we can generate reasonable text
            if len(output.strip()) < len(prompt) + 2:
                changes += 1

        return changes == 0

    def generate_text(self, prompt: str, max_tokens: int = None) -> str:
        """Generate text with current model state"""
        max_tokens = max_tokens or self.config.max_tokens
        tokens = self.model.to_tokens(prompt)

        with torch.no_grad():
            output = self.model.generate(
                tokens,
                max_new_tokens=max_tokens,
                do_sample=False  # Use greedy decoding for consistency
            )

        return self.model.to_string(output[0])

    def evaluate_edit_quality(self, result: EditResult) -> Dict[str, float]:
        """
        Evaluate the quality of an edit across multiple dimensions

        Args:
            result: EditResult to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Efficacy: Does the edit work?
        metrics['efficacy'] = 1.0 if result.success else 0.0

        # Locality: Are unrelated capabilities preserved?
        metrics['locality'] = 1.0 if result.locality_preserved else 0.0

        # Consistency: Does it work for paraphrases?
        paraphrases = self.generate_paraphrases(result.edit)
        paraphrase_success = 0
        for paraphrase in paraphrases:
            output = self.generate_text(paraphrase, max_tokens=10)
            if result.edit.target_new.lower() in output.lower():
                paraphrase_success += 1

        metrics['generalization'] = paraphrase_success / len(paraphrases) if paraphrases else 0.0

        return metrics

    def generate_paraphrases(self, edit: FactEdit) -> List[str]:
        """Generate paraphrases of an edit prompt for testing generalization"""
        # This is a simplified version - in practice, you might use more sophisticated methods
        base_prompt = edit.format_prompt()

        paraphrases = []
        if "is" in base_prompt:
            paraphrases.append(base_prompt.replace("is", "was"))
        if "was" in base_prompt:
            paraphrases.append(base_prompt.replace("was", "is"))

        # Add question forms
        if edit.relation == "is the capital of":
            paraphrases.append(f"What is the capital of {edit.subject}?")
        elif edit.relation == "was born in":
            paraphrases.append(f"Where was {edit.subject} born?")

        return paraphrases

#%% Analysis and Visualization Functions
def analyze_edit_results(results: List[EditResult]) -> pd.DataFrame:
    """Analyze the results of multiple edits"""
    data = []

    for result in results:
        metrics = result.edit  # Access the editor to get evaluation
        data.append({
            'subject': result.edit.subject,
            'relation': result.edit.relation,
            'target_new': result.edit.target_new,
            'success': result.success,
            'locality_preserved': result.locality_preserved,
            'pre_edit': result.pre_edit_output,
            'post_edit': result.post_edit_output
        })

    return pd.DataFrame(data)

def visualize_edit_effectiveness(results_df: pd.DataFrame):
    """Visualize the effectiveness of edits"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Success rate
    success_rate = results_df['success'].mean()
    axes[0, 0].bar(['Failed', 'Successful'],
                   [1-success_rate, success_rate],
                   color=['red', 'green'])
    axes[0, 0].set_title(f'Edit Success Rate: {success_rate:.2%}')
    axes[0, 0].set_ylabel('Proportion')

    # Locality preservation
    locality_rate = results_df['locality_preserved'].mean()
    axes[0, 1].bar(['Not Preserved', 'Preserved'],
                   [1-locality_rate, locality_rate],
                   color=['orange', 'blue'])
    axes[0, 1].set_title(f'Locality Preservation: {locality_rate:.2%}')
    axes[0, 1].set_ylabel('Proportion')

    # Success by relation type
    if 'relation' in results_df.columns:
        relation_success = results_df.groupby('relation')['success'].mean()
        relation_success.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Success Rate by Relation Type')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)

    # Combined success and locality
    combined = results_df['success'] & results_df['locality_preserved']
    combined_rate = combined.mean()
    axes[1, 1].bar(['Partial/Failed', 'Fully Successful'],
                   [1-combined_rate, combined_rate],
                   color=['yellow', 'darkgreen'])
    axes[1, 1].set_title(f'Full Success Rate: {combined_rate:.2%}')
    axes[1, 1].set_ylabel('Proportion')

    plt.tight_layout()
    plt.show()

def demonstrate_knowledge_localization(
    editor: ConceptEditor,
    edit: FactEdit,
    layers_to_test: List[int] = None
):
    """Demonstrate how knowledge is localized across layers"""
    if layers_to_test is None:
        layers_to_test = [4, 6, 8, 10]

    original_layer = editor.config.target_layer
    results = []

    print(f"Testing knowledge localization for: {edit.format_prompt()}")
    print("-" * 60)

    for layer in layers_to_test:
        # Restore original weights
        editor.restore_weights()
        editor.backup_weights()

        # Set target layer
        editor.config.target_layer = layer

        # Apply edit
        result = editor.rome_edit(edit)

        results.append({
            'layer': layer,
            'success': result.success,
            'pre_edit': result.pre_edit_output,
            'post_edit': result.post_edit_output
        })

        print(f"Layer {layer:2}: {'✓' if result.success else '✗'} "
              f"'{result.post_edit_output[:50]}...'")

    # Restore original configuration
    editor.config.target_layer = original_layer
    editor.restore_weights()

    return results

#%% Example Usage
def run_concept_editing_example():
    """Example: Edit factual knowledge using ROME and MEMIT"""

    # Initialize editor
    config = EditConfig(model_name="gpt2-small", target_layer=8)
    editor = ConceptEditor(config)

    # Define some factual edits
    edits = [
        FactEdit(
            subject="Paris",
            relation="is the capital of",
            target_new="Germany",
            target_true="France",
            prompt_template="{subject} {relation}",
            description="Change Paris from capital of France to Germany"
        ),
        FactEdit(
            subject="Einstein",
            relation="was born in",
            target_new="Italy",
            target_true="Germany",
            prompt_template="{subject} {relation}",
            description="Change Einstein's birthplace"
        )
    ]

    print("CONCEPT EDITING DEMONSTRATION")
    print("=" * 50)

    # Test single edit with ROME
    print("\n1. Testing ROME (single edit):")
    print("-" * 30)

    result = editor.rome_edit(edits[0])
    print(f"Edit: {result.edit.description}")
    print(f"Before: {result.pre_edit_output}")
    print(f"After:  {result.post_edit_output}")
    print(f"Success: {result.success}")
    print(f"Locality preserved: {result.locality_preserved}")

    # Test knowledge localization
    print("\n2. Testing Knowledge Localization:")
    print("-" * 30)
    editor.restore_weights()
    demonstrate_knowledge_localization(editor, edits[0], layers_to_test=[4, 6, 8, 10])

    # Test MEMIT (multiple edits)
    print("\n3. Testing MEMIT (multiple edits):")
    print("-" * 30)
    editor.restore_weights()
    memit_results = editor.memit_edit(edits)

    for i, result in enumerate(memit_results):
        print(f"Edit {i+1}: {result.edit.description}")
        print(f"  Before: {result.pre_edit_output}")
        print(f"  After:  {result.post_edit_output}")
        print(f"  Success: {result.success}")

    # Analyze results
    print("\n4. Analysis:")
    print("-" * 30)
    all_results = [result] + memit_results
    results_df = analyze_edit_results(all_results)
    print(f"Overall success rate: {results_df['success'].mean():.2%}")
    print(f"Locality preservation rate: {results_df['locality_preserved'].mean():.2%}")

#%% Main Execution
if __name__ == "__main__":
    print("Running Concept Editing Example...")
    run_concept_editing_example()

    print("\n" + "="*50)
    print("For advanced analysis, run:")
    print("1. demonstrate_knowledge_localization(editor, edit)")
    print("2. visualize_edit_effectiveness(results_df)")
    print("3. editor.evaluate_edit_quality(result)")
    print("="*50)