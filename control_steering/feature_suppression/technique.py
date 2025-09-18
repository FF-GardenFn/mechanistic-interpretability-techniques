"""
Feature Suppression Implementation
Techniques for selectively suppressing neurons, attention heads, and model components
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Configuration Classes
@dataclass
class SuppressionConfig:
    """Configuration for feature suppression experiments"""
    model_name: str = "gpt2-small"
    suppression_strength: float = 1.0  # 0.0 = no suppression, 1.0 = complete suppression
    method: str = "zero"  # "zero", "noise", "mean", "learned"
    target_layers: Optional[List[int]] = None
    max_tokens: int = 50

@dataclass
class SuppressionTarget:
    """Represents a target for suppression"""
    component_type: str  # "neuron", "attention_head", "mlp", "attention"
    layer: int
    indices: Union[int, List[int], slice]  # Which neurons/heads to suppress
    strength: float = 1.0
    description: str = ""

#%% Core Feature Suppression Class
class FeatureSuppression:
    """
    Main class for implementing feature suppression techniques
    """

    def __init__(self, config: SuppressionConfig):
        self.config = config
        self.model = HookedTransformer.from_pretrained(config.model_name)
        self.model.to(device)
        self.suppression_targets = []
        self.baseline_cache = None

    def add_suppression_target(self, target: SuppressionTarget):
        """Add a new suppression target"""
        self.suppression_targets.append(target)

    def neuron_suppression_hook(
        self,
        activations: torch.Tensor,
        hook,
        target_indices: Union[int, List[int], slice],
        strength: float = 1.0,
        method: str = "zero"
    ):
        """
        Hook function to suppress specific neurons

        Args:
            activations: Input activations
            hook: Hook object
            target_indices: Which neurons to suppress
            strength: Suppression strength (0.0 = no suppression, 1.0 = complete)
            method: Suppression method ("zero", "noise", "mean")
        """
        if method == "zero":
            # Zero out the targeted neurons
            if isinstance(target_indices, slice):
                activations[:, :, target_indices] *= (1.0 - strength)
            elif isinstance(target_indices, list):
                for idx in target_indices:
                    activations[:, :, idx] *= (1.0 - strength)
            else:
                activations[:, :, target_indices] *= (1.0 - strength)

        elif method == "noise":
            # Replace with noise
            noise_scale = activations.std() * 0.1
            noise = torch.randn_like(activations) * noise_scale
            if isinstance(target_indices, slice):
                activations[:, :, target_indices] = (
                    activations[:, :, target_indices] * (1.0 - strength) +
                    noise[:, :, target_indices] * strength
                )
            elif isinstance(target_indices, list):
                for idx in target_indices:
                    activations[:, :, idx] = (
                        activations[:, :, idx] * (1.0 - strength) +
                        noise[:, :, idx] * strength
                    )
            else:
                activations[:, :, target_indices] = (
                    activations[:, :, target_indices] * (1.0 - strength) +
                    noise[:, :, target_indices] * strength
                )

        elif method == "mean":
            # Replace with mean activation
            if isinstance(target_indices, slice):
                mean_val = activations[:, :, target_indices].mean()
                activations[:, :, target_indices] = (
                    activations[:, :, target_indices] * (1.0 - strength) +
                    mean_val * strength
                )
            elif isinstance(target_indices, list):
                for idx in target_indices:
                    mean_val = activations[:, :, idx].mean()
                    activations[:, :, idx] = (
                        activations[:, :, idx] * (1.0 - strength) +
                        mean_val * strength
                    )
            else:
                mean_val = activations[:, :, target_indices].mean()
                activations[:, :, target_indices] = (
                    activations[:, :, target_indices] * (1.0 - strength) +
                    mean_val * strength
                )

        return activations

    def attention_suppression_hook(
        self,
        pattern: torch.Tensor,
        hook,
        target_heads: Union[int, List[int]],
        strength: float = 1.0
    ):
        """
        Hook function to suppress specific attention heads

        Args:
            pattern: Attention patterns [batch, head, seq_len, seq_len]
            hook: Hook object
            target_heads: Which attention heads to suppress
            strength: Suppression strength
        """
        if isinstance(target_heads, list):
            for head in target_heads:
                pattern[:, head, :, :] *= (1.0 - strength)
        else:
            pattern[:, target_heads, :, :] *= (1.0 - strength)

        return pattern

    def generate_with_suppression(
        self,
        prompt: str,
        max_tokens: int = None,
        return_cache: bool = False
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Generate text with feature suppression applied

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            return_cache: Whether to return activation cache

        Returns:
            Generated text or (text, cache) if return_cache=True
        """
        max_tokens = max_tokens or self.config.max_tokens

        # Create hooks for all suppression targets
        hooks = []
        for target in self.suppression_targets:
            if target.component_type == "neuron":
                act_name = get_act_name("resid_post", target.layer)
                hook_fn = lambda activations, hook, t=target: self.neuron_suppression_hook(
                    activations, hook, t.indices, t.strength, self.config.method
                )
                hooks.append((act_name, hook_fn))

            elif target.component_type == "mlp":
                act_name = get_act_name("mlp_out", target.layer)
                hook_fn = lambda activations, hook, t=target: self.neuron_suppression_hook(
                    activations, hook, t.indices, t.strength, self.config.method
                )
                hooks.append((act_name, hook_fn))

            elif target.component_type == "attention_head":
                act_name = get_act_name("pattern", target.layer)
                hook_fn = lambda pattern, hook, t=target: self.attention_suppression_hook(
                    pattern, hook, t.indices, t.strength
                )
                hooks.append((act_name, hook_fn))

        # Generate with hooks
        tokens = self.model.to_tokens(prompt)
        with self.model.hooks(hooks):
            if return_cache:
                output, cache = self.model.run_with_cache(
                    tokens,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7
                )
                return self.model.to_string(output[0]), cache
            else:
                output = self.model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7
                )
                return self.model.to_string(output[0])

    def systematic_neuron_ablation(
        self,
        prompt: str,
        layer: int,
        neuron_range: Optional[Tuple[int, int]] = None,
        batch_size: int = 50
    ) -> Dict[int, str]:
        """
        Systematically ablate individual neurons and measure effects

        Args:
            prompt: Test prompt
            layer: Layer to ablate
            neuron_range: Range of neurons to test (default: all)
            batch_size: How many neurons to test in each batch

        Returns:
            Dictionary mapping neuron index to generated text
        """
        d_model = self.model.cfg.d_model
        if neuron_range is None:
            neuron_range = (0, d_model)

        results = {}
        start_idx, end_idx = neuron_range

        print(f"Testing neurons {start_idx} to {end_idx} in layer {layer}")

        # Clear existing targets
        self.suppression_targets = []

        for neuron_idx in range(start_idx, end_idx, batch_size):
            batch_end = min(neuron_idx + batch_size, end_idx)
            batch_neurons = list(range(neuron_idx, batch_end))

            for neuron in batch_neurons:
                # Clear previous targets and add current neuron
                self.suppression_targets = []
                target = SuppressionTarget(
                    component_type="neuron",
                    layer=layer,
                    indices=neuron,
                    strength=1.0,
                    description=f"Neuron {neuron} in layer {layer}"
                )
                self.add_suppression_target(target)

                # Generate with this neuron suppressed
                try:
                    generated = self.generate_with_suppression(prompt, max_tokens=20)
                    results[neuron] = generated
                    print(f"Neuron {neuron}: {generated[:50]}...")
                except Exception as e:
                    print(f"Error with neuron {neuron}: {e}")
                    results[neuron] = f"ERROR: {str(e)}"

        return results

    def attention_head_ablation(
        self,
        prompt: str,
        layer: int = None
    ) -> Dict[Tuple[int, int], str]:
        """
        Systematically ablate attention heads and measure effects

        Args:
            prompt: Test prompt
            layer: Specific layer to test (if None, tests all layers)

        Returns:
            Dictionary mapping (layer, head) to generated text
        """
        results = {}
        layers_to_test = [layer] if layer is not None else range(self.model.cfg.n_layers)

        for test_layer in layers_to_test:
            for head in range(self.model.cfg.n_heads):
                # Clear previous targets and add current head
                self.suppression_targets = []
                target = SuppressionTarget(
                    component_type="attention_head",
                    layer=test_layer,
                    indices=head,
                    strength=1.0,
                    description=f"Head {head} in layer {test_layer}"
                )
                self.add_suppression_target(target)

                # Generate with this head suppressed
                try:
                    generated = self.generate_with_suppression(prompt, max_tokens=20)
                    results[(test_layer, head)] = generated
                    print(f"Layer {test_layer}, Head {head}: {generated[:50]}...")
                except Exception as e:
                    print(f"Error with layer {test_layer}, head {head}: {e}")
                    results[(test_layer, head)] = f"ERROR: {str(e)}"

        return results

#%% Analysis Functions
def analyze_suppression_effects(
    suppressor: FeatureSuppression,
    prompts: List[str],
    component_type: str = "neuron",
    layer: int = 6
):
    """Analyze the effects of suppressing different components"""

    results = []
    baseline_results = []

    # Get baseline generations
    suppressor.suppression_targets = []  # Clear all suppressions
    for prompt in prompts:
        baseline = suppressor.model.generate(
            suppressor.model.to_tokens(prompt),
            max_new_tokens=20,
            do_sample=False
        )
        baseline_results.append(suppressor.model.to_string(baseline[0]))

    # Test suppression effects
    if component_type == "neuron":
        # Test a sample of neurons
        neurons_to_test = [0, 100, 200, 300, 400, 500]  # Adjust based on model size
        for neuron in neurons_to_test:
            suppressor.suppression_targets = []
            target = SuppressionTarget(
                component_type="neuron",
                layer=layer,
                indices=neuron,
                strength=1.0
            )
            suppressor.add_suppression_target(target)

            for i, prompt in enumerate(prompts):
                suppressed = suppressor.generate_with_suppression(prompt, max_tokens=20)
                results.append({
                    'prompt_idx': i,
                    'prompt': prompt,
                    'component': f"neuron_{neuron}",
                    'baseline': baseline_results[i],
                    'suppressed': suppressed,
                    'changed': baseline_results[i] != suppressed
                })

    elif component_type == "attention_head":
        # Test all attention heads in the specified layer
        for head in range(suppressor.model.cfg.n_heads):
            suppressor.suppression_targets = []
            target = SuppressionTarget(
                component_type="attention_head",
                layer=layer,
                indices=head,
                strength=1.0
            )
            suppressor.add_suppression_target(target)

            for i, prompt in enumerate(prompts):
                suppressed = suppressor.generate_with_suppression(prompt, max_tokens=20)
                results.append({
                    'prompt_idx': i,
                    'prompt': prompt,
                    'component': f"head_{head}",
                    'baseline': baseline_results[i],
                    'suppressed': suppressed,
                    'changed': baseline_results[i] != suppressed
                })

    return pd.DataFrame(results)

#%% Visualization Functions
def visualize_suppression_effects(results_df: pd.DataFrame):
    """Visualize the effects of component suppression"""

    # Calculate change rates for each component
    change_rates = results_df.groupby('component')['changed'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    change_rates.plot(kind='bar')
    plt.title('Effect of Component Suppression on Text Generation')
    plt.xlabel('Component')
    plt.ylabel('Fraction of Prompts Changed')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Show examples of changes
    plt.subplot(2, 1, 2)
    changed_examples = results_df[results_df['changed']].groupby('component').first()
    if len(changed_examples) > 0:
        example_component = changed_examples.index[0]
        example_row = changed_examples.loc[example_component]

        plt.text(0.05, 0.7, f"Example: {example_component}", fontsize=12, weight='bold')
        plt.text(0.05, 0.5, f"Baseline: {example_row['baseline'][:60]}...", fontsize=10)
        plt.text(0.05, 0.3, f"Suppressed: {example_row['suppressed'][:60]}...", fontsize=10)
        plt.text(0.05, 0.1, f"Prompt: {example_row['prompt'][:60]}...", fontsize=10)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_suppression_heatmap(
    suppressor: FeatureSuppression,
    prompts: List[str],
    layers: List[int],
    component_type: str = "attention_head"
):
    """Create heatmap showing suppression effects across layers and components"""

    effect_matrix = []

    for layer in layers:
        layer_effects = []
        if component_type == "attention_head":
            for head in range(suppressor.model.cfg.n_heads):
                suppressor.suppression_targets = []
                target = SuppressionTarget(
                    component_type="attention_head",
                    layer=layer,
                    indices=head,
                    strength=1.0
                )
                suppressor.add_suppression_target(target)

                # Count how many prompts are affected
                changes = 0
                for prompt in prompts:
                    # Get baseline
                    suppressor.suppression_targets = []
                    baseline = suppressor.model.generate(
                        suppressor.model.to_tokens(prompt),
                        max_new_tokens=10,
                        do_sample=False
                    )
                    baseline_text = suppressor.model.to_string(baseline[0])

                    # Get suppressed
                    suppressor.suppression_targets = [target]
                    suppressed_text = suppressor.generate_with_suppression(prompt, max_tokens=10)

                    if baseline_text != suppressed_text:
                        changes += 1

                layer_effects.append(changes / len(prompts))

        effect_matrix.append(layer_effects)

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        effect_matrix,
        xticklabels=[f"Head {i}" for i in range(suppressor.model.cfg.n_heads)],
        yticklabels=[f"Layer {i}" for i in layers],
        cmap='Reds',
        annot=True,
        fmt='.2f'
    )
    plt.title(f'Suppression Effect Heatmap: {component_type.replace("_", " ").title()}')
    plt.xlabel('Component Index')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.show()

    return np.array(effect_matrix)

#%% Example Usage
def run_feature_suppression_example():
    """Example: Suppress neurons to modify model behavior"""

    # Initialize suppression system
    config = SuppressionConfig(model_name="gpt2-small", method="zero")
    suppressor = FeatureSuppression(config)

    # Test prompts
    test_prompts = [
        "The weather today is",
        "I think the movie was",
        "Technology in the future will",
        "The most important thing in life is"
    ]

    print("Testing neuron suppression effects...")

    # Test systematic neuron ablation on a small range
    results = suppressor.systematic_neuron_ablation(
        prompt="The weather today is",
        layer=6,
        neuron_range=(0, 50),  # Test first 50 neurons
        batch_size=10
    )

    print(f"\nTested {len(results)} neurons")
    print("Most different results:")

    # Find the most different results
    baseline = suppressor.model.generate(
        suppressor.model.to_tokens("The weather today is"),
        max_new_tokens=20,
        do_sample=False
    )
    baseline_text = suppressor.model.to_string(baseline[0])

    different_results = []
    for neuron, generated in results.items():
        if generated != baseline_text and not generated.startswith("ERROR"):
            different_results.append((neuron, generated))

    for neuron, text in different_results[:5]:  # Show top 5
        print(f"Neuron {neuron}: {text}")

    # Test attention head suppression
    print("\n" + "="*50)
    print("Testing attention head suppression...")

    head_results = suppressor.attention_head_ablation(
        prompt="The weather today is",
        layer=6
    )

    print(f"Tested {len(head_results)} attention heads")

#%% Main Execution
if __name__ == "__main__":
    print("Running Feature Suppression Example...")
    run_feature_suppression_example()

    print("\n" + "="*50)
    print("For advanced analysis, run:")
    print("1. analyze_suppression_effects(suppressor, prompts)")
    print("2. visualize_suppression_effects(results_df)")
    print("3. plot_suppression_heatmap(suppressor, prompts, layers)")
    print("="*50)