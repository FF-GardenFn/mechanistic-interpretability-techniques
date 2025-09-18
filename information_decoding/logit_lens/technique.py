# %% [markdown]
# # Logit Lens for Mechanistic Interpretability
#
# This notebook implements the logit lens technique to analyze what tokens
# transformer models are "thinking about" at each layer by projecting
# intermediate representations through the final unembedding matrix.

# %% [markdown]
# ## Imports and Setup

# %%
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## Logit Lens Class

# %%
class LogitLens:
    """
    Implementation of the logit lens technique for analyzing transformer representations.

    The logit lens projects intermediate activations through the final unembedding matrix
    to reveal what tokens the model is considering at each layer.
    """

    def __init__(self, model_name: str = "gpt2-small", device: str = "cpu"):
        """
        Initialize the logit lens with a transformer model.

        Args:
            model_name: Name of the transformer model to load
            device: Device to run computations on
        """
        self.device = device
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.W_U = self.model.W_U  # Unembedding matrix
        self.ln_final = self.model.ln_final if hasattr(self.model, 'ln_final') else None

    def apply_logit_lens(self,
                        activations: torch.Tensor,
                        apply_ln: bool = True) -> torch.Tensor:
        """
        Apply the logit lens to a set of activations.

        Args:
            activations: Tensor of shape [batch, seq_len, d_model]
            apply_ln: Whether to apply final layer norm before unembedding

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size]
        """
        if apply_ln and self.ln_final is not None:
            # Apply final layer normalization
            normalized_acts = self.ln_final(activations)
        else:
            normalized_acts = activations

        # Project through unembedding matrix
        logits = normalized_acts @ self.W_U

        return logits

    def get_layer_predictions(self,
                            text: str,
                            layers: Optional[List[int]] = None,
                            positions: Optional[List[int]] = None,
                            component: str = "resid_post",
                            top_k: int = 10) -> Dict[int, Dict[str, any]]:
        """
        Get top-k token predictions for each layer using the logit lens.

        Args:
            text: Input text to analyze
            layers: Layers to analyze (default: all layers)
            positions: Token positions to analyze (default: all positions)
            component: Model component to extract (resid_post, resid_pre, etc.)
            top_k: Number of top predictions to return

        Returns:
            Dictionary mapping layers to prediction information
        """
        if layers is None:
            layers = list(range(self.model.cfg.n_layers))

        # Tokenize input
        tokens = self.model.to_tokens(text)
        batch_size, seq_len = tokens.shape

        if positions is None:
            positions = list(range(seq_len))

        results = {}

        # Extract activations and apply logit lens for each layer
        def hook_fn(activation, hook):
            layer_num = int(hook.name.split('.')[1])
            if layer_num in layers:
                # Apply logit lens
                logits = self.apply_logit_lens(activation)

                # Get probabilities
                probs = F.softmax(logits, dim=-1)

                # Store results for specified positions
                layer_results = {}
                for pos in positions:
                    if pos < seq_len:
                        pos_logits = logits[0, pos, :]  # [vocab_size]
                        pos_probs = probs[0, pos, :]    # [vocab_size]

                        # Get top-k predictions
                        top_k_probs, top_k_indices = torch.topk(pos_probs, top_k)
                        top_k_tokens = [self.model.tokenizer.decode([idx]) for idx in top_k_indices]

                        layer_results[pos] = {
                            'logits': pos_logits.detach().cpu(),
                            'probs': pos_probs.detach().cpu(),
                            'top_k_indices': top_k_indices.detach().cpu(),
                            'top_k_probs': top_k_probs.detach().cpu(),
                            'top_k_tokens': top_k_tokens
                        }

                results[layer_num] = layer_results

        # Register hooks
        hook_names = [f"blocks.{layer}.{component}" for layer in layers]
        hooks = []
        for name in hook_names:
            if name in self.model.hook_dict:
                hooks.append(self.model.add_hook(name, hook_fn))

        # Run forward pass
        with torch.no_grad():
            _ = self.model(tokens)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return results

    def analyze_token_evolution(self,
                              text: str,
                              target_token: str,
                              position: int = -1) -> Dict[int, float]:
        """
        Track how the probability of a specific token evolves across layers.

        Args:
            text: Input text to analyze
            target_token: Token to track
            position: Position to analyze (-1 for last token)

        Returns:
            Dictionary mapping layer numbers to token probabilities
        """
        # Get target token ID
        target_token_id = self.model.tokenizer.encode(target_token)[0]

        # Get predictions for all layers
        predictions = self.get_layer_predictions(text, positions=[position])

        # Extract probabilities for target token
        token_probs = {}
        for layer, layer_data in predictions.items():
            if position in layer_data:
                pos_data = layer_data[position]
                prob = pos_data['probs'][target_token_id].item()
                token_probs[layer] = prob

        return token_probs

    def compare_components(self,
                          text: str,
                          layer: int,
                          position: int = -1,
                          top_k: int = 5) -> Dict[str, Dict[str, any]]:
        """
        Compare predictions after different components (attention, MLP, etc.).

        Args:
            text: Input text to analyze
            layer: Layer to analyze
            position: Token position to analyze
            top_k: Number of top predictions to return

        Returns:
            Dictionary comparing different components
        """
        components = ['resid_pre', 'attn_out', 'resid_mid', 'mlp_out', 'resid_post']
        results = {}

        for component in components:
            try:
                preds = self.get_layer_predictions(
                    text, layers=[layer], positions=[position],
                    component=component, top_k=top_k
                )
                if layer in preds and position in preds[layer]:
                    results[component] = preds[layer][position]
            except Exception as e:
                print(f"Could not analyze component {component}: {e}")

        return results

# %% [markdown]
# ## Visualization Functions

# %%
def plot_layer_predictions(predictions: Dict[int, Dict[str, any]],
                          position: int,
                          top_k: int = 5,
                          title: str = "Top Predictions Across Layers"):
    """
    Visualize top predictions across layers for a specific position.
    """
    layers = sorted(predictions.keys())
    n_layers = len(layers)

    fig, axes = plt.subplots(1, min(n_layers, 6), figsize=(3*min(n_layers, 6), 4))
    if n_layers == 1:
        axes = [axes]

    for i, layer in enumerate(layers[:6]):  # Limit to 6 layers for visualization
        if position in predictions[layer]:
            data = predictions[layer][position]
            tokens = data['top_k_tokens'][:top_k]
            probs = data['top_k_probs'][:top_k]

            # Clean token representation
            clean_tokens = [repr(token) for token in tokens]

            axes[i].barh(range(len(clean_tokens)), probs, alpha=0.7)
            axes[i].set_yticks(range(len(clean_tokens)))
            axes[i].set_yticklabels(clean_tokens)
            axes[i].set_xlabel('Probability')
            axes[i].set_title(f'Layer {layer}')
            axes[i].invert_yaxis()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_token_evolution(token_probs: Dict[int, float],
                        target_token: str,
                        title: str = None):
    """
    Plot how a specific token's probability evolves across layers.
    """
    layers = sorted(token_probs.keys())
    probs = [token_probs[layer] for layer in layers]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, probs, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Layer')
    plt.ylabel('Probability')
    plt.title(title or f"Evolution of '{target_token}' Probability")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(probs) * 1.1)

    # Add probability values as text
    for layer, prob in zip(layers, probs):
        plt.text(layer, prob + max(probs) * 0.02, f'{prob:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_component_comparison(component_results: Dict[str, Dict[str, any]],
                            top_k: int = 5):
    """
    Compare predictions across different model components.
    """
    components = list(component_results.keys())
    n_components = len(components)

    fig, axes = plt.subplots(1, n_components, figsize=(4*n_components, 6))
    if n_components == 1:
        axes = [axes]

    for i, component in enumerate(components):
        data = component_results[component]
        tokens = data['top_k_tokens'][:top_k]
        probs = data['top_k_probs'][:top_k]

        # Clean token representation
        clean_tokens = [repr(token) for token in tokens]

        axes[i].barh(range(len(clean_tokens)), probs, alpha=0.7)
        axes[i].set_yticks(range(len(clean_tokens)))
        axes[i].set_yticklabels(clean_tokens)
        axes[i].set_xlabel('Probability')
        axes[i].set_title(component)
        axes[i].invert_yaxis()

    plt.suptitle('Component Comparison')
    plt.tight_layout()
    plt.show()

def create_prediction_heatmap(predictions: Dict[int, Dict[str, any]],
                            positions: List[int],
                            vocab_subset: List[str],
                            title: str = "Token Probabilities Across Layers"):
    """
    Create a heatmap showing probabilities for specific tokens across layers and positions.
    """
    layers = sorted(predictions.keys())

    # Get token IDs for the vocabulary subset
    tokenizer = None  # Would need access to tokenizer

    # Create matrix: layers x positions x tokens
    data_matrix = []

    for layer in layers:
        layer_probs = []
        for pos in positions:
            if pos in predictions[layer]:
                pos_data = predictions[layer][pos]
                # Extract probabilities for vocab subset
                # This would need proper token ID mapping
                layer_probs.append([0.1] * len(vocab_subset))  # Placeholder
            else:
                layer_probs.append([0.0] * len(vocab_subset))
        data_matrix.append(layer_probs)

    # Note: This is a simplified version - would need proper implementation
    # with access to tokenizer for token ID mapping
    print("Heatmap visualization would show token probabilities across layers and positions")
    print("Implementation requires proper token ID mapping")

# %% [markdown]
# ## Example Usage

# %%
# Initialize logit lens
logit_lens = LogitLens(model_name="gpt2-small", device="cpu")

print(f"Model: {logit_lens.model.cfg.model_name}")
print(f"Number of layers: {logit_lens.model.cfg.n_layers}")
print(f"Vocabulary size: {logit_lens.model.cfg.d_vocab}")

# %% [markdown]
# ## Basic Logit Lens Analysis

# %%
# Analyze a simple sentence
text = "The capital of France is"
print(f"Analyzing: '{text}'")

# Get predictions for all layers
predictions = logit_lens.get_layer_predictions(text, top_k=10)

# Show predictions for the last token position
last_position = len(logit_lens.model.to_tokens(text)[0]) - 1
print(f"\nAnalyzing position {last_position} (last token)")

# Display top predictions for selected layers
selected_layers = [0, 3, 6, 9, 11]  # Sample layers
for layer in selected_layers:
    if layer in predictions and last_position in predictions[layer]:
        data = predictions[layer][last_position]
        print(f"\nLayer {layer} top predictions:")
        for i, (token, prob) in enumerate(zip(data['top_k_tokens'][:5], data['top_k_probs'][:5])):
            print(f"  {i+1}. {repr(token):>8} ({prob:.3f})")

# %% [markdown]
# ## Visualize Predictions Across Layers

# %%
# Plot predictions for selected layers
plot_layer_predictions(predictions, last_position, top_k=5,
                      title=f"Top Predictions for '{text}' (position {last_position})")

# %% [markdown]
# ## Track Specific Token Evolution

# %%
# Track how "Paris" probability evolves
target_token = " Paris"
token_evolution = logit_lens.analyze_token_evolution(text, target_token, last_position)

print(f"Evolution of '{target_token}' probability:")
for layer, prob in sorted(token_evolution.items()):
    print(f"Layer {layer:2d}: {prob:.4f}")

# Plot evolution
plot_token_evolution(token_evolution, target_token,
                    title=f"Evolution of '{target_token}' probability for '{text}'")

# %% [markdown]
# ## Component-wise Analysis

# %%
# Compare different components at layer 6
layer_to_analyze = 6
component_results = logit_lens.compare_components(text, layer_to_analyze, last_position)

print(f"Component analysis for layer {layer_to_analyze}:")
for component, data in component_results.items():
    print(f"\n{component}:")
    for i, (token, prob) in enumerate(zip(data['top_k_tokens'][:3], data['top_k_probs'][:3])):
        print(f"  {i+1}. {repr(token):>8} ({prob:.3f})")

# Visualize component comparison
if component_results:
    plot_component_comparison(component_results, top_k=5)

# %% [markdown]
# ## Advanced Analysis Functions

# %%
def analyze_multiple_examples(logit_lens: LogitLens,
                            examples: List[str],
                            target_tokens: List[str]) -> pd.DataFrame:
    """
    Analyze multiple examples and track target token probabilities.
    """
    results = []

    for text, target in zip(examples, target_tokens):
        token_evolution = logit_lens.analyze_token_evolution(text, target)

        for layer, prob in token_evolution.items():
            results.append({
                'text': text,
                'target_token': target,
                'layer': layer,
                'probability': prob
            })

    return pd.DataFrame(results)

def find_layer_convergence(predictions: Dict[int, Dict[str, any]],
                         position: int,
                         threshold: float = 0.1) -> Optional[int]:
    """
    Find the layer where predictions converge (top prediction probability > threshold).
    """
    layers = sorted(predictions.keys())

    for layer in layers:
        if position in predictions[layer]:
            top_prob = predictions[layer][position]['top_k_probs'][0]
            if top_prob > threshold:
                return layer

    return None

def calculate_prediction_entropy(predictions: Dict[int, Dict[str, any]],
                               position: int) -> Dict[int, float]:
    """
    Calculate prediction entropy across layers to measure uncertainty.
    """
    entropies = {}

    for layer, layer_data in predictions.items():
        if position in layer_data:
            probs = layer_data[position]['probs']
            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies[layer] = entropy

    return entropies

# %% [markdown]
# ## Multi-Example Analysis

# %%
# Analyze multiple factual completion examples
examples = [
    "The capital of France is",
    "The president of the United States is",
    "The largest planet in our solar system is",
    "The author of Romeo and Juliet is"
]

target_tokens = [" Paris", " Joe", " Jupiter", " Shakespeare"]

# Get detailed analysis
multi_results = analyze_multiple_examples(logit_lens, examples, target_tokens)

# Group by example and show max probability achieved
print("Maximum probabilities achieved:")
for (text, target), group in multi_results.groupby(['text', 'target_token']):
    max_prob = group['probability'].max()
    best_layer = group.loc[group['probability'].idxmax(), 'layer']
    print(f"'{text}' -> '{target}': {max_prob:.3f} (layer {best_layer})")

# %% [markdown]
# ## Uncertainty Analysis

# %%
# Analyze prediction uncertainty using entropy
text_to_analyze = "The capital of France is"
predictions = logit_lens.get_layer_predictions(text_to_analyze)
last_pos = len(logit_lens.model.to_tokens(text_to_analyze)[0]) - 1

entropies = calculate_prediction_entropy(predictions, last_pos)

print(f"Prediction entropy across layers for '{text_to_analyze}':")
layers = sorted(entropies.keys())
entropy_values = [entropies[layer] for layer in layers]

plt.figure(figsize=(10, 6))
plt.plot(layers, entropy_values, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Layer')
plt.ylabel('Entropy')
plt.title('Prediction Uncertainty Across Layers')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find convergence layer
convergence_layer = find_layer_convergence(predictions, last_pos, threshold=0.3)
if convergence_layer is not None:
    print(f"Predictions converge at layer {convergence_layer}")
else:
    print("Predictions do not converge strongly")

# %% [markdown]
# ## Conclusion and Next Steps
#
# This logit lens implementation provides insights into how transformer models process
# information across layers. Key observations:
#
# 1. **Progressive Refinement**: Models often refine predictions across layers
# 2. **Component Roles**: Different components (attention vs. MLP) contribute differently
# 3. **Uncertainty Patterns**: Entropy analysis reveals when models become confident
# 4. **Factual Recall**: The technique is particularly useful for analyzing factual knowledge
#
# **Extensions to explore:**
# - Apply to different model architectures and sizes
# - Analyze failure cases and error patterns
# - Combine with attention analysis for deeper insights
# - Investigate the role of specific attention heads in prediction formation
# - Compare with tuned lens for more accurate intermediate predictions