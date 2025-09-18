# %% [markdown]
# # Tuned Lens for Mechanistic Interpretability
#
# This notebook implements the tuned lens technique, which learns layer-specific
# transformations to more accurately interpret what each layer of a transformer
# "knows" by projecting through learned transformations before unembedding.

# %% [markdown]
# ## Imports and Setup

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## Tuned Lens Implementation

# %%
class TunedLensTransformation(nn.Module):
    """
    Learnable transformation for a specific layer in the tuned lens.
    """

    def __init__(self,
                 d_model: int,
                 transformation_type: str = "affine",
                 hidden_dim: Optional[int] = None):
        """
        Initialize a tuned lens transformation.

        Args:
            d_model: Model dimension
            transformation_type: Type of transformation ("affine", "mlp", "scaling")
            hidden_dim: Hidden dimension for MLP (if applicable)
        """
        super().__init__()
        self.d_model = d_model
        self.transformation_type = transformation_type

        if transformation_type == "affine":
            # Simple affine transformation: Wx + b
            self.linear = nn.Linear(d_model, d_model, bias=True)
        elif transformation_type == "mlp":
            # Small MLP transformation
            hidden_dim = hidden_dim or d_model // 2
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, d_model)
            )
        elif transformation_type == "scaling":
            # Just learnable scaling factors
            self.scale = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            raise ValueError(f"Unknown transformation type: {transformation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learned transformation."""
        if self.transformation_type == "affine":
            return self.linear(x)
        elif self.transformation_type == "mlp":
            return self.mlp(x)
        elif self.transformation_type == "scaling":
            return x * self.scale + self.bias

class TunedLens:
    """
    Implementation of the tuned lens technique for improved transformer interpretability.
    """

    def __init__(self,
                 model_name: str = "gpt2-small",
                 device: str = "cpu",
                 transformation_type: str = "affine"):
        """
        Initialize the tuned lens.

        Args:
            model_name: Name of the transformer model to analyze
            device: Device to run computations on
            transformation_type: Type of transformation to learn
        """
        self.device = device
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.W_U = self.model.W_U  # Unembedding matrix
        self.ln_final = self.model.ln_final if hasattr(self.model, 'ln_final') else None
        self.transformation_type = transformation_type

        # Initialize transformations for each layer
        self.transformations = nn.ModuleDict()
        for layer in range(self.model.cfg.n_layers):
            self.transformations[str(layer)] = TunedLensTransformation(
                self.model.cfg.d_model, transformation_type
            ).to(device)

        self.is_trained = False

    def apply_tuned_lens(self,
                        activations: torch.Tensor,
                        layer: int,
                        apply_ln: bool = True) -> torch.Tensor:
        """
        Apply the tuned lens to activations from a specific layer.

        Args:
            activations: Tensor of shape [batch, seq_len, d_model]
            layer: Layer number
            apply_ln: Whether to apply final layer norm

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size]
        """
        # Apply learned transformation
        transformed = self.transformations[str(layer)](activations)

        # Apply final layer normalization if specified
        if apply_ln and self.ln_final is not None:
            transformed = self.ln_final(transformed)

        # Project through unembedding matrix
        logits = transformed @ self.W_U

        return logits

    def collect_training_data(self,
                            texts: List[str],
                            max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Collect activations and targets for training the tuned lens.

        Args:
            texts: List of training texts
            max_length: Maximum sequence length

        Returns:
            Dictionary containing activations and targets
        """
        all_activations = {str(layer): [] for layer in range(self.model.cfg.n_layers)}
        all_targets = []

        print("Collecting training data...")

        for text in tqdm(texts):
            # Tokenize and truncate if necessary
            tokens = self.model.to_tokens(text)
            if tokens.shape[1] > max_length:
                tokens = tokens[:, :max_length]

            # Get targets (next tokens)
            targets = tokens[:, 1:].contiguous()  # [batch, seq_len-1]

            # Hook function to collect activations
            activations_batch = {str(layer): None for layer in range(self.model.cfg.n_layers)}

            def hook_fn(activation, hook):
                layer_num = int(hook.name.split('.')[1])
                activations_batch[str(layer_num)] = activation[:, :-1].detach().cpu()  # [batch, seq_len-1, d_model]

            # Register hooks
            hooks = []
            for layer in range(self.model.cfg.n_layers):
                hook_name = f"blocks.{layer}.hook_resid_post"
                if hook_name in self.model.hook_dict:
                    hooks.append(self.model.add_hook(hook_name, hook_fn))

            # Forward pass
            with torch.no_grad():
                _ = self.model(tokens)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Store data
            for layer in range(self.model.cfg.n_layers):
                if activations_batch[str(layer)] is not None:
                    all_activations[str(layer)].append(activations_batch[str(layer)])

            all_targets.append(targets.cpu())

        # Concatenate all data
        training_data = {}
        for layer in range(self.model.cfg.n_layers):
            if all_activations[str(layer)]:
                training_data[f"activations_{layer}"] = torch.cat(all_activations[str(layer)], dim=0)

        training_data["targets"] = torch.cat(all_targets, dim=0).flatten()

        return training_data

    def train_transformations(self,
                            training_data: Dict[str, torch.Tensor],
                            learning_rate: float = 1e-3,
                            num_epochs: int = 10,
                            batch_size: int = 32,
                            validation_split: float = 0.1) -> Dict[str, List[float]]:
        """
        Train the tuned lens transformations.

        Args:
            training_data: Dictionary containing activations and targets
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary containing training history
        """
        print("Training tuned lens transformations...")

        # Prepare data
        targets = training_data["targets"]
        n_samples = len(targets)

        # Split into train and validation
        indices = torch.randperm(n_samples)
        split_idx = int(n_samples * (1 - validation_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        history = {f"layer_{layer}": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
                  for layer in range(self.model.cfg.n_layers)}

        # Train each layer's transformation separately
        for layer in range(self.model.cfg.n_layers):
            print(f"\nTraining layer {layer}...")

            activations = training_data[f"activations_{layer}"].flatten(0, 1)  # [total_tokens, d_model]
            layer_targets = targets

            # Split data
            train_activations = activations[train_indices].to(self.device)
            train_targets = layer_targets[train_indices].to(self.device)
            val_activations = activations[val_indices].to(self.device)
            val_targets = layer_targets[val_indices].to(self.device)

            # Setup optimizer
            optimizer = optim.Adam(self.transformations[str(layer)].parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            # Training loop
            for epoch in range(num_epochs):
                # Training phase
                self.transformations[str(layer)].train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                # Mini-batch training
                n_batches = (len(train_activations) + batch_size - 1) // batch_size
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(train_activations))

                    batch_activations = train_activations[start_idx:end_idx]
                    batch_targets = train_targets[start_idx:end_idx]

                    # Forward pass
                    optimizer.zero_grad()
                    logits = self.apply_tuned_lens(
                        batch_activations.unsqueeze(1), layer, apply_ln=True
                    ).squeeze(1)

                    loss = criterion(logits, batch_targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    train_total += batch_targets.size(0)
                    train_correct += (predicted == batch_targets).sum().item()

                # Validation phase
                self.transformations[str(layer)].eval()
                with torch.no_grad():
                    val_logits = self.apply_tuned_lens(
                        val_activations.unsqueeze(1), layer, apply_ln=True
                    ).squeeze(1)
                    val_loss = criterion(val_logits, val_targets).item()
                    _, val_predicted = torch.max(val_logits.data, 1)
                    val_correct = (val_predicted == val_targets).sum().item()

                # Record metrics
                train_accuracy = train_correct / train_total
                val_accuracy = val_correct / len(val_targets)

                history[f"layer_{layer}"]["train_loss"].append(train_loss / n_batches)
                history[f"layer_{layer}"]["val_loss"].append(val_loss)
                history[f"layer_{layer}"]["train_acc"].append(train_accuracy)
                history[f"layer_{layer}"]["val_acc"].append(val_accuracy)

                if epoch % 2 == 0 or epoch == num_epochs - 1:
                    print(f"  Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {train_loss/n_batches:.4f}, "
                          f"Train Acc: {train_accuracy:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_accuracy:.4f}")

        self.is_trained = True
        print("\nTraining completed!")
        return history

    def get_tuned_predictions(self,
                            text: str,
                            layers: Optional[List[int]] = None,
                            positions: Optional[List[int]] = None,
                            top_k: int = 10) -> Dict[int, Dict[str, Any]]:
        """
        Get predictions using the tuned lens for specified layers and positions.

        Args:
            text: Input text to analyze
            layers: Layers to analyze (default: all layers)
            positions: Token positions to analyze (default: all positions)
            top_k: Number of top predictions to return

        Returns:
            Dictionary mapping layers to prediction information
        """
        if not self.is_trained:
            raise ValueError("Tuned lens transformations must be trained first!")

        if layers is None:
            layers = list(range(self.model.cfg.n_layers))

        # Tokenize input
        tokens = self.model.to_tokens(text)
        batch_size, seq_len = tokens.shape

        if positions is None:
            positions = list(range(seq_len))

        results = {}

        # Extract activations and apply tuned lens for each layer
        def hook_fn(activation, hook):
            layer_num = int(hook.name.split('.')[1])
            if layer_num in layers:
                # Apply tuned lens
                logits = self.apply_tuned_lens(activation, layer_num)

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
        hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
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

    def compare_with_logit_lens(self,
                              text: str,
                              layer: int,
                              position: int = -1,
                              top_k: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Compare tuned lens predictions with standard logit lens.

        Args:
            text: Input text to analyze
            layer: Layer to compare
            position: Token position to analyze
            top_k: Number of top predictions to show

        Returns:
            Dictionary comparing both methods
        """
        if not self.is_trained:
            raise ValueError("Tuned lens transformations must be trained first!")

        results = {"tuned_lens": None, "logit_lens": None}

        # Get tuned lens predictions
        tuned_preds = self.get_tuned_predictions(text, layers=[layer], positions=[position], top_k=top_k)
        if layer in tuned_preds and position in tuned_preds[layer]:
            results["tuned_lens"] = tuned_preds[layer][position]

        # Get logit lens predictions (without transformation)
        tokens = self.model.to_tokens(text)

        def hook_fn(activation, hook):
            # Apply standard logit lens (no transformation)
            if self.ln_final is not None:
                normalized = self.ln_final(activation)
            else:
                normalized = activation

            logits = normalized @ self.W_U
            probs = F.softmax(logits, dim=-1)

            pos_probs = probs[0, position, :]
            top_k_probs, top_k_indices = torch.topk(pos_probs, top_k)
            top_k_tokens = [self.model.tokenizer.decode([idx]) for idx in top_k_indices]

            results["logit_lens"] = {
                'top_k_probs': top_k_probs.detach().cpu(),
                'top_k_tokens': top_k_tokens
            }

        # Register hook
        hook_name = f"blocks.{layer}.hook_resid_post"
        if hook_name in self.model.hook_dict:
            hook = self.model.add_hook(hook_name, hook_fn)

            with torch.no_grad():
                _ = self.model(tokens)

            hook.remove()

        return results

# %% [markdown]
# ## Visualization Functions

# %%
def plot_training_history(history: Dict[str, Dict[str, List[float]]],
                         layers_to_plot: Optional[List[int]] = None):
    """Plot training history for selected layers."""
    if layers_to_plot is None:
        layers_to_plot = list(range(min(6, len(history))))  # Plot first 6 layers

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Tuned Lens Training History")

    for layer in layers_to_plot:
        layer_key = f"layer_{layer}"
        if layer_key in history:
            layer_history = history[layer_key]

            # Plot losses
            axes[0, 0].plot(layer_history["train_loss"], label=f"Layer {layer}")
            axes[0, 1].plot(layer_history["val_loss"], label=f"Layer {layer}")

            # Plot accuracies
            axes[1, 0].plot(layer_history["train_acc"], label=f"Layer {layer}")
            axes[1, 1].plot(layer_history["val_acc"], label=f"Layer {layer}")

    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    axes[1, 0].set_title("Training Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()

    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

def plot_lens_comparison(comparison_results: Dict[str, Dict[str, Any]],
                        title: str = "Tuned Lens vs Logit Lens"):
    """Compare tuned lens and logit lens predictions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Tuned lens predictions
    if comparison_results["tuned_lens"] is not None:
        tuned_data = comparison_results["tuned_lens"]
        tokens = tuned_data['top_k_tokens']
        probs = tuned_data['top_k_probs']

        clean_tokens = [repr(token) for token in tokens]
        ax1.barh(range(len(clean_tokens)), probs, alpha=0.7, color='blue')
        ax1.set_yticks(range(len(clean_tokens)))
        ax1.set_yticklabels(clean_tokens)
        ax1.set_xlabel('Probability')
        ax1.set_title('Tuned Lens')
        ax1.invert_yaxis()

    # Logit lens predictions
    if comparison_results["logit_lens"] is not None:
        logit_data = comparison_results["logit_lens"]
        tokens = logit_data['top_k_tokens']
        probs = logit_data['top_k_probs']

        clean_tokens = [repr(token) for token in tokens]
        ax2.barh(range(len(clean_tokens)), probs, alpha=0.7, color='red')
        ax2.set_yticks(range(len(clean_tokens)))
        ax2.set_yticklabels(clean_tokens)
        ax2.set_xlabel('Probability')
        ax2.set_title('Logit Lens')
        ax2.invert_yaxis()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_layer_accuracies_comparison(tuned_history: Dict[str, Dict[str, List[float]]]):
    """Plot final validation accuracies across layers."""
    layers = []
    accuracies = []

    for layer_key, history in tuned_history.items():
        layer_num = int(layer_key.split('_')[1])
        final_acc = history["val_acc"][-1] if history["val_acc"] else 0
        layers.append(layer_num)
        accuracies.append(final_acc)

    # Sort by layer number
    sorted_data = sorted(zip(layers, accuracies))
    layers, accuracies = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.plot(layers, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Layer')
    plt.ylabel('Final Validation Accuracy')
    plt.title('Tuned Lens Accuracy Across Layers')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(accuracies) * 1.1)

    # Add accuracy values as text
    for layer, acc in zip(layers, accuracies):
        plt.text(layer, acc + max(accuracies) * 0.02, f'{acc:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Example Usage

# %%
# Initialize tuned lens
tuned_lens = TunedLens(model_name="gpt2-small", device="cpu", transformation_type="affine")

print(f"Model: {tuned_lens.model.cfg.model_name}")
print(f"Number of layers: {tuned_lens.model.cfg.n_layers}")
print(f"Transformation type: {tuned_lens.transformation_type}")

# %% [markdown]
# ## Prepare Training Data

# %%
# Create training dataset
training_texts = [
    "The capital of France is Paris.",
    "The largest planet in our solar system is Jupiter.",
    "The president of the United States is Joe Biden.",
    "The author of Romeo and Juliet is William Shakespeare.",
    "The chemical symbol for gold is Au.",
    "The fastest land animal is the cheetah.",
    "The currency of Japan is the yen.",
    "The capital of Australia is Canberra.",
    "The inventor of the telephone was Alexander Graham Bell.",
    "The smallest prime number is two."
]

print(f"Training on {len(training_texts)} examples")

# Collect training data
training_data = tuned_lens.collect_training_data(training_texts, max_length=128)

print("Training data shapes:")
for key, value in training_data.items():
    if torch.is_tensor(value):
        print(f"  {key}: {value.shape}")

# %% [markdown]
# ## Train Tuned Lens Transformations

# %%
# Train the transformations (using fewer epochs for demo)
training_history = tuned_lens.train_transformations(
    training_data,
    learning_rate=1e-3,
    num_epochs=5,  # Reduced for demo
    batch_size=64,
    validation_split=0.2
)

# %% [markdown]
# ## Visualize Training Results

# %%
# Plot training history for first few layers
plot_training_history(training_history, layers_to_plot=[0, 3, 6, 9, 11])

# %%
# Plot final accuracies across layers
plot_layer_accuracies_comparison(training_history)

# %% [markdown]
# ## Test Tuned Lens Predictions

# %%
# Test on a new example
test_text = "The capital of Germany is"
print(f"Testing on: '{test_text}'")

# Get tuned lens predictions
tuned_predictions = tuned_lens.get_tuned_predictions(test_text, layers=[6, 9, 11], top_k=5)

last_position = len(tuned_lens.model.to_tokens(test_text)[0]) - 1
print(f"Predictions for position {last_position}:")

for layer in sorted(tuned_predictions.keys()):
    if last_position in tuned_predictions[layer]:
        data = tuned_predictions[layer][last_position]
        print(f"\nLayer {layer} (Tuned Lens):")
        for i, (token, prob) in enumerate(zip(data['top_k_tokens'][:5], data['top_k_probs'][:5])):
            print(f"  {i+1}. {repr(token):>10} ({prob:.3f})")

# %% [markdown]
# ## Compare Tuned Lens vs Logit Lens

# %%
# Compare predictions for a specific layer
comparison_layer = 9
comparison = tuned_lens.compare_with_logit_lens(test_text, comparison_layer, last_position)

print(f"Comparison for layer {comparison_layer}:")
print("\nTuned Lens:")
if comparison["tuned_lens"] is not None:
    for i, (token, prob) in enumerate(zip(
        comparison["tuned_lens"]['top_k_tokens'][:5],
        comparison["tuned_lens"]['top_k_probs'][:5]
    )):
        print(f"  {i+1}. {repr(token):>10} ({prob:.3f})")

print("\nLogit Lens:")
if comparison["logit_lens"] is not None:
    for i, (token, prob) in enumerate(zip(
        comparison["logit_lens"]['top_k_tokens'][:5],
        comparison["logit_lens"]['top_k_probs'][:5]
    )):
        print(f"  {i+1}. {repr(token):>10} ({prob:.3f})")

# Visualize comparison
plot_lens_comparison(comparison, f"Layer {comparison_layer} Comparison: '{test_text}'")

# %% [markdown]
# ## Advanced Analysis Functions

# %%
def analyze_transformation_parameters(tuned_lens: TunedLens, layer: int):
    """Analyze the learned transformation parameters."""
    transformation = tuned_lens.transformations[str(layer)]

    if tuned_lens.transformation_type == "affine":
        weight = transformation.linear.weight.data.cpu().numpy()
        bias = transformation.linear.bias.data.cpu().numpy()

        print(f"Layer {layer} transformation analysis:")
        print(f"  Weight matrix shape: {weight.shape}")
        print(f"  Weight matrix norm: {np.linalg.norm(weight):.4f}")
        print(f"  Bias norm: {np.linalg.norm(bias):.4f}")

        # Check if transformation is close to identity
        identity_diff = np.linalg.norm(weight - np.eye(weight.shape[0]))
        print(f"  Distance from identity: {identity_diff:.4f}")

        return {"weight": weight, "bias": bias, "identity_distance": identity_diff}

    elif tuned_lens.transformation_type == "scaling":
        scale = transformation.scale.data.cpu().numpy()
        bias = transformation.bias.data.cpu().numpy()

        print(f"Layer {layer} scaling transformation:")
        print(f"  Scale mean: {np.mean(scale):.4f}")
        print(f"  Scale std: {np.std(scale):.4f}")
        print(f"  Bias mean: {np.mean(bias):.4f}")
        print(f"  Bias std: {np.std(bias):.4f}")

        return {"scale": scale, "bias": bias}

def evaluate_prediction_quality(tuned_lens: TunedLens,
                               test_texts: List[str],
                               layers: List[int]) -> Dict[int, float]:
    """Evaluate prediction quality across layers on test data."""
    layer_accuracies = {layer: [] for layer in layers}

    for text in test_texts:
        tokens = tuned_lens.model.to_tokens(text)
        if tokens.shape[1] > 1:  # Need at least 2 tokens
            # Get true next token
            true_next = tokens[0, -1].item()

            # Get predictions from each layer
            predictions = tuned_lens.get_tuned_predictions(
                text[:-1] if text.endswith('.') else text,  # Remove last char to predict it
                layers=layers,
                top_k=1
            )

            last_pos = len(tuned_lens.model.to_tokens(text[:-1] if text.endswith('.') else text)[0]) - 1

            for layer in layers:
                if layer in predictions and last_pos in predictions[layer]:
                    pred_token_id = predictions[layer][last_pos]['top_k_indices'][0].item()
                    is_correct = pred_token_id == true_next
                    layer_accuracies[layer].append(is_correct)

    # Calculate average accuracies
    avg_accuracies = {}
    for layer in layers:
        if layer_accuracies[layer]:
            avg_accuracies[layer] = np.mean(layer_accuracies[layer])
        else:
            avg_accuracies[layer] = 0.0

    return avg_accuracies

# %% [markdown]
# ## Advanced Analysis

# %%
# Analyze transformation parameters for a few layers
for layer in [3, 6, 9]:
    print()
    analyze_transformation_parameters(tuned_lens, layer)

# %% [markdown]
# ## Evaluate on Test Data

# %%
# Test on new examples
test_examples = [
    "The capital of Spain is Madrid.",
    "The speed of light is approximately 300,000 kilometers per second.",
    "The largest ocean on Earth is the Pacific Ocean."
]

test_accuracies = evaluate_prediction_quality(tuned_lens, test_examples, [3, 6, 9, 11])

print("Test accuracies by layer:")
for layer, accuracy in test_accuracies.items():
    print(f"  Layer {layer}: {accuracy:.3f}")

# %% [markdown]
# ## Conclusion and Next Steps
#
# This tuned lens implementation demonstrates how to learn layer-specific transformations
# to more accurately interpret transformer representations. Key insights:
#
# 1. **Improved Accuracy**: Tuned lens often provides more accurate predictions than raw logit lens
# 2. **Layer Specialization**: Different layers may require different transformations
# 3. **Training Efficiency**: Simple affine transformations are often sufficient
# 4. **Representation Quality**: Shows how well each layer's representations can be interpreted
#
# **Extensions to explore:**
# - Train on larger and more diverse datasets
# - Experiment with different transformation architectures
# - Apply to different model architectures and sizes
# - Combine with other interpretability techniques
# - Use tuned lens for knowledge editing and model debugging
# - Analyze what the learned transformations reveal about internal representations