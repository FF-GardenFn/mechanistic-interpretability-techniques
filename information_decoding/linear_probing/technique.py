# %% [markdown]
# # Linear Probing for Mechanistic Interpretability
#
# This notebook implements linear probing techniques to analyze what information
# is linearly accessible in transformer representations across different layers.

# %% [markdown]
# ## Imports and Setup

# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformer_lens import HookedTransformer
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## Linear Probe Class

# %%
class LinearProbe:
    """
    A class for conducting linear probing experiments on transformer representations.

    This class extracts activations from specified layers and trains linear classifiers
    to predict various linguistic or semantic properties.
    """

    def __init__(self, model_name: str = "gpt2-small", device: str = "cpu"):
        """
        Initialize the linear probe with a transformer model.

        Args:
            model_name: Name of the transformer model to load
            device: Device to run computations on
        """
        self.device = device
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.activations = {}
        self.probes = {}

    def extract_activations(self,
                          texts: List[str],
                          layers: List[int] = None,
                          component: str = "resid_post") -> Dict[int, torch.Tensor]:
        """
        Extract activations from specified layers for given texts.

        Args:
            texts: List of input texts
            layers: Layers to extract from (default: all layers)
            component: Component to extract (resid_post, attn_out, mlp_out, etc.)

        Returns:
            Dictionary mapping layer numbers to activation tensors
        """
        if layers is None:
            layers = list(range(self.model.cfg.n_layers))

        activations = {layer: [] for layer in layers}

        # Hook function to capture activations
        def hook_fn(activation, hook):
            layer_num = int(hook.name.split('.')[1])
            if layer_num in layers:
                activations[layer_num].append(activation.detach().cpu())

        # Register hooks
        hook_names = [f"blocks.{layer}.{component}" for layer in layers]
        hooks = []
        for name in hook_names:
            if name in self.model.hook_dict:
                hooks.append(self.model.add_hook(name, hook_fn))

        # Run forward passes
        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting activations"):
                tokens = self.model.to_tokens(text)
                _ = self.model(tokens)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Concatenate activations
        for layer in layers:
            if activations[layer]:
                activations[layer] = torch.cat(activations[layer], dim=0)
            else:
                print(f"Warning: No activations captured for layer {layer}")

        self.activations = activations
        return activations

    def train_probe(self,
                   layer: int,
                   labels: np.ndarray,
                   token_positions: Optional[List[int]] = None,
                   test_size: float = 0.2,
                   standardize: bool = True) -> Dict[str, Any]:
        """
        Train a linear probe on activations from a specific layer.

        Args:
            layer: Layer number to probe
            labels: Target labels for classification
            token_positions: Specific token positions to use (default: last token)
            test_size: Fraction of data to use for testing
            standardize: Whether to standardize features

        Returns:
            Dictionary containing probe results and metrics
        """
        if layer not in self.activations:
            raise ValueError(f"No activations found for layer {layer}")

        # Get activations for this layer
        acts = self.activations[layer]  # [batch, seq_len, hidden_dim]

        # Select token positions
        if token_positions is None:
            # Use last token by default
            X = acts[:, -1, :].numpy()
        else:
            # Average over specified positions
            X = acts[:, token_positions, :].mean(dim=1).numpy()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Standardize features if requested
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            scaler = None

        # Train probe
        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(X_train, y_train)

        # Evaluate
        train_pred = probe.predict(X_train)
        test_pred = probe.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        results = {
            'layer': layer,
            'probe': probe,
            'scaler': scaler,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'y_train': y_train,
            'y_test': y_test,
            'classification_report': classification_report(y_test, test_pred, output_dict=True)
        }

        self.probes[layer] = results
        return results

    def probe_all_layers(self,
                        labels: np.ndarray,
                        layers: Optional[List[int]] = None,
                        **kwargs) -> Dict[int, Dict[str, Any]]:
        """
        Train probes on all specified layers.

        Args:
            labels: Target labels for classification
            layers: Layers to probe (default: all available layers)
            **kwargs: Additional arguments passed to train_probe

        Returns:
            Dictionary mapping layer numbers to probe results
        """
        if layers is None:
            layers = list(self.activations.keys())

        results = {}
        for layer in tqdm(layers, desc="Training probes"):
            try:
                results[layer] = self.train_probe(layer, labels, **kwargs)
            except Exception as e:
                print(f"Error training probe for layer {layer}: {e}")

        return results

# %% [markdown]
# ## Visualization Functions

# %%
def plot_layer_accuracies(probe_results: Dict[int, Dict[str, Any]],
                         title: str = "Probing Accuracies Across Layers"):
    """Plot test accuracies across layers."""
    layers = sorted(probe_results.keys())
    accuracies = [probe_results[layer]['test_accuracy'] for layer in layers]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Layer')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # Add accuracy values as text
    for layer, acc in zip(layers, accuracies):
        plt.text(layer, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(probe_results: Dict[str, Any],
                         class_names: Optional[List[str]] = None):
    """Plot confusion matrix for a specific probe."""
    y_true = probe_results['y_test']
    y_pred = probe_results['test_predictions']

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix - Layer {probe_results['layer']}")
    plt.tight_layout()
    plt.show()

def compare_train_test_accuracies(probe_results: Dict[int, Dict[str, Any]]):
    """Compare training and test accuracies to detect overfitting."""
    layers = sorted(probe_results.keys())
    train_accs = [probe_results[layer]['train_accuracy'] for layer in layers]
    test_accs = [probe_results[layer]['test_accuracy'] for layer in layers]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, train_accs, 'ro-', label='Train Accuracy', alpha=0.7)
    plt.plot(layers, test_accs, 'bo-', label='Test Accuracy', alpha=0.7)
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracies Across Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Example Usage: Part-of-Speech Tagging Probe

# %%
def create_pos_tagging_dataset():
    """
    Create a simple POS tagging dataset for demonstration.
    In practice, you would use a proper dataset like Penn Treebank.
    """
    # Simple examples with clear POS patterns
    sentences = [
        "The cat sits on the mat",
        "A dog runs quickly through the park",
        "She reads books in the library",
        "They play soccer every weekend",
        "The teacher explains the lesson clearly",
        "Birds fly south during winter",
        "He writes code for the project",
        "The flowers bloom in spring",
        "Students study hard for exams",
        "Rain falls gently on the roof"
    ]

    # Focus on specific words and their POS tags
    # For simplicity, we'll classify words as NOUN, VERB, or OTHER
    examples = []
    labels = []

    pos_patterns = {
        'NOUN': ['cat', 'mat', 'dog', 'park', 'books', 'library', 'soccer',
                'weekend', 'teacher', 'lesson', 'birds', 'winter', 'code',
                'project', 'flowers', 'spring', 'students', 'exams', 'rain', 'roof'],
        'VERB': ['sits', 'runs', 'reads', 'play', 'explains', 'fly', 'writes',
                'bloom', 'study', 'falls'],
        'OTHER': ['the', 'on', 'a', 'quickly', 'through', 'she', 'in', 'they',
                 'every', 'clearly', 'south', 'during', 'he', 'for', 'gently']
    }

    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            examples.append(word)
            if word in pos_patterns['NOUN']:
                labels.append(0)  # NOUN
            elif word in pos_patterns['VERB']:
                labels.append(1)  # VERB
            else:
                labels.append(2)  # OTHER

    return examples, np.array(labels), ['NOUN', 'VERB', 'OTHER']

# %%
# Create dataset
texts, labels, class_names = create_pos_tagging_dataset()
print(f"Created dataset with {len(texts)} examples")
print(f"Label distribution: {np.bincount(labels)}")
print(f"Classes: {class_names}")

# Initialize probe
probe = LinearProbe(model_name="gpt2-small", device="cpu")

# Extract activations from multiple layers
layers_to_probe = [0, 3, 6, 9, 11]  # Sample of layers for GPT-2 small
activations = probe.extract_activations(texts, layers=layers_to_probe)

print(f"Extracted activations from layers: {list(activations.keys())}")
for layer, acts in activations.items():
    print(f"Layer {layer}: {acts.shape}")

# %% [markdown]
# ## Train Probes and Analyze Results

# %%
# Train probes on all layers
results = probe.probe_all_layers(labels, layers=layers_to_probe)

# Print results summary
print("Probing Results Summary:")
print("=" * 50)
for layer in sorted(results.keys()):
    result = results[layer]
    print(f"Layer {layer:2d}: Test Accuracy = {result['test_accuracy']:.3f}")

# %% [markdown]
# ## Visualize Results

# %%
# Plot accuracy across layers
plot_layer_accuracies(results, "POS Tagging Accuracy Across Layers")

# %%
# Compare train vs test accuracies
compare_train_test_accuracies(results)

# %%
# Show confusion matrix for best performing layer
best_layer = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
print(f"Best performing layer: {best_layer}")
plot_confusion_matrix(results[best_layer], class_names)

# %% [markdown]
# ## Advanced Analysis Functions

# %%
def analyze_feature_importance(probe_results: Dict[str, Any],
                              feature_names: Optional[List[str]] = None,
                              top_k: int = 10):
    """
    Analyze which features (dimensions) are most important for the probe.
    """
    probe = probe_results['probe']
    if hasattr(probe, 'coef_'):
        # For multiclass, coef_ has shape (n_classes, n_features)
        coef = probe.coef_

        # Calculate feature importance as the L2 norm across classes
        importance = np.linalg.norm(coef, axis=0)

        # Get top features
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_importance = importance[top_indices]

        if feature_names:
            top_features = [feature_names[i] for i in top_indices]
        else:
            top_features = [f"dim_{i}" for i in top_indices]

        return {
            'top_indices': top_indices,
            'top_importance': top_importance,
            'top_features': top_features,
            'all_importance': importance
        }
    else:
        print("Feature importance not available for this probe type")
        return None

def probe_specific_examples(probe_obj: LinearProbe,
                           probe_results: Dict[str, Any],
                           test_texts: List[str],
                           true_labels: List[int],
                           class_names: List[str]):
    """
    Test the probe on specific examples and show predictions.
    """
    layer = probe_results['layer']
    probe_model = probe_results['probe']
    scaler = probe_results['scaler']

    # Extract activations for test texts
    test_activations = probe_obj.extract_activations(test_texts, layers=[layer])
    X_test = test_activations[layer][:, -1, :].numpy()

    if scaler:
        X_test = scaler.transform(X_test)

    # Get predictions and confidence scores
    predictions = probe_model.predict(X_test)
    probabilities = probe_model.predict_proba(X_test)

    print(f"Predictions for Layer {layer}:")
    print("=" * 60)
    for i, (text, true_label, pred_label, probs) in enumerate(
        zip(test_texts, true_labels, predictions, probabilities)
    ):
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        confidence = probs[pred_label]

        status = "✓" if true_label == pred_label else "✗"
        print(f"{status} Text: '{text}'")
        print(f"   True: {true_class}, Predicted: {pred_class} (confidence: {confidence:.3f})")
        print()

# %% [markdown]
# ## Example: Testing on New Examples

# %%
# Test on some new examples
test_examples = ["dog", "running", "quickly", "computer", "thinking"]
test_labels = [0, 1, 2, 0, 1]  # NOUN, VERB, OTHER, NOUN, VERB

probe_specific_examples(
    probe,
    results[best_layer],
    test_examples,
    test_labels,
    class_names
)

# %% [markdown]
# ## Conclusion and Next Steps
#
# This linear probing implementation provides a foundation for analyzing what information
# is linearly accessible in transformer representations. Key insights from this example:
#
# 1. **Layer-wise Analysis**: Different layers may encode different types of information
# 2. **Information Accessibility**: Linear probes reveal what's easily extractable
# 3. **Comparative Analysis**: Train/test gaps indicate overfitting vs. genuine patterns
#
# **Extensions to explore:**
# - Probe for different linguistic phenomena (syntax, semantics, pragmatics)
# - Compare across different model architectures and sizes
# - Use more sophisticated probes (MLP, structural probes)
# - Combine with causal interventions to test if detected information is used
# - Analyze probe weights to understand feature importance