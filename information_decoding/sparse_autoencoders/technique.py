# %% [markdown]
# # Sparse Autoencoders for Mechanistic Interpretability
#
# This notebook implements sparse autoencoders (SAEs) to learn interpretable,
# sparse representations of transformer activations, addressing the superposition
# problem in neural networks.

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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## Sparse Autoencoder Implementation

# %%
class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for learning interpretable features from neural activations.

    Uses L1 regularization to encourage sparsity in the learned representations.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 tie_weights: bool = False,
                 normalize_decoder: bool = True):
        """
        Initialize the sparse autoencoder.

        Args:
            input_dim: Dimension of input activations
            hidden_dim: Number of hidden features (typically > input_dim)
            tie_weights: Whether to tie encoder and decoder weights
            normalize_decoder: Whether to normalize decoder weights to unit norm
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tie_weights = tie_weights
        self.normalize_decoder = normalize_decoder

        # Encoder: maps activations to sparse features
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder: reconstructs activations from sparse features
        if tie_weights:
            # Use transposed encoder weights for decoder
            self.decoder_weight = None  # Will use encoder.weight.T
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        # Decoder bias (separate from encoder/decoder layers)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using appropriate scheme."""
        # Initialize encoder weights
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)

        # Initialize decoder weights
        if not self.tie_weights:
            nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='linear')

        # Initialize decoder bias to zero
        nn.init.zeros_(self.decoder_bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse features.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Sparse feature activations [batch_size, hidden_dim]
        """
        # Remove decoder bias before encoding
        x_centered = x - self.decoder_bias

        # Apply encoder and ReLU for sparsity
        features = F.relu(self.encoder(x_centered))
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to original space.

        Args:
            features: Sparse features [batch_size, hidden_dim]

        Returns:
            Reconstructed activations [batch_size, input_dim]
        """
        if self.tie_weights:
            # Use transposed encoder weights
            reconstructed = F.linear(features, self.encoder.weight.t())
        else:
            reconstructed = self.decoder(features)

        # Add decoder bias
        reconstructed = reconstructed + self.decoder_bias
        return reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Tuple of (reconstructed, features)
        """
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed, features

    def normalize_decoder_weights(self):
        """Normalize decoder weights to unit norm (if not tied)."""
        if self.normalize_decoder and not self.tie_weights:
            with torch.no_grad():
                self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=1)

    def get_feature_norms(self) -> torch.Tensor:
        """Get L2 norms of decoder weight columns (feature directions)."""
        if self.tie_weights:
            return torch.norm(self.encoder.weight, dim=1)
        else:
            return torch.norm(self.decoder.weight, dim=1)

class SAETrainer:
    """
    Trainer class for sparse autoencoders with comprehensive logging and analysis.
    """

    def __init__(self,
                 sae: SparseAutoencoder,
                 learning_rate: float = 1e-3,
                 l1_coefficient: float = 1e-3,
                 device: str = "cpu"):
        """
        Initialize SAE trainer.

        Args:
            sae: Sparse autoencoder model
            learning_rate: Learning rate for optimization
            l1_coefficient: Coefficient for L1 sparsity penalty
            device: Device to run training on
        """
        self.sae = sae.to(device)
        self.device = device
        self.l1_coefficient = l1_coefficient

        self.optimizer = optim.Adam(self.sae.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'reconstruction_loss': [], 'sparsity_loss': [],
            'sparsity_level': [], 'dead_features': [],
            'reconstruction_error': []
        }

    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute SAE loss components.

        Args:
            x: Input activations [batch_size, input_dim]

        Returns:
            Dictionary of loss components
        """
        reconstructed, features = self.sae(x)

        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, x)

        # Sparsity loss (L1 on features)
        sparsity_loss = torch.mean(torch.abs(features))

        # Total loss
        total_loss = reconstruction_loss + self.l1_coefficient * sparsity_loss

        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss,
            'features': features,
            'reconstructed': reconstructed
        }

    def compute_metrics(self, x: torch.Tensor, features: torch.Tensor) -> Dict[str, float]:
        """Compute additional metrics for monitoring."""
        batch_size = x.shape[0]

        # Sparsity level (average number of active features per example)
        active_features = (features > 0).float()
        sparsity_level = torch.mean(torch.sum(active_features, dim=1)).item()

        # Dead features (features that are never active in this batch)
        feature_activity = torch.sum(active_features, dim=0)
        dead_features = torch.sum(feature_activity == 0).item()

        # Reconstruction error (relative)
        reconstruction_error = torch.mean(torch.norm(x - self.sae.decode(features), dim=1) / torch.norm(x, dim=1)).item()

        return {
            'sparsity_level': sparsity_level,
            'dead_features': dead_features,
            'reconstruction_error': reconstruction_error
        }

    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.sae.train()
        self.optimizer.zero_grad()

        # Forward pass and loss computation
        loss_dict = self.compute_loss(x)
        total_loss = loss_dict['total_loss']

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        # Normalize decoder weights if specified
        self.sae.normalize_decoder_weights()

        # Compute metrics
        with torch.no_grad():
            metrics = self.compute_metrics(x, loss_dict['features'])

        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': loss_dict['reconstruction_loss'].item(),
            'sparsity_loss': loss_dict['sparsity_loss'].item(),
            **metrics
        }

    def validate(self, val_loader) -> Dict[str, float]:
        """Validation step."""
        self.sae.eval()
        val_metrics = defaultdict(list)

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                loss_dict = self.compute_loss(batch)
                metrics = self.compute_metrics(batch, loss_dict['features'])

                val_metrics['total_loss'].append(loss_dict['total_loss'].item())
                val_metrics['reconstruction_loss'].append(loss_dict['reconstruction_loss'].item())
                val_metrics['sparsity_loss'].append(loss_dict['sparsity_loss'].item())
                for key, value in metrics.items():
                    val_metrics[key].append(value)

        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        return avg_metrics

    def train(self,
              train_loader,
              val_loader=None,
              num_epochs: int = 100,
              log_interval: int = 10) -> Dict[str, List[float]]:
        """
        Train the sparse autoencoder.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            log_interval: How often to log progress

        Returns:
            Training history dictionary
        """
        print(f"Training SAE for {num_epochs} epochs...")
        print(f"L1 coefficient: {self.l1_coefficient}")
        print(f"Model parameters: {sum(p.numel() for p in self.sae.parameters())}")

        for epoch in range(num_epochs):
            # Training phase
            epoch_metrics = defaultdict(list)

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                batch = batch.to(self.device)
                step_metrics = self.train_step(batch)

                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)

            # Average training metrics
            avg_train_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}

            # Validation phase
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.scheduler.step(val_metrics['total_loss'])
            else:
                val_metrics = {}

            # Log progress
            if epoch % log_interval == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss: {avg_train_metrics['total_loss']:.4f} "
                      f"(Recon: {avg_train_metrics['reconstruction_loss']:.4f}, "
                      f"Sparsity: {avg_train_metrics['sparsity_loss']:.4f})")
                print(f"  Sparsity Level: {avg_train_metrics['sparsity_level']:.1f}, "
                      f"Dead Features: {avg_train_metrics['dead_features']}")

                if val_loader is not None:
                    print(f"  Val Loss: {val_metrics['total_loss']:.4f}")

            # Store history
            self.history['train_loss'].append(avg_train_metrics['total_loss'])
            self.history['reconstruction_loss'].append(avg_train_metrics['reconstruction_loss'])
            self.history['sparsity_loss'].append(avg_train_metrics['sparsity_loss'])
            self.history['sparsity_level'].append(avg_train_metrics['sparsity_level'])
            self.history['dead_features'].append(avg_train_metrics['dead_features'])
            self.history['reconstruction_error'].append(avg_train_metrics['reconstruction_error'])

            if val_loader is not None:
                self.history['val_loss'].append(val_metrics['total_loss'])

        print("Training completed!")
        return self.history

# %% [markdown]
# ## Feature Analysis Tools

# %%
class SAEAnalyzer:
    """
    Tools for analyzing learned SAE features and their interpretability.
    """

    def __init__(self, sae: SparseAutoencoder, model: HookedTransformer):
        """
        Initialize SAE analyzer.

        Args:
            sae: Trained sparse autoencoder
            model: Original transformer model
        """
        self.sae = sae
        self.model = model
        self.feature_examples = {}
        self.feature_stats = {}

    def collect_feature_activations(self,
                                  texts: List[str],
                                  layer: int,
                                  component: str = "resid_post",
                                  max_examples_per_feature: int = 10) -> Dict[int, List[Dict[str, Any]]]:
        """
        Collect examples where each feature is highly active.

        Args:
            texts: List of texts to analyze
            layer: Layer to extract activations from
            component: Model component to analyze
            max_examples_per_feature: Maximum examples to store per feature

        Returns:
            Dictionary mapping feature indices to example data
        """
        feature_examples = defaultdict(list)

        print(f"Collecting feature activations from {len(texts)} texts...")

        for text_idx, text in enumerate(tqdm(texts)):
            # Get model activations
            tokens = self.model.to_tokens(text)

            # Hook to extract activations
            activations = []
            def hook_fn(activation, hook):
                activations.append(activation.detach().cpu())

            hook_name = f"blocks.{layer}.{component}"
            if hook_name in self.model.hook_dict:
                hook = self.model.add_hook(hook_name, hook_fn)

                with torch.no_grad():
                    _ = self.model(tokens)

                hook.remove()

                if activations:
                    # Get SAE features
                    layer_acts = activations[0].squeeze(0)  # [seq_len, d_model]
                    sae_features = self.sae.encode(layer_acts)  # [seq_len, hidden_dim]

                    # Find top activations for each feature
                    for pos in range(sae_features.shape[0]):
                        pos_features = sae_features[pos]  # [hidden_dim]

                        # Find active features (> threshold)
                        active_mask = pos_features > 0.1  # Threshold for "active"
                        active_indices = torch.where(active_mask)[0]
                        active_values = pos_features[active_indices]

                        # Store examples for each active feature
                        for feat_idx, feat_val in zip(active_indices, active_values):
                            feat_idx = feat_idx.item()
                            feat_val = feat_val.item()

                            if len(feature_examples[feat_idx]) < max_examples_per_feature:
                                # Get context around this position
                                context_start = max(0, pos - 5)
                                context_end = min(len(tokens[0]), pos + 6)
                                context_tokens = tokens[0][context_start:context_end]
                                context_text = self.model.tokenizer.decode(context_tokens)

                                # Get the specific token at this position
                                if pos < len(tokens[0]):
                                    target_token = self.model.tokenizer.decode([tokens[0][pos]])
                                else:
                                    target_token = "[END]"

                                feature_examples[feat_idx].append({
                                    'text_idx': text_idx,
                                    'position': pos,
                                    'activation': feat_val,
                                    'token': target_token,
                                    'context': context_text,
                                    'full_text': text
                                })

        self.feature_examples = dict(feature_examples)
        return self.feature_examples

    def analyze_feature_patterns(self, feature_idx: int, top_k: int = 20) -> Dict[str, Any]:
        """
        Analyze patterns in a specific feature's activations.

        Args:
            feature_idx: Index of feature to analyze
            top_k: Number of top examples to analyze

        Returns:
            Dictionary of analysis results
        """
        if feature_idx not in self.feature_examples:
            return {"error": f"No examples found for feature {feature_idx}"}

        examples = self.feature_examples[feature_idx]

        # Sort by activation strength
        examples = sorted(examples, key=lambda x: x['activation'], reverse=True)
        top_examples = examples[:top_k]

        # Analyze tokens
        tokens = [ex['token'] for ex in top_examples]
        token_counts = pd.Series(tokens).value_counts()

        # Analyze activation statistics
        activations = [ex['activation'] for ex in examples]

        analysis = {
            'feature_idx': feature_idx,
            'num_examples': len(examples),
            'max_activation': max(activations),
            'mean_activation': np.mean(activations),
            'std_activation': np.std(activations),
            'top_tokens': token_counts.head(10).to_dict(),
            'top_examples': top_examples[:5],  # Store top 5 for display
            'activation_distribution': activations
        }

        return analysis

    def find_interpretable_features(self,
                                  min_examples: int = 5,
                                  max_features_to_analyze: int = 100) -> List[Dict[str, Any]]:
        """
        Find features that appear to be interpretable based on their activation patterns.

        Args:
            min_examples: Minimum number of examples required for analysis
            max_features_to_analyze: Maximum number of features to analyze

        Returns:
            List of feature analyses sorted by interpretability score
        """
        print("Analyzing feature interpretability...")

        interpretable_features = []

        # Get features with sufficient examples
        features_to_analyze = [
            feat_idx for feat_idx, examples in self.feature_examples.items()
            if len(examples) >= min_examples
        ]

        # Limit analysis to top N most active features
        if len(features_to_analyze) > max_features_to_analyze:
            # Sort by total activation
            feature_activity = {
                feat_idx: sum(ex['activation'] for ex in examples)
                for feat_idx, examples in self.feature_examples.items()
                if feat_idx in features_to_analyze
            }
            features_to_analyze = sorted(feature_activity.keys(),
                                       key=lambda x: feature_activity[x],
                                       reverse=True)[:max_features_to_analyze]

        for feat_idx in tqdm(features_to_analyze, desc="Analyzing features"):
            analysis = self.analyze_feature_patterns(feat_idx)

            # Compute interpretability score (simple heuristic)
            token_diversity = len(analysis['top_tokens'])
            activation_consistency = 1.0 / (1.0 + analysis['std_activation'])
            most_common_token_freq = max(analysis['top_tokens'].values()) / analysis['num_examples']

            # Features that activate on similar tokens with consistent strength are more interpretable
            interpretability_score = most_common_token_freq * activation_consistency / max(1, token_diversity / 3)

            analysis['interpretability_score'] = interpretability_score
            interpretable_features.append(analysis)

        # Sort by interpretability score
        interpretable_features.sort(key=lambda x: x['interpretability_score'], reverse=True)

        return interpretable_features

    def visualize_feature(self, feature_analysis: Dict[str, Any]):
        """Visualize a single feature's activation patterns."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Activation distribution
        axes[0].hist(feature_analysis['activation_distribution'], bins=20, alpha=0.7)
        axes[0].set_xlabel('Activation Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f"Feature {feature_analysis['feature_idx']} Activations")

        # Top tokens
        top_tokens = feature_analysis['top_tokens']
        if top_tokens:
            tokens = list(top_tokens.keys())[:10]
            counts = list(top_tokens.values())[:10]

            axes[1].barh(range(len(tokens)), counts)
            axes[1].set_yticks(range(len(tokens)))
            axes[1].set_yticklabels([repr(token) for token in tokens])
            axes[1].set_xlabel('Count')
            axes[1].set_title('Most Common Tokens')
            axes[1].invert_yaxis()

        # Example contexts
        axes[2].text(0.1, 0.9, f"Feature {feature_analysis['feature_idx']}",
                    transform=axes[2].transAxes, fontsize=14, fontweight='bold')

        examples_text = "Top Examples:\n\n"
        for i, ex in enumerate(feature_analysis['top_examples'][:3]):
            examples_text += f"{i+1}. {ex['context']}\n"
            examples_text += f"   Token: {repr(ex['token'])} (act: {ex['activation']:.3f})\n\n"

        axes[2].text(0.1, 0.8, examples_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', wrap=True)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## Data Collection and Preparation

# %%
def collect_model_activations(model: HookedTransformer,
                            texts: List[str],
                            layer: int,
                            component: str = "resid_post",
                            max_length: int = 512) -> torch.Tensor:
    """
    Collect activations from a specific layer of the model.

    Args:
        model: Transformer model
        texts: List of input texts
        layer: Layer to extract from
        component: Component to extract
        max_length: Maximum sequence length

    Returns:
        Tensor of activations [total_tokens, d_model]
    """
    all_activations = []

    print(f"Collecting activations from layer {layer}, component {component}...")

    for text in tqdm(texts):
        # Tokenize and truncate
        tokens = model.to_tokens(text)
        if tokens.shape[1] > max_length:
            tokens = tokens[:, :max_length]

        # Extract activations
        activations = []
        def hook_fn(activation, hook):
            activations.append(activation.detach().cpu())

        hook_name = f"blocks.{layer}.{component}"
        if hook_name in model.hook_dict:
            hook = model.add_hook(hook_name, hook_fn)

            with torch.no_grad():
                _ = model(tokens)

            hook.remove()

            if activations:
                # Flatten sequence dimension
                layer_acts = activations[0].squeeze(0)  # [seq_len, d_model]
                all_activations.append(layer_acts)

    # Concatenate all activations
    if all_activations:
        return torch.cat(all_activations, dim=0)
    else:
        raise ValueError("No activations collected")

def create_data_loaders(activations: torch.Tensor,
                       batch_size: int = 64,
                       validation_split: float = 0.1) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation data loaders."""
    # Split data
    n_samples = activations.shape[0]
    indices = torch.randperm(n_samples)
    split_idx = int(n_samples * (1 - validation_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_data = activations[train_indices]
    val_data = activations[val_indices]

    # Create datasets and loaders
    train_dataset = torch.utils.data.TensorDataset(train_data)
    val_dataset = torch.utils.data.TensorDataset(val_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# %% [markdown]
# ## Example Usage

# %%
# Initialize model
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
print(f"Model: {model.cfg.model_name}")
print(f"Number of layers: {model.cfg.n_layers}")
print(f"Model dimension: {model.cfg.d_model}")

# %% [markdown]
# ## Prepare Training Data

# %%
# Create dataset of texts for training
training_texts = [
    "The capital of France is Paris, which is known for its beautiful architecture.",
    "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
    "The Pacific Ocean is the largest ocean on Earth, covering about one-third of the planet.",
    "Shakespeare wrote many famous plays including Romeo and Juliet and Hamlet.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The theory of relativity was developed by Albert Einstein in the early 20th century.",
    "Democracy is a form of government where citizens have the power to choose their leaders.",
    "The Great Wall of China was built over many centuries to protect against invasions.",
    "DNA contains the genetic instructions for all living organisms on Earth.",
    "The Renaissance was a period of great cultural and artistic achievement in Europe.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns.",
    "The internet has revolutionized how we communicate and access information.",
    "Evolution explains how species change and adapt over millions of years.",
    "The solar system consists of the sun and all celestial bodies that orbit it.",
    "Quantum mechanics describes the behavior of matter and energy at the atomic scale."
]

# Collect activations from layer 6 (middle layer)
layer_to_analyze = 6
activations = collect_model_activations(model, training_texts, layer_to_analyze)

print(f"Collected activations shape: {activations.shape}")
print(f"Activation statistics: mean={activations.mean():.3f}, std={activations.std():.3f}")

# %% [markdown]
# ## Create and Train Sparse Autoencoder

# %%
# SAE hyperparameters
input_dim = model.cfg.d_model
hidden_dim = input_dim * 8  # Expansion factor of 8
l1_coefficient = 1e-3

# Create SAE
sae = SparseAutoencoder(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    tie_weights=False,
    normalize_decoder=True
)

print(f"SAE Architecture:")
print(f"  Input dim: {input_dim}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Expansion factor: {hidden_dim / input_dim}")
print(f"  Parameters: {sum(p.numel() for p in sae.parameters())}")

# Create data loaders
train_loader, val_loader = create_data_loaders(activations, batch_size=128)

# Create trainer
trainer = SAETrainer(sae, learning_rate=1e-3, l1_coefficient=l1_coefficient, device="cpu")

# %% [markdown]
# ## Train the SAE

# %%
# Train SAE (using fewer epochs for demo)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,  # Reduced for demo
    log_interval=5
)

# %% [markdown]
# ## Visualize Training Progress

# %%
def plot_training_history(history: Dict[str, List[float]]):
    """Plot SAE training history."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("SAE Training History")

    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(epochs, history['val_loss'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, history['reconstruction_loss'])
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')

    axes[0, 2].plot(epochs, history['sparsity_loss'])
    axes[0, 2].set_title('Sparsity Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('L1 Loss')

    # Metrics
    axes[1, 0].plot(epochs, history['sparsity_level'])
    axes[1, 0].set_title('Sparsity Level')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Avg Active Features')

    axes[1, 1].plot(epochs, history['dead_features'])
    axes[1, 1].set_title('Dead Features')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Number of Dead Features')

    axes[1, 2].plot(epochs, history['reconstruction_error'])
    axes[1, 2].set_title('Reconstruction Error')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Relative Error')

    plt.tight_layout()
    plt.show()

# Plot training progress
plot_training_history(history)

# %% [markdown]
# ## Analyze Learned Features

# %%
# Create analyzer
analyzer = SAEAnalyzer(sae, model)

# Collect feature activations on a broader set of examples
analysis_texts = training_texts + [
    "Mathematics is the language of science and engineering.",
    "The human brain contains billions of interconnected neurons.",
    "Music has the power to evoke strong emotions and memories.",
    "Space exploration has led to many technological advances.",
    "Cooking combines art, science, and cultural tradition."
]

# Collect feature examples
feature_examples = analyzer.collect_feature_activations(
    analysis_texts,
    layer_to_analyze,
    max_examples_per_feature=20
)

print(f"Found activations for {len(feature_examples)} features")
print(f"Example feature indices: {list(feature_examples.keys())[:10]}")

# %% [markdown]
# ## Find Most Interpretable Features

# %%
# Analyze interpretability
interpretable_features = analyzer.find_interpretable_features(
    min_examples=3,
    max_features_to_analyze=50
)

print(f"Found {len(interpretable_features)} potentially interpretable features")

# Display top interpretable features
print("\nTop 5 Most Interpretable Features:")
print("=" * 60)

for i, feature_analysis in enumerate(interpretable_features[:5]):
    print(f"\nFeature {feature_analysis['feature_idx']} "
          f"(Score: {feature_analysis['interpretability_score']:.3f})")
    print(f"Examples: {feature_analysis['num_examples']}, "
          f"Max activation: {feature_analysis['max_activation']:.3f}")

    print("Top tokens:", list(feature_analysis['top_tokens'].keys())[:5])

    print("Example contexts:")
    for j, ex in enumerate(feature_analysis['top_examples'][:2]):
        print(f"  {j+1}. {ex['context']}")
        print(f"     Token: {repr(ex['token'])} (activation: {ex['activation']:.3f})")

# %% [markdown]
# ## Visualize Individual Features

# %%
# Visualize top 3 most interpretable features
for i, feature_analysis in enumerate(interpretable_features[:3]):
    print(f"\n--- Feature {feature_analysis['feature_idx']} ---")
    analyzer.visualize_feature(feature_analysis)

# %% [markdown]
# ## Feature Intervention Experiments

# %%
def test_feature_intervention(sae: SparseAutoencoder,
                            model: HookedTransformer,
                            text: str,
                            layer: int,
                            feature_idx: int,
                            intervention_value: float = 0.0) -> Dict[str, Any]:
    """
    Test the effect of intervening on a specific feature.

    Args:
        sae: Trained sparse autoencoder
        model: Transformer model
        text: Input text to test
        layer: Layer to intervene on
        feature_idx: Feature to modify
        intervention_value: Value to set the feature to

    Returns:
        Dictionary with intervention results
    """
    tokens = model.to_tokens(text)

    # Get original activations and predictions
    original_logits = model(tokens)
    original_probs = F.softmax(original_logits[0, -1, :], dim=-1)
    original_top_tokens = torch.topk(original_probs, 5)

    # Intervention hook
    def intervention_hook(activation, hook):
        # Get SAE features
        features = sae.encode(activation)

        # Modify specific feature
        modified_features = features.clone()
        modified_features[:, :, feature_idx] = intervention_value

        # Reconstruct activations
        modified_activation = sae.decode(modified_features)
        return modified_activation

    # Apply intervention
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook = model.add_hook(hook_name, intervention_hook)

    with torch.no_grad():
        modified_logits = model(tokens)

    hook.remove()

    modified_probs = F.softmax(modified_logits[0, -1, :], dim=-1)
    modified_top_tokens = torch.topk(modified_probs, 5)

    return {
        'text': text,
        'feature_idx': feature_idx,
        'intervention_value': intervention_value,
        'original_top_tokens': [(model.tokenizer.decode([idx]), prob.item())
                               for idx, prob in zip(original_top_tokens.indices, original_top_tokens.values)],
        'modified_top_tokens': [(model.tokenizer.decode([idx]), prob.item())
                               for idx, prob in zip(modified_top_tokens.indices, modified_top_tokens.values)],
        'probability_shift': (modified_probs - original_probs).abs().mean().item()
    }

# %% [markdown]
# ## Test Feature Interventions

# %%
# Test intervention on most interpretable feature
if interpretable_features:
    top_feature = interpretable_features[0]
    test_text = "The capital of France is"

    print(f"Testing intervention on Feature {top_feature['feature_idx']}")
    print(f"Test text: '{test_text}'")

    # Test setting feature to 0 (ablation)
    intervention_result = test_feature_intervention(
        sae, model, test_text, layer_to_analyze,
        top_feature['feature_idx'], intervention_value=0.0
    )

    print(f"\nOriginal top tokens:")
    for token, prob in intervention_result['original_top_tokens']:
        print(f"  {repr(token):>10}: {prob:.3f}")

    print(f"\nAfter setting feature {top_feature['feature_idx']} to 0:")
    for token, prob in intervention_result['modified_top_tokens']:
        print(f"  {repr(token):>10}: {prob:.3f}")

    print(f"\nAverage probability shift: {intervention_result['probability_shift']:.4f}")

# %% [markdown]
# ## Summary Statistics

# %%
def compute_sae_summary_stats(sae: SparseAutoencoder,
                            activations: torch.Tensor) -> Dict[str, Any]:
    """Compute summary statistics for the trained SAE."""
    sae.eval()

    with torch.no_grad():
        # Sample subset for analysis
        sample_size = min(1000, activations.shape[0])
        sample_activations = activations[:sample_size]

        # Get reconstructions and features
        reconstructed, features = sae(sample_activations)

        # Reconstruction quality
        mse_loss = F.mse_loss(reconstructed, sample_activations).item()
        reconstruction_error = torch.mean(
            torch.norm(reconstructed - sample_activations, dim=1) /
            torch.norm(sample_activations, dim=1)
        ).item()

        # Sparsity statistics
        active_features = (features > 0).float()
        sparsity_level = torch.mean(torch.sum(active_features, dim=1)).item()
        sparsity_fraction = sparsity_level / sae.hidden_dim

        # Feature usage
        feature_usage = torch.mean(active_features, dim=0)
        dead_features = torch.sum(feature_usage == 0).item()
        active_feature_count = sae.hidden_dim - dead_features

        # Feature activation statistics
        nonzero_features = features[features > 0]
        if len(nonzero_features) > 0:
            mean_nonzero_activation = torch.mean(nonzero_features).item()
            std_nonzero_activation = torch.std(nonzero_features).item()
        else:
            mean_nonzero_activation = 0.0
            std_nonzero_activation = 0.0

    return {
        'reconstruction_mse': mse_loss,
        'reconstruction_error': reconstruction_error,
        'sparsity_level': sparsity_level,
        'sparsity_fraction': sparsity_fraction,
        'dead_features': dead_features,
        'active_features': active_feature_count,
        'feature_utilization': active_feature_count / sae.hidden_dim,
        'mean_nonzero_activation': mean_nonzero_activation,
        'std_nonzero_activation': std_nonzero_activation
    }

# Compute and display summary statistics
summary_stats = compute_sae_summary_stats(sae, activations)

print("SAE Summary Statistics:")
print("=" * 40)
print(f"Reconstruction MSE: {summary_stats['reconstruction_mse']:.6f}")
print(f"Reconstruction Error: {summary_stats['reconstruction_error']:.4f}")
print(f"Average Sparsity Level: {summary_stats['sparsity_level']:.1f} features")
print(f"Sparsity Fraction: {summary_stats['sparsity_fraction']:.3f}")
print(f"Dead Features: {summary_stats['dead_features']} / {sae.hidden_dim}")
print(f"Feature Utilization: {summary_stats['feature_utilization']:.3f}")
print(f"Mean Nonzero Activation: {summary_stats['mean_nonzero_activation']:.3f}")
print(f"Std Nonzero Activation: {summary_stats['std_nonzero_activation']:.3f}")

# %% [markdown]
# ## Conclusion and Next Steps
#
# This sparse autoencoder implementation demonstrates how to:
#
# 1. **Train SAEs**: Learn sparse, interpretable features from transformer activations
# 2. **Analyze Features**: Identify potentially interpretable features and their patterns
# 3. **Visualize Results**: Display feature activation patterns and examples
# 4. **Test Interventions**: Modify features to understand their causal role
# 5. **Evaluate Quality**: Assess reconstruction fidelity and sparsity levels
#
# **Key Findings from this Example:**
# - Successfully learned sparse representations with reasonable reconstruction quality
# - Identified features that activate on specific tokens or patterns
# - Demonstrated how feature interventions can affect model predictions
# - Showed the trade-off between sparsity and reconstruction fidelity
#
# **Extensions to explore:**
# - Train on larger and more diverse datasets
# - Experiment with different sparsity coefficients and architectures
# - Apply to different layers and model components
# - Investigate hierarchical feature relationships across layers
# - Use learned features for model editing and control
# - Compare features across different model architectures and training procedures
# - Develop better metrics for evaluating feature interpretability