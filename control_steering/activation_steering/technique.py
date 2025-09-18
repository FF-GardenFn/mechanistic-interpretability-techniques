"""
Activation Steering Implementation
Techniques for steering language model behavior through activation manipulation
"""

#%% Imports and Setup
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Configuration and Data Classes
@dataclass
class SteeringConfig:
    """Configuration for activation steering experiments"""
    model_name: str = "gpt2-small"
    layer_idx: int = 6  # Which layer to intervene at
    position: str = "last"  # Position to intervene: "last", "all", or specific index
    steering_strength: float = 1.0
    max_tokens: int = 50

@dataclass
class ConceptVector:
    """Represents a concept vector for steering"""
    name: str
    vector: torch.Tensor
    layer: int
    description: str = ""

#%% Core Activation Steering Class
class ActivationSteering:
    """
    Main class for implementing activation steering techniques
    """

    def __init__(self, config: SteeringConfig):
        self.config = config
        self.model = HookedTransformer.from_pretrained(config.model_name)
        self.model.to(device)
        self.concept_vectors = {}

    def extract_concept_vector(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        concept_name: str,
        layer_idx: Optional[int] = None
    ) -> ConceptVector:
        """
        Extract a concept vector by contrasting positive and negative examples

        Args:
            positive_prompts: Examples that exhibit the concept
            negative_prompts: Examples that don't exhibit the concept
            concept_name: Name for the concept
            layer_idx: Layer to extract from (uses config default if None)

        Returns:
            ConceptVector object
        """
        layer_idx = layer_idx or self.config.layer_idx
        act_name = get_act_name("resid_post", layer_idx)

        # Get activations for positive examples
        pos_acts = []
        for prompt in positive_prompts:
            tokens = self.model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)
                if self.config.position == "last":
                    act = cache[act_name][0, -1, :]  # Last token position
                else:
                    act = cache[act_name][0, :, :].mean(dim=0)  # Average over positions
                pos_acts.append(act)

        # Get activations for negative examples
        neg_acts = []
        for prompt in negative_prompts:
            tokens = self.model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)
                if self.config.position == "last":
                    act = cache[act_name][0, -1, :]
                else:
                    act = cache[act_name][0, :, :].mean(dim=0)
                neg_acts.append(act)

        # Compute concept vector as difference of means
        pos_mean = torch.stack(pos_acts).mean(dim=0)
        neg_mean = torch.stack(neg_acts).mean(dim=0)
        concept_vector = pos_mean - neg_mean

        # Normalize the vector
        concept_vector = F.normalize(concept_vector, dim=0)

        concept = ConceptVector(
            name=concept_name,
            vector=concept_vector,
            layer=layer_idx,
            description=f"Concept vector for {concept_name} at layer {layer_idx}"
        )

        self.concept_vectors[concept_name] = concept
        return concept

    def steering_hook(self, activations: torch.Tensor, hook, concept_vector: torch.Tensor, strength: float):
        """
        Hook function to modify activations during forward pass
        """
        if self.config.position == "last":
            activations[:, -1, :] += strength * concept_vector
        else:
            activations += strength * concept_vector.unsqueeze(0).unsqueeze(0)
        return activations

    def generate_with_steering(
        self,
        prompt: str,
        concept_name: str,
        strength: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate text with activation steering applied

        Args:
            prompt: Input prompt
            concept_name: Name of concept to steer with
            strength: Steering strength (uses config default if None)
            max_tokens: Max tokens to generate (uses config default if None)

        Returns:
            Generated text
        """
        if concept_name not in self.concept_vectors:
            raise ValueError(f"Concept '{concept_name}' not found. Available: {list(self.concept_vectors.keys())}")

        concept = self.concept_vectors[concept_name]
        strength = strength or self.config.steering_strength
        max_tokens = max_tokens or self.config.max_tokens

        # Create hook
        act_name = get_act_name("resid_post", concept.layer)
        hook_fn = lambda activations, hook: self.steering_hook(
            activations, hook, concept.vector, strength
        )

        # Generate with steering
        tokens = self.model.to_tokens(prompt)
        with self.model.hooks([(act_name, hook_fn)]):
            output = self.model.generate(
                tokens,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        return self.model.to_string(output[0])

#%% Example Usage and Experiments
def run_sentiment_steering_example():
    """Example: Steering sentiment from negative to positive"""

    # Initialize steering system
    config = SteeringConfig(model_name="gpt2-small", layer_idx=6)
    steerer = ActivationSteering(config)

    # Define example prompts for concept extraction
    positive_prompts = [
        "I love this movie, it's absolutely fantastic!",
        "What a wonderful day, everything is going perfectly!",
        "This is the best experience I've ever had!",
        "I'm so happy and grateful for this opportunity!"
    ]

    negative_prompts = [
        "I hate this movie, it's terrible and boring.",
        "This is the worst day ever, nothing is going right.",
        "I'm so disappointed and frustrated with this.",
        "This experience was awful and completely ruined my mood."
    ]

    # Extract sentiment concept vector
    concept = steerer.extract_concept_vector(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        concept_name="positive_sentiment"
    )

    print(f"Extracted concept vector: {concept.name}")
    print(f"Vector shape: {concept.vector.shape}")
    print(f"Vector norm: {concept.vector.norm().item():.4f}")

    # Test steering on neutral prompts
    test_prompts = [
        "The weather today is",
        "I think about the future and",
        "When I look at this situation, I feel",
        "My opinion about this topic is"
    ]

    print("\n" + "="*50)
    print("STEERING RESULTS")
    print("="*50)

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")

        # Generate without steering
        original = steerer.model.generate(
            steerer.model.to_tokens(prompt),
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7
        )
        print(f"Original: {steerer.model.to_string(original[0])}")

        # Generate with positive steering
        steered = steerer.generate_with_steering(
            prompt, "positive_sentiment", strength=2.0, max_tokens=30
        )
        print(f"Steered:  {steered}")

#%% Visualization Functions
def visualize_concept_vector(concept: ConceptVector, top_k: int = 20):
    """Visualize the strongest components of a concept vector"""

    vector = concept.vector.cpu().numpy()
    indices = np.argsort(np.abs(vector))[-top_k:]
    values = vector[indices]

    plt.figure(figsize=(12, 6))
    colors = ['red' if v < 0 else 'blue' for v in values]
    plt.barh(range(len(values)), values, color=colors)
    plt.xlabel('Activation Value')
    plt.ylabel('Neuron Index')
    plt.title(f'Top {top_k} Components of {concept.name} Vector')
    plt.grid(True, alpha=0.3)

    # Add legend
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Negative')
    blue_patch = mpatches.Patch(color='blue', label='Positive')
    plt.legend(handles=[red_patch, blue_patch])

    plt.tight_layout()
    plt.show()

def analyze_steering_strength_effects(steerer: ActivationSteering, prompt: str, concept_name: str):
    """Analyze how different steering strengths affect generation"""

    strengths = [-3, -2, -1, 0, 1, 2, 3]
    results = []

    print(f"Analyzing steering strength effects for prompt: '{prompt}'")
    print("-" * 60)

    for strength in strengths:
        if strength == 0:
            # Generate without steering
            tokens = steerer.model.to_tokens(prompt)
            output = steerer.model.generate(
                tokens, max_new_tokens=20, do_sample=False
            )
            generated = steerer.model.to_string(output[0])
        else:
            generated = steerer.generate_with_steering(
                prompt, concept_name, strength=strength, max_tokens=20
            )

        results.append({
            'strength': strength,
            'generated': generated,
            'length': len(generated.split())
        })

        print(f"Strength {strength:2}: {generated}")

    return results

#%% Layer Analysis Functions
def analyze_steering_across_layers(
    steerer: ActivationSteering,
    positive_prompts: List[str],
    negative_prompts: List[str],
    concept_name: str,
    test_prompt: str
):
    """Analyze steering effectiveness across different layers"""

    n_layers = steerer.model.cfg.n_layers
    results = []

    for layer in range(n_layers):
        # Extract concept vector for this layer
        concept = steerer.extract_concept_vector(
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            concept_name=f"{concept_name}_layer_{layer}",
            layer_idx=layer
        )

        # Test steering
        steered_text = steerer.generate_with_steering(
            test_prompt, f"{concept_name}_layer_{layer}", strength=2.0, max_tokens=20
        )

        results.append({
            'layer': layer,
            'concept_name': concept.name,
            'generated': steered_text,
            'vector_norm': concept.vector.norm().item()
        })

        print(f"Layer {layer:2}: {steered_text}")

    return results

#%% Main Execution
if __name__ == "__main__":
    print("Running Activation Steering Example...")
    run_sentiment_steering_example()

    print("\n" + "="*50)
    print("For more advanced analysis, run:")
    print("1. visualize_concept_vector(concept)")
    print("2. analyze_steering_strength_effects(steerer, prompt, concept_name)")
    print("3. analyze_steering_across_layers(...)")
    print("="*50)