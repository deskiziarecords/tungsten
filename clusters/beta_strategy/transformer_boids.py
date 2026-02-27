import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

class BoidAttention(nn.Module):
    """
    Relational Attention for Swarm Intelligence.
    Each Boid attends to others to calculate its next 'Social Force'.
    """
    embed_dim: int = 128
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None):
        # x: (batch_size, num_boids, features)
        # Self-attention allows every boid to 'sense' the global state
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads
        )(x, x, mask=mask)
        
        # Residual connection + LayerNorm for stability (NRO #48)
        return nn.LayerNorm()(x + attn_out)

class TransformerBoidEngine(nn.Module):
    """
    The 'Social Brain' of the Lattice. 
    Predicts velocity deltas based on swarm context.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, boid_states: jnp.ndarray):
        # 1. State Embedding (Position + Velocity + Thermal)
        x = nn.Dense(self.hidden_dim)(boid_states)
        
        # 2. Relational Processing (The 'Social' Layer)
        x = BoidAttention()(x)
        
        # 3. Policy Head: Output Acceleration Vector (DMPC-LD #34)
        acceleration = nn.Dense(3)(x) 
        return jnp.tanh(acceleration) # Capped reflexes
