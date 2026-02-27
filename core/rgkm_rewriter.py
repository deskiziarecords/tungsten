import jax
import jax.numpy as jnp
import flax.linen as nn

class RGKM(nn.Module):
    """
    Self-Rewriting Logic: The system adjusts its own 'Metabolism' 
    rate based on task complexity.
    """
    @nn.compact
    def __call__(self, task_complexity: jnp.ndarray, weights: jnp.ndarray):
        # El modelo genera un 'Delta de Pesos' para sí mismo
        # Si la complejidad sube, reescribe su propia velocidad de aprendizaje
        rewrite_gate = nn.Dense(weights.shape[-1])(task_complexity)
        rewritten_weights = weights * jax.nn.sigmoid(rewrite_gate)
        
        return rewritten_weights
