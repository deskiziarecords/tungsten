import jax
import jax.numpy as jnp
from jax import jit, vmap

@jit
def weierstrass_kernel(query_vec, letter_coords, sigma=1.0):
    """
    Smooths the decision manifold. 
    Instead of 1 or 0, it gives a 'Resonance Score'.
    """
    # L2 distance between query and the 100 letters
    dists = jnp.sum((query_vec - letter_coords)**2, axis=-1)
    
    # The Heat Kernel (Weierstrass Transform)
    # G(x) = (1 / sqrt(4 * pi * sigma)) * exp(-d^2 / (4 * sigma))
    normalization = 1.0 / jnp.sqrt(4 * jnp.pi * sigma)
    resonance = normalization * jnp.exp(-dists / (4 * sigma))
    
    return resonance # Returns a probability 'blur' over the lattice
