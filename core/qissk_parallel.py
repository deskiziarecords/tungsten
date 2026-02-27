@jax.vmap
def qissk_state_eval(state_a, state_b, objective_fn):
    """
    Evaluates two potential future states in parallel (Superposition).
    """
    val_a = objective_fn(state_a)
    val_b = objective_fn(state_b)
    
    # Mantiene ambos hasta que uno demuestra 'Pureza' (Purity) superior
    return jnp.where(val_a > val_b, state_a, state_b)
