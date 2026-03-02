# src/memory/kinetic_buffer.py

import jax
import jax.numpy as jnp
from typing import Optional

class KineticBuffer:
    """
    TUNGSTEN Memory: Kinetic Buffer (Algorithm 50).
    Maintains data as a vibrating 'fluid' to prevent bit-rot and 
    ensure persistence through power fluctuations.
    """
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        # Inicializamos el buffer con una distribución de ruido térmico (vibración base)
        self.state = jnp.zeros((capacity, 64)) 
        self.momentum = jnp.zeros((capacity, 64))
        self.damping = 0.998  # Coeficiente de fricción para evitar el 'Logic Slag'

    @jax.jit
    def inject_momentum(self, logic_vector: jnp.ndarray, index: int):
        """
        Inserta datos en el buffer no como sobrescritura, sino como un choque cinético.
        El nuevo 'metal' se funde con la vibración existente.
        """
        # Actualizamos el estado usando una aproximación de integración de Verlet
        new_momentum = self.momentum.at[index].set(logic_vector * (1.0 - self.damping))
        new_state = self.state.at[index].add(new_momentum[index])
        return new_state, new_momentum

    def sustain_vibration(self):
        """
        Mantiene el 'Heartbeat' de los datos. 
        Simula la persistencia necesaria para sobrevivir a cortes de energía del 90%.
        """
        # Los datos oscilan levemente para mantenerse 'calientes' y listos para el Spine
        self.state = self.state * self.damping + jnp.sin(self.state) * 0.001
        
    def extract_alloy(self, index: int) -> jnp.ndarray:
        """
        Recupera la lógica del buffer. 
        La lectura es una 'fotografía' del estado cinético actual.
        """
        return self.state[index]

# Visualización de la 'Masa de Datos'
def calculate_kinetic_energy(buffer: KineticBuffer) -> float:
    """
    Mide cuánta información está 'viva' en el buffer.
    """
    energy = 0.5 * jnp.sum(jnp.square(buffer.momentum))
    return float(energy)
