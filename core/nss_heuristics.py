import jax
import jax.numpy as jnp
import flax.linen as nn

class NSSIntuition(nn.Module):
    """
    NSS (#3): Solving logic puzzles using 'gut feelings'.
    Acts as a heuristic guide for the Gödel Transformer.
    """
    
    @nn.compact
    def __call__(self, system_state, candidate_proofs):
        # El 'Instinto' evalúa qué pruebas 'se sienten' correctas 
        # basándose en patrones pasados (RAP #27)
        intuition_score = nn.Dense(256)(system_state)
        intuition_score = nn.relu(intuition_score)
        
        # Genera un peso (Bias) para priorizar caminos de prueba
        prior = nn.Dense(candidate_proofs.shape[-1])(intuition_score)
        return jax.nn.softmax(prior)

class IntuitiveGodel(nn.Module):
    """
    The Full Loop: 
    1. Action Proposal
    2. NSS Gut Feeling (Where to look for proof)
    3. Gödel Proof Generation
    """
    def setup(self):
        self.godel = GodelTransformer()
        self.nss = NSSIntuition()

    def __call__(self, state):
        # El instinto guía al transformador para no perder tiempo
        gut_feeling = self.nss(state)
        action, proof = self.godel(state, hint=gut_feeling)
        return action, proof
