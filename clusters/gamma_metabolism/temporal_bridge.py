# src/tungsten/clusters/gamma_metabolism/temporal_bridge.py
from ...vector.temporal_decay_rgf import RGFTemporalDecay

class GammaTemporalBridge:
    def __init__(self):
        self.rgf = RGFTemporalDecay(vector_dim=768)

    def synchronize_clock(self, new_telemetry):
        """
        Applies RGF-F decay to old state before 
        fusing with new telemetry.
        """
        # Retrieve the 'Decayed' previous state
        decayed_state, uncertainty = self.rgf.query("cluster_alpha_v1")
        
        # If uncertainty is too high, the 'Weld' requires more new data.
        if jnp.mean(uncertainty) > 0.8:
            return self.trigger_re_anneal(new_telemetry)
            
        return self.fuse(decayed_state, new_telemetry)
