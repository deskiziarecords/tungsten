# /tungsten/forge.py
import sys
from tungsten.engine.spine import TungstenSpine

def ignite():
    """The entry point for the TUNGSTEN engine."""
    spine = TungstenSpine()
    print("🚀 TUNGSTEN Forge: Igniting Lattice...")
    
    # The 'Melt' - Ingesting telemetry
    # The 'Weld' - Executing Alpha/Beta/Gamma/Delta synchronization
    result = spine.recursive_weld(sys.stdin.read())
    print(f"✅ Weld Complete: {result}")

if __name__ == "__main__":
    ignite()
