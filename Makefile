# TUNGSTEN-v1.0 Makefile

WELD = python3 -m tungsten.forge
TARGET = ./src/engine/spine.py

all: initialize_lattice ignite_thrusters

initialize_lattice:
	@echo "INGESTING: Activating GALLIUM-Melt for Substrate Detection..."
	$(WELD) --bridge=detect --purity=high

ignite_thrusters:
	@echo "LAUNCHING: 4-Tier Logic Fusion initiated."
	$(WELD) --clusters=alpha,beta,gamma,delta --mode=kinetic_drift
