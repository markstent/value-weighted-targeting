.PHONY: all simulation hillstrom clean

all: simulation hillstrom

simulation:
	python src/simulation_analysis.py

hillstrom:
	python src/hillstrom_analysis.py

clean:
	rm -f outputs/simulation_results.json outputs/hillstrom_results.json
