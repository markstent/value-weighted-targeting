PYTHON ?= ./.venv/bin/python

.PHONY: all simulation supplementary hillstrom manuscript paper test clean

all: simulation supplementary hillstrom manuscript

simulation:
	$(PYTHON) src/simulation_analysis.py

supplementary:
	$(PYTHON) src/supplementary_analysis.py

hillstrom:
	$(PYTHON) src/hillstrom_analysis.py

manuscript:
	cd docs && tectonic manuscript_v18.tex
	cp docs/manuscript_v18.pdf outputs/manuscript_v18.pdf

paper: manuscript

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

clean:
	rm -f outputs/simulation_results.json outputs/hillstrom_results.json outputs/supplementary_results.json
