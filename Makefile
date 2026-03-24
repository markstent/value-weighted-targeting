PYTHON ?= ./.venv/bin/python

.PHONY: all simulation hillstrom manuscript paper test clean

all: simulation hillstrom manuscript

simulation:
	$(PYTHON) src/simulation_analysis.py

hillstrom:
	$(PYTHON) src/hillstrom_analysis.py

manuscript:
	cd docs && tectonic manuscript_v4.tex
	cp docs/manuscript_v4.pdf outputs/bonus_allocation_v4.pdf

paper: manuscript

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

clean:
	rm -f outputs/simulation_results.json outputs/hillstrom_results.json outputs/bonus_allocation_v3.docx
