PYTHON ?= ./.venv/bin/python

.PHONY: all simulation hillstrom manuscript paper test clean

all: simulation hillstrom manuscript

simulation:
	$(PYTHON) src/simulation_analysis.py

hillstrom:
	$(PYTHON) src/hillstrom_analysis.py

manuscript:
	@if [ -f docs/manuscript_v3.md ]; then $(PYTHON) scripts/generate_paper_v3.py; else echo "No manuscript source found. Create docs/manuscript_v4.md first."; fi

paper: manuscript

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

clean:
	rm -f outputs/simulation_results.json outputs/hillstrom_results.json outputs/bonus_allocation_v3.docx
