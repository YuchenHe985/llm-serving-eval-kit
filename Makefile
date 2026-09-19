PY ?= python3
export PYTHONPATH := src

.PHONY: test demo docs clean

test:
	$(PY) -m unittest discover -s tests -v

# Regenerate generated docs and the simulated demo (uses the fake server, no GPU).
docs:
	$(PY) -m llmeval catalog > docs/failure-catalog.md

demo:
	./examples/run_demo.sh

clean:
	rm -rf build dist src/*.egg-info __pycache__ */__pycache__
