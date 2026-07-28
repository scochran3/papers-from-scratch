# LIMA from (mostly) scratch — common tasks.
#
# Run everything from the repo root:  make <target>
# See all targets:                    make help

# Use one shell per recipe so multi-line commands share state.
.ONESHELL:
.DEFAULT_GOAL := help

PYTHON  ?= python
UV      ?= uv
CONFIG  ?= configs/qwen.yaml

.PHONY: help lock install analyze clean

help:  ## List available targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

lock:  ## Pin requirements.in -> requirements.txt (fully resolved).
	$(UV) pip compile requirements.in -o requirements.txt

install:  ## Install pinned deps (and the editable lima package) into the active venv.
	$(UV) pip install -r requirements.txt

analyze:  ## Inspect the dataset described by CONFIG (default configs/qwen.yaml).
	$(PYTHON) scripts/analyze_data.py --config $(CONFIG)

clean:  ## Remove Python caches.
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
