.PHONY: install install-dev test lint format clean all

PYTHON := python3.11
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

install: $(VENV)/bin/python
	$(PIP) install -e .

install-dev: $(VENV)/bin/python
	$(PIP) install -e ".[dev]"

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

test: install-dev
	$(PYTHON_VENV) -m pytest tests/ -v

lint: install-dev
	$(PYTHON_VENV) -m ruff check crypto_mm_research tests
	$(PYTHON_VENV) -m mypy crypto_mm_research --ignore-missing-imports

format: install-dev
	$(PYTHON_VENV) -m black crypto_mm_research tests
	$(PYTHON_VENV) -m ruff check --fix crypto_mm_research tests

clean:
	rm -rf $(VENV)
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

all: install-dev lint test
