# Makefile
SHELL = /bin/bash

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | findstr -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | findstr -E ".pytest_cache" | xargs rm -rf
	find . | findstr -E ".ipynb_checkpoints" | xargs rm -rf
	find . | findstr -E ".trash" | xargs rm -rf
	del -f .coverage

# Styling
.Phony: style
style:
	black .
	flake8
	python -m isort .

# Virtual Environment
.ONESHELL:
venv:
	python -m venv venv
	& venv/scripts/activate && \
	python -m pip install pip && \
	python -m pip install -e

# Test
.PHONY: test
test:
	pytest -m "not training"
	cd tests
	great_expectations checkpoint run projects
	great_expectations checkpoint run tags
	great_expectations checkpoint run labeled_projects

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."
	@echo "test    : performs tests for inputs and outputs."
