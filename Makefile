#* Variables
PYTHON := python3
PYTHONPATH := `pwd`
#* Formatters
.PHONY: format
format:
	ruff --version
	ruff format --config pyproject.toml examples src tests

.PHONY: format-check
format-check:
	ruff --version
	ruff format --diff --check --config pyproject.toml examples src tests

.PHONY: flake8
flake8:
	flake8 --version
	flake8 src tests

.PHONY: pre-commit-install
pre-commit-install:
	pre-commit install

.PHONY: test
test:
	pytest

.PHONY: test_coverage
test_coverage:
	pytest --cov=src

.PHONY: test_coverage_xml
test_coverage_xml:
	pytest --cov=src --cov-report=xml

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

#* Documentation
.PHONY: docs-serve
docs-serve:
	conda run --live-stream --name jsrm mkdocs serve

.PHONY: docs-build
docs-build:
	conda run --live-stream --name jsrm mkdocs build --clean

.PHONY: docs-build-strict
docs-build-strict:
	conda run --live-stream --name jsrm mkdocs build --clean --strict

.PHONY: docs-deploy
docs-deploy:
	conda run --live-stream --name jsrm mkdocs gh-deploy --force

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove ipynbcheckpoints-remove pytestcache-remove

all: format-codestyle cleanup test

ci: check-codestyle
