#* Variables
PYTHON := python3
PYTHONPATH := `pwd`
#* Formatters
.PHONY: black
black:
	black --version
	black --config pyproject.toml examples src tests

.PHONY: black-check
black-check:
	black --version
	black --diff --check --config pyproject.toml examples src tests

.PHONY: flake8
flake8:
	flake8 --version
	flake8 src tests

.PHONY: format-codestyle
format-codestyle: black flake8

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

.PHONY: check-codestyle
check-codestyle: black-check flake8

.PHONY: formatting
formatting: format-codestyle

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

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove ipynbcheckpoints-remove pytestcache-remove

all: format-codestyle cleanup test

ci: check-codestyle
