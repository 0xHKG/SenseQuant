.PHONY: help install lint format type test test-cov run clean all

help:
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  lint       - Run ruff linter (check only)"
	@echo "  format     - Run ruff formatter (auto-fix)"
	@echo "  type       - Run mypy type checker"
	@echo "  test       - Run pytest"
	@echo "  test-cov   - Run pytest with coverage report"
	@echo "  run        - Run application in dry-run mode"
	@echo "  clean      - Remove generated files"

install:
	pip install -r requirements.txt

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

type:
	mypy src

test:
	pytest -q

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

run:
	python -m src.app.main

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov .ruff_cache

all: lint type test
