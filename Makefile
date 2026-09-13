PROJECT_NAME := yase

.PHONY: help sync test lint format build check clean

help:
	@echo "Targets: sync test lint format build check clean"

sync:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

build:
	uv build

check: lint
	uv run ruff format --check src tests
	uv run pytest

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage dist
