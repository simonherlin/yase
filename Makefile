PROJECT_NAME := yase

.PHONY: help sync test lint typecheck format native build build-isolated build-native wheels runtime-smoke check check-dist clean

help:
	@echo "Targets: sync test lint typecheck format native build build-isolated build-native wheels runtime-smoke check check-dist clean"

sync:
	uv sync --dev

test:
	uv run pytest

runtime-smoke:
	uv run python tools/runtime_smoke.py

lint:
	uv run ruff check src tests tools

typecheck:
	uv run mypy --ignore-missing-imports src/yase

format:
	uv run ruff format src tests tools

native:
	uv run python tools/build_native.py

build:
	uv run python tools/build_package.py --no-isolation

build-isolated:
	uv run python tools/build_package.py

build-native:
	YASE_BUILD_NATIVE=1 uv run python tools/build_package.py --native --no-isolation

wheels:
	uv run python tools/build_wheels.py --platform auto

check: lint typecheck
	uv lock --check
	uv run ruff format --check src tests tools
	uv run pytest --cov=yase --cov-report=term-missing

check-dist: build
	uv run python tools/check_distribution.py

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage dist
