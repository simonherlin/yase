PROJECT_NAME := yase

.PHONY: help sync test lint format native build build-isolated build-native wheels runtime-smoke check check-dist clean

help:
	@echo "Targets: sync test lint format native build build-isolated build-native wheels runtime-smoke check check-dist clean"

sync:
	uv sync --dev

test:
	uv run pytest

runtime-smoke:
	uv run python tools/runtime_smoke.py

lint:
	uv run ruff check src tests tools

format:
	uv run ruff format src tests tools

native:
	uv run python tools/build_native.py

build:
	uv run python tools/build_package.py --no-isolation

build-isolated:
	uv run python tools/build_package.py

build-native:
	YASE_BUILD_NATIVE=1 uv run python tools/build_package.py --native

wheels:
	uv run python tools/build_wheels.py --platform auto

check: lint
	uv lock --check
	uv run ruff format --check src tests tools
	uv run pytest --cov=yase --cov-report=term-missing

check-dist: build
	uv run python -c "import pathlib, zipfile; wheels = sorted(pathlib.Path('dist').glob('yase-*.whl'), key=lambda path: path.stat().st_mtime); assert wheels, 'no Yase wheel found'; wheel = wheels[-1]; names = zipfile.ZipFile(wheel).namelist(); assert 'yase/__init__.py' in names; assert wheel.name.endswith('py3-none-any.whl'), wheel; assert not any('/_native' in name for name in names); print(wheel)"

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage dist
