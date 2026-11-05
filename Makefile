.PHONY: install test lint web-build bench demo demo-script

install:
	python -m pip install -e ".[api,agents,benchmark,dev]"
	npm --prefix web ci

test:
	python -m pytest

lint:
	python -m ruff check core api benchmark tests
	python -m mypy core api benchmark

web-build:
	npm --prefix web run build

bench:
	python -m benchmark.run

demo:
	docker compose up --build

demo-script:
	python -m demo.run_demo
