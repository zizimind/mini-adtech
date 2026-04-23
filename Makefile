PYTHON ?= python3
VENV_DIR ?= .venv
PIP := $(VENV_DIR)/bin/pip
PYTEST := $(VENV_DIR)/bin/pytest
PYTHON_BIN := $(VENV_DIR)/bin/python
IMAGE_NAME ?= mini-adtech:latest

.PHONY: venv install install-rag test run benchmark clean docker-build docker-run docker-test compose-run compose-test

venv:
	$(PYTHON) -m venv "$(VENV_DIR)"

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-rag: install
	$(PIP) install -r requirements-rag.txt

test:
	$(PYTEST) -q

run:
	$(PYTHON_BIN) mini_adtech.py

benchmark:
	$(PYTHON_BIN) benchmark.py

clean:
	rm -rf "$(VENV_DIR)" .pytest_cache artifacts

docker-build:
	docker build -t "$(IMAGE_NAME)" .

docker-run:
	docker run --rm "$(IMAGE_NAME)"

docker-test:
	docker run --rm "$(IMAGE_NAME)" pytest -q

compose-run:
	docker compose run --rm app

compose-test:
	docker compose run --rm test
