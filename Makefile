# ============================================================
# MangaVrse — Top-level Makefile
# ============================================================
PYTHON ?= python
VENV := .venv

ifeq ($(OS),Windows_NT)
    SHELL := cmd.exe
    .SHELLFLAGS := /c
    VENV_PYTHON := $(VENV)\Scripts\python.exe
    VENV_PIP := $(VENV_PYTHON) -m pip
    PYTEST := $(VENV_PYTHON) -m pytest
    UVICORN := $(VENV_PYTHON) -m uvicorn
    MKDIR_P := mkdir
    RMDIR := rmdir /s /q
else
    VENV_PYTHON := $(VENV)/bin/python
    VENV_PIP := $(VENV_PYTHON) -m pip
    PYTEST := $(VENV_PYTHON) -m pytest
    UVICORN := $(VENV_PYTHON) -m uvicorn
    MKDIR_P := mkdir -p
    RMDIR := rm -rf
endif

.PHONY: setup setup-backend setup-frontend test test-backend gpu-check serve clean

# ---- Setup (one command) ----------------------------------------
setup: setup-backend setup-frontend
	@echo Setup complete.

setup-backend:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip setuptools wheel
	$(VENV_PIP) install -e ./backend[dev]

setup-frontend:
	cd frontend && npm install

# ---- Tests -------------------------------------------------------
test: test-backend
	@echo All tests passed.

test-backend:
	$(PYTEST) tests/ -v --tb=short

# ---- GPU check ---------------------------------------------------
gpu-check:
	$(VENV_PYTHON) scripts/gpu_check.py

# ---- Dev server --------------------------------------------------
serve:
	$(UVICORN) mangavrse.api.app:app --host 0.0.0.0 --port 8000 --reload

# ---- Housekeeping ------------------------------------------------
clean:
ifeq ($(OS),Windows_NT)
	if exist $(VENV) $(RMDIR) $(VENV)
	if exist frontend\.next $(RMDIR) frontend\.next
	if exist frontend\node_modules $(RMDIR) frontend\node_modules
else
	$(RMDIR) $(VENV) frontend/.next frontend/node_modules
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
endif
