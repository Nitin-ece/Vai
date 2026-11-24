<#
.SYNOPSIS
    MangaVrse setup, test, and utility commands for Windows.
.DESCRIPTION
    PowerShell equivalent of the Makefile for systems without GNU Make.
.EXAMPLE
    .\setup.ps1 setup       # Create venv, install deps, npm install
    .\setup.ps1 test        # Run pytest
    .\setup.ps1 gpu-check   # Print CUDA diagnostics
    .\setup.ps1 serve       # Start FastAPI dev server
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "setup-backend", "setup-frontend", "test", "gpu-check", "serve", "clean")]
    [string]$Command = "setup"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VENV = Join-Path $Root ".venv"
$PIP = Join-Path $VENV "Scripts\pip.exe"
$PYTEST = Join-Path $VENV "Scripts\pytest.exe"
$PYTHON_VENV = Join-Path $VENV "Scripts\python.exe"
$UVICORN = Join-Path $VENV "Scripts\uvicorn.exe"

function Setup-Backend {
    Write-Host "`n=== Setting up Python backend ===" -ForegroundColor Cyan
    if (-not (Test-Path $VENV)) {
        python -m venv $VENV
    }
    & $PIP install --upgrade pip setuptools wheel
    & $PIP install -e "./backend[dev]"
    Write-Host "✓ Backend ready" -ForegroundColor Green
}

function Setup-Frontend {
    Write-Host "`n=== Setting up Next.js frontend ===" -ForegroundColor Cyan
    Set-Location (Join-Path $Root "frontend")
    npm install
    Set-Location $Root
    Write-Host "✓ Frontend ready" -ForegroundColor Green
}

switch ($Command) {
    "setup" {
        Setup-Backend
        Setup-Frontend
        Write-Host "`n✓ Setup complete.  Activate venv: .venv\Scripts\activate" -ForegroundColor Green
    }
    "setup-backend" { Setup-Backend }
    "setup-frontend" { Setup-Frontend }
    "test" {
        Write-Host "`n=== Running tests ===" -ForegroundColor Cyan
        & $PYTEST tests/ -v --tb=short
    }
    "gpu-check" {
        & $PYTHON_VENV scripts/gpu_check.py
    }
    "serve" {
        & $UVICORN mangavrse.api.app:app --host 0.0.0.0 --port 8000 --reload
    }
    "clean" {
        if (Test-Path $VENV) { Remove-Item $VENV -Recurse -Force }
        if (Test-Path (Join-Path $Root "frontend\.next")) { Remove-Item (Join-Path $Root "frontend\.next") -Recurse -Force }
        if (Test-Path (Join-Path $Root "frontend\node_modules")) { Remove-Item (Join-Path $Root "frontend\node_modules") -Recurse -Force }
        Get-ChildItem -Path $Root -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
        Write-Host "✓ Cleaned" -ForegroundColor Green
    }
}
