"""
FastAPI application entry point.

Run with:
    uvicorn mangavrse.api.app:app --reload
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mangavrse import __version__
from mangavrse.api.routes import router

app = FastAPI(
    title="VAI — Volumetric Art Intelligence API",
    version=__version__,
    description="VAI: Manga to 2.5D Volumetric Art Intelligence Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# Mount data/output and benchmark directories as static assets
output_dir = Path("data/output")
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/assets/benchmark_panels", StaticFiles(directory="data/benchmark_panels"), name="benchmark_panels")
app.mount("/assets", StaticFiles(directory=str(output_dir)), name="assets")
app.mount("/data", StaticFiles(directory="data"), name="data")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}
