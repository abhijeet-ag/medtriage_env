"""
server/app.py — FastAPI application entry point for MedTriageEnv.

The OpenEnv framework handles all routing via create_app().  This file
wires together the environment, action/observation types, and any
project-specific configuration, then exposes the assembled app for
uvicorn to serve.

Endpoints provided by create_app():
    WS   /ws      — WebSocket interface for real-time agent interaction
    POST /reset   — Start a new episode; accepts {"difficulty": "easy|medium|hard"}
    POST /step    — Advance the current episode with an action

Endpoints added here:
    GET  /health  — Liveness probe; returns {"status": "healthy", "environment": "MedTriageEnv"}

Environment variables:
    ENABLE_WEB_INTERFACE   — Set to "true" to mount the built-in OpenEnv web UI
                             (useful during local development; disable in prod)
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app

from medtriage_env.models import MedTriageAction, MedTriageObservation
from server.environment import MedTriageEnvironment

# ---------------------------------------------------------------------------
# Environment instantiation
# ---------------------------------------------------------------------------

environment = MedTriageEnvironment  # pass class, not instance

# ---------------------------------------------------------------------------
# App creation via OpenEnv factory
# ---------------------------------------------------------------------------
# create_app() registers /ws and /reset (and /step if included in the spec).
# Passing the concrete Action and Observation types lets the framework
# generate accurate OpenAPI schemas and perform request validation.

_enable_web: bool = os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true"

app: FastAPI = create_app(
    environment,
    MedTriageAction,
    MedTriageObservation,
)

# ---------------------------------------------------------------------------
# /health endpoint — override create_app() default
# ---------------------------------------------------------------------------

# Remove the /health route registered by create_app() and replace with ours
app.routes[:] = [r for r in app.routes if getattr(r, "path", "") != "/health"]

@app.get("/health", tags=["ops"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "healthy", "environment": "MedTriageEnv"})

# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
