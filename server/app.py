from __future__ import annotations

import os
import asyncio

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app

from medtriage_env.models import MedTriageAction, MedTriageObservation
from server.environment import MedTriageEnvironment

environment = MedTriageEnvironment

app: FastAPI = create_app(
    environment,
    MedTriageAction,
    MedTriageObservation,
)

# Remove default /health and replace with ours
app.routes[:] = [r for r in app.routes if getattr(r, "path", "") != "/health"]

@app.get("/health", tags=["ops"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "healthy", "environment": "MedTriageEnv"})

@app.on_event("startup")
async def warmup_mcp():
    """Hit /mcp internally on startup so the handler is warm before validators arrive."""
    await asyncio.sleep(2)  # let uvicorn finish binding
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://127.0.0.1:8000/mcp",
                json={"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 0},
                timeout=10.0,
            )
    except Exception:
        pass  # warmup is best-effort — never crash startup

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
