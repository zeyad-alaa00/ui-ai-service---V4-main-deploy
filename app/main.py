# app/main.py

import logging
from contextlib import asynccontextmanager

import torch

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.model import (
    load_models,
    is_model_ready,
)

from app.router import router

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "ui-ai-service"
)

logger.info(
    "Booting UI AI Service..."
)


if torch.cuda.is_available():

    try:

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        torch.backends.cudnn.benchmark = True

        logger.info(
            "CUDA optimizations enabled"
        )

        logger.info(
            "GPU: %s",
            torch.cuda.get_device_name(0),
        )

        total_vram = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        logger.info(
            "VRAM: %.2f GB",
            total_vram,
        )

    except Exception:

        logger.exception(
            "CUDA optimization failed"
        )

else:

    logger.warning(
        "CUDA not available — running on CPU"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting AI service..."
    )

    try:

        load_models()

        logger.info(
            "Qwen models ready ✅"
        )

    except Exception:

        logger.exception(
            "Qwen loading failed ❌"
            
        )

    logger.info(
        "Startup sequence completed"
    )

    yield


    logger.info(
        "Shutting down AI service..."
    )

    try:

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            torch.cuda.ipc_collect()

            logger.info(
                "CUDA cache cleared"
            )

    except Exception:

        logger.exception(
            "CUDA cleanup failed"
        )



app = FastAPI(
    title="UI AI Service",
    version="1.0.0",
    lifespan=lifespan,
)



app.include_router(router)



@app.get("/")
async def root():

    return {
        "service": "ui-ai-service",
        "status": "running",
        "gpu": torch.cuda.is_available(),
    }



@app.get(
    "/health",
    tags=["health"],
)
async def health():

    return {
        "status": "ok",
    }


@app.get("/ready")
def ready():
    return {"ready": True}

@app.get("/")
def root():
    return {"status": "running"}

@app.get(
    "/ready",
    tags=["health"],
)
async def ready():

    if not is_model_ready():

        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
            },
        )

    return {
        "status": "ready",
    }