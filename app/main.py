"""FastAPI application main entrypoint for Olist Late Delivery Inference Service."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.monitoring import PrometheusMiddleware
from app.routes import router
from config.logging_config import logger
from config.settings import settings
from src.predict import prediction_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info(f"Starting {settings.project.name} v{settings.project.version} ({settings.environment})")
    try:
        # Preload model and preprocessor at startup
        _ = prediction_service.model
        logger.info(f"Model '{settings.model.name}' successfully loaded into memory.")
    except Exception as e:
        logger.error(f"Error during model startup loading: {e}")

    yield

    logger.info("Shutting down Olist MLOps inference service.")


def create_app() -> FastAPI:
    """Factory function creating configured FastAPI application."""
    app = FastAPI(
        title="Olist Late Delivery Prediction Service",
        description=(
            "Production MLOps inference service predicting whether an e-commerce order "
            "will be delivered on-time or late. Built with FastAPI, Scikit-learn, "
            "MLflow, Great Expectations, DVC, and Docker."
        ),
        version=settings.project.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add Prometheus monitoring middleware
    app.add_middleware(PrometheusMiddleware)

    # Add CORS middleware for frontend / dashboard integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
