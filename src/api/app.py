"""
FastAPI Service for Time Series Forecasting
RESTful API for model inference and management
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import pandas as pd
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import mlflow
import mlflow.pytorch
from loguru import logger
import redis
import json
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response


# Prometheus metrics
PREDICTION_REQUESTS = Counter("forecast_requests_total", "Total forecast requests")
PREDICTION_LATENCY = Histogram("forecast_latency_seconds", "Forecast response time")
MODEL_LOAD_COUNTER = Counter("model_loads_total", "Total model loads")
ERROR_COUNTER = Counter("api_errors_total", "Total API errors", ["error_type"])

# Initialize FastAPI app
app = FastAPI(
    title="ProbForecast API",
    description="Production-ready time series forecasting service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
redis_client = None
model_cache = {}


# Pydantic models
class ForecastRequest(BaseModel):
    """Request model for forecasting"""
    series_id: str = Field(..., description="Time series identifier")
    horizon: int = Field(30, ge=1, le=365, description="Forecast horizon in time steps")
    model_version: Optional[str] = Field("latest", description="Model version to use")
    model_type: Optional[str] = Field("deepar", description="Model type")
    quantiles: Optional[List[float]] = Field([0.1, 0.5, 0.9], description="Quantiles for prediction intervals")
    include_history: bool = Field(False, description="Include historical data in response")

    @field_validator('quantiles')
    @classmethod
    def validate_quantiles(cls, v):
        if v is not None:
            for q in v:
                if not 0 < q < 1:
                    raise ValueError('Quantiles must be between 0 and 1')
        return v


class ForecastPoint(BaseModel):
    """Single forecast point"""
    timestamp: datetime
    mean: float
    quantiles: Dict[str, float] = Field(default_factory=dict)


class ForecastResponse(BaseModel):
    """Response model for forecasting"""
    series_id: str
    forecast: List[ForecastPoint]
    model_version: str
    model_type: str
    forecast_horizon: int
    generated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    models_loaded: int
    redis_status: str


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client

    logger.info("Starting ProbForecast API...")

    # Initialize Redis
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None

    logger.info("ProbForecast API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ProbForecast API...")

    if redis_client:
        redis_client.close()


async def get_cached_prediction(cache_key: str):
    """Get cached prediction"""
    if not redis_client:
        return None

    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache retrieval failed: {e}")

    return None


async def cache_prediction(cache_key: str, prediction: dict, ttl: int = 3600):
    """Cache prediction result"""
    if not redis_client:
        return

    try:
        redis_client.setex(
            cache_key,
            ttl,
            json.dumps(prediction, default=str)
        )
    except Exception as e:
        logger.warning(f"Cache storage failed: {e}")


# API Routes
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint"""
    return {"message": "ProbForecast API", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    redis_status = "connected" if redis_client else "disconnected"

    try:
        if redis_client:
            redis_client.ping()
    except:
        redis_status = "error"

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        models_loaded=len(model_cache),
        redis_status=redis_status
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(
    request: ForecastRequest,
    background_tasks: BackgroundTasks
):
    """Generate forecast for time series"""
    PREDICTION_REQUESTS.inc()

    with PREDICTION_LATENCY.time():
        try:
            # Generate cache key
            cache_key = f"forecast_{request.series_id}_{request.horizon}_{request.model_type}_{request.model_version}"

            # Check cache
            cached_result = await get_cached_prediction(cache_key)
            if cached_result:
                logger.info(f"Returning cached prediction for {request.series_id}")
                return ForecastResponse(**cached_result)

            # Generate forecast (mock implementation for demo)
            forecast_points = []
            base_time = datetime.now()

            for i in range(request.horizon):
                timestamp = base_time + timedelta(hours=i)

                # Mock prediction values with realistic probabilistic forecasts
                mean_value = 100 + np.random.normal(0, 10)
                quantile_values = {}

                for q in request.quantiles:
                    if q < 0.5:
                        quantile_values[f"p{int(q*100)}"] = mean_value - abs(np.random.normal(0, 5))
                    elif q > 0.5:
                        quantile_values[f"p{int(q*100)}"] = mean_value + abs(np.random.normal(0, 5))
                    else:
                        quantile_values[f"p{int(q*100)}"] = mean_value

                forecast_points.append(
                    ForecastPoint(
                        timestamp=timestamp,
                        mean=mean_value,
                        quantiles=quantile_values
                    )
                )

            # Create response
            response = ForecastResponse(
                series_id=request.series_id,
                forecast=forecast_points,
                model_version=request.model_version,
                model_type=request.model_type,
                forecast_horizon=request.horizon,
                generated_at=datetime.now(),
                metadata={
                    "quantiles_requested": request.quantiles
                }
            )

            # Cache result
            background_tasks.add_task(
                cache_prediction,
                cache_key,
                response.dict()
            )

            return response

        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            ERROR_COUNTER.labels(error_type="prediction").inc()
            raise HTTPException(
                status_code=500,
                detail=f"Forecast generation failed: {str(e)}"
            )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    ERROR_COUNTER.labels(error_type="internal").inc()
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
