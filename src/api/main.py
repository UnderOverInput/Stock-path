"""
FastAPI Stock Prediction API
Deploy as a standalone service or integrate into existing FastAPI app
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from stock_predictor import StockPredictor, quick_predict, PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price predictions for frameworx.site integration",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for frameworx.site
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frameworx.site",
        "https://www.frameworx.site",
        "http://localhost:3000",
        "http://localhost:5173",
        "*"  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive predictions
executor = ThreadPoolExecutor(max_workers=4)

# In-memory cache for predictions (use Redis in production)
prediction_cache: Dict[str, Dict[str, Any]] = {}


# Request/Response Models
class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol", example="AAPL")
    period: str = Field(default="1y", description="Historical data period")
    include_historical: bool = Field(default=False, description="Include recent historical data")


class BatchPredictionRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of stock ticker symbols")
    period: str = Field(default="1y", description="Historical data period")


class PredictionResponse(BaseModel):
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    rmse: Optional[float]
    confidence: Optional[str]
    prediction_date: Optional[str]
    historical_data: Optional[list] = None
    cached: bool = False


class BatchPredictionResponse(BaseModel):
    predictions: Dict[str, Optional[PredictionResponse]]
    success_count: int
    failed_tickers: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str


# Helper functions
def run_prediction(ticker: str, period: str = "1y",
                   include_historical: bool = False) -> Optional[Dict[str, Any]]:
    """Run prediction synchronously (for thread pool)"""
    predictor = StockPredictor()
    result = predictor.predict(ticker, period, include_historical)

    if result is None:
        return None

    return {
        "ticker": result.ticker,
        "open": result.open,
        "high": result.high,
        "low": result.low,
        "close": result.close,
        "volume": result.volume,
        "rmse": result.rmse,
        "confidence": result.confidence,
        "prediction_date": result.prediction_date,
        "historical_data": result.historical_data,
        "cached": False
    }


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Stock Price Prediction API"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Stock Price Prediction API"
    }


@app.get("/api/predict/{ticker}", response_model=PredictionResponse)
async def predict_stock(
    ticker: str,
    period: str = Query(default="1y", description="Historical data period (1mo, 3mo, 6mo, 1y, 2y)"),
    include_historical: bool = Query(default=False, description="Include recent historical data"),
    use_cache: bool = Query(default=True, description="Use cached predictions if available")
):
    """
    Get stock price prediction for a single ticker

    - **ticker**: Stock symbol (e.g., AAPL, GOOGL, MSFT)
    - **period**: Historical data period for training
    - **include_historical**: Include last 10 days of actual data
    - **use_cache**: Use cached prediction if available (within 1 hour)
    """
    ticker = ticker.upper().strip()

    # Check cache
    if use_cache and ticker in prediction_cache:
        cached = prediction_cache[ticker]
        cached["cached"] = True
        return cached

    # Run prediction in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        run_prediction,
        ticker,
        period,
        include_historical
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not generate prediction for {ticker}. "
                   "Ensure it's a valid ticker with sufficient historical data."
        )

    # Cache result
    prediction_cache[ticker] = result

    return result


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_stock_post(request: PredictionRequest):
    """
    Get stock price prediction (POST method)

    Use this endpoint when you need to send prediction parameters in the request body.
    """
    return await predict_stock(
        ticker=request.ticker,
        period=request.period,
        include_historical=request.include_historical
    )


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Get predictions for multiple stocks

    - **tickers**: List of stock symbols
    - **period**: Historical data period for training

    Note: Batch predictions may take longer as each ticker requires model training.
    """
    predictions = {}
    failed_tickers = []

    for ticker in request.tickers:
        try:
            result = await predict_stock(
                ticker=ticker,
                period=request.period,
                include_historical=False,
                use_cache=True
            )
            predictions[ticker] = result
        except HTTPException:
            failed_tickers.append(ticker)
            predictions[ticker] = None

    return {
        "predictions": predictions,
        "success_count": len(request.tickers) - len(failed_tickers),
        "failed_tickers": failed_tickers
    }


@app.get("/api/supported-tickers")
async def get_supported_tickers():
    """
    Get a list of commonly supported stock tickers

    Note: Any valid stock ticker from Yahoo Finance is supported,
    this is just a curated list of popular ones.
    """
    return {
        "popular": [
            "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA",
            "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "DIS"
        ],
        "tech": [
            "AAPL", "GOOGL", "MSFT", "META", "NVDA", "AMD", "INTC",
            "CRM", "ORCL", "ADBE", "CSCO", "IBM", "QCOM", "TXN"
        ],
        "crypto_related": [
            "COIN", "MSTR", "SQ", "PYPL", "RIOT", "MARA", "HUT"
        ],
        "finance": [
            "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW"
        ]
    }


@app.delete("/api/cache")
async def clear_cache():
    """Clear prediction cache"""
    prediction_cache.clear()
    return {"message": "Cache cleared successfully"}


@app.get("/api/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "cached_tickers": list(prediction_cache.keys()),
        "cache_size": len(prediction_cache)
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "type": type(exc).__name__
        }
    )


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
