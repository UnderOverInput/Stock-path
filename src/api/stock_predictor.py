"""
Stock Price Prediction API Module
Standalone module that can be integrated into any FastAPI/Flask backend
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from torch.utils.data import TensorDataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class StockPriceLSTM(nn.Module):
    """LSTM Neural Network for Stock Price Prediction"""

    def __init__(self, input_size: int = 5, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 5, dropout: float = 0.2):
        super(StockPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )

        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)

        return out


@dataclass
class PredictionResult:
    """Container for prediction results"""
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    rmse: Optional[float] = None
    confidence: Optional[str] = None
    historical_data: Optional[list] = None
    prediction_date: Optional[str] = None


class StockPredictor:
    """
    Stock Price Predictor using LSTM Neural Network

    Usage:
        predictor = StockPredictor()
        result = predictor.predict("AAPL")
        print(result.close)  # Predicted close price
    """

    def __init__(self, sequence_length: int = 60, epochs: int = 5,
                 batch_size: int = 32, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model: Optional[StockPriceLSTM] = None
        self.scaler: Optional[MinMaxScaler] = None

    def _load_and_preprocess_data(self, ticker: str, period: str = '1y') -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[MinMaxScaler]]:
        """Load historical data and preprocess for LSTM"""
        try:
            data = yf.download(ticker, period=period, progress=False)[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.dropna(inplace=True)

            if len(data) < self.sequence_length:
                logger.warning(f"Insufficient data for {ticker}: {len(data)} rows")
                return None, None, None

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            X, y = [], []
            for i in range(self.sequence_length, len(scaled_data)):
                X.append(scaled_data[i - self.sequence_length:i])
                y.append(scaled_data[i])

            return np.array(X), np.array(y), scaler

        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None, None, None

    def _train_model(self, X: np.ndarray, y: np.ndarray,
                     progress_callback: Optional[callable] = None) -> StockPriceLSTM:
        """Train the LSTM model"""
        model = StockPriceLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                output = model(batch_X)
                loss = criterion(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()

            if progress_callback:
                progress_callback(epoch + 1, self.epochs, epoch_loss)

        return model

    def _evaluate_model(self, model: StockPriceLSTM, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model and return RMSE"""
        model.eval()
        X_eval = X[-100:] if len(X) > 100 else X
        y_eval = y[-100:] if len(y) > 100 else y

        with torch.no_grad():
            X_tensor = torch.tensor(X_eval, dtype=torch.float32)
            y_tensor = torch.tensor(y_eval, dtype=torch.float32)

            predictions = model(X_tensor).numpy()
            actuals = y_tensor.numpy()

        rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))
        return rmse

    def predict(self, ticker: str, period: str = '1y',
                include_historical: bool = False,
                progress_callback: Optional[callable] = None) -> Optional[PredictionResult]:
        """
        Predict next day's stock prices

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "GOOGL")
            period: Historical data period (default "1y")
            include_historical: Include recent historical data in result
            progress_callback: Optional callback for training progress

        Returns:
            PredictionResult with predicted values or None if prediction fails
        """
        ticker = ticker.upper().strip()

        # Load and preprocess data
        X, y, scaler = self._load_and_preprocess_data(ticker, period)

        if X is None or y is None or scaler is None:
            return None

        self.scaler = scaler

        # Train model
        model = self._train_model(X, y, progress_callback)
        self.model = model

        # Evaluate model
        rmse = self._evaluate_model(model, X, y)

        # Make prediction
        model.eval()
        with torch.no_grad():
            # Get last 60 days of data
            recent_data = yf.download(ticker, period='70d', progress=False)[['Open', 'High', 'Low', 'Close', 'Volume']]
            recent_data.dropna(inplace=True)
            last_60_days = scaler.transform(recent_data[-self.sequence_length:])

            input_tensor = torch.tensor(last_60_days, dtype=torch.float32).unsqueeze(0)
            predicted_scaled = model(input_tensor)
            predicted_values = scaler.inverse_transform(predicted_scaled.numpy())[0]

        # Determine confidence level based on RMSE
        if rmse < 0.05:
            confidence = "High"
        elif rmse < 0.1:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Build result
        result = PredictionResult(
            ticker=ticker,
            open=float(predicted_values[0]),
            high=float(predicted_values[1]),
            low=float(predicted_values[2]),
            close=float(predicted_values[3]),
            volume=float(predicted_values[4]),
            rmse=rmse,
            confidence=confidence,
            prediction_date=pd.Timestamp.now().strftime("%Y-%m-%d")
        )

        if include_historical:
            hist = yf.download(ticker, period='10d', progress=False)[['Open', 'High', 'Low', 'Close', 'Volume']]
            result.historical_data = hist.reset_index().to_dict(orient='records')

        return result

    def predict_batch(self, tickers: list, period: str = '1y') -> Dict[str, Optional[PredictionResult]]:
        """Predict for multiple tickers"""
        results = {}
        for ticker in tickers:
            results[ticker] = self.predict(ticker, period)
        return results


# Singleton instance for reuse
_predictor_instance: Optional[StockPredictor] = None


def get_predictor() -> StockPredictor:
    """Get or create singleton predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = StockPredictor()
    return _predictor_instance


def quick_predict(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Quick prediction function for simple integration

    Usage:
        from stock_predictor import quick_predict
        result = quick_predict("AAPL")
        if result:
            print(f"Predicted close: ${result['close']:.2f}")
    """
    predictor = get_predictor()
    result = predictor.predict(ticker)

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
        "prediction_date": result.prediction_date
    }
