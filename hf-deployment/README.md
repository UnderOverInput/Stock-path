---
title: Stock Price Prediction API
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
- fastapi
- stock-prediction
- lstm
- machine-learning
pinned: false
short_description: LSTM-based stock price prediction API
license: mit
---

# LSTM Stock Price Prediction API

A FastAPI backend service that provides stock price predictions using LSTM neural networks trained on historical data from Yahoo Finance.

## API Endpoints

### Health Check
```bash
GET /api/health
```

### Get Stock Prediction
```bash
GET /api/predict/{ticker}
```

Example:
```bash
curl https://your-space-name.hf.space/api/predict/AAPL
```

Response:
```json
{
  "ticker": "AAPL",
  "open": 185.42,
  "high": 187.89,
  "low": 184.15,
  "close": 186.73,
  "volume": 52340000,
  "rmse": 0.0423,
  "confidence": "High"
}
```

## Integration Example

```javascript
// Fetch prediction from your app
const response = await fetch('https://your-space-name.hf.space/api/predict/AAPL');
const prediction = await response.json();
console.log(`Predicted close: $${prediction.close}`);
```

## Features

- LSTM-based price prediction for next trading day
- Returns Open, High, Low, Close, and Volume predictions
- Model confidence scoring based on RMSE
- CORS enabled for cross-origin requests
- Simple HTML UI at root for testing
- RESTful API for easy integration

## Model Details

- **Architecture**: 2-layer LSTM with dropout
- **Training Data**: 1 year of historical stock data
- **Sequence Length**: 60 days
- **Training Epochs**: 5
- **Data Source**: Yahoo Finance via yfinance
