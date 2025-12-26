---
title: Stock Price Prediction
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# LSTM Stock Price Predictor

Predict next day's stock prices using an LSTM neural network trained on historical data.

## Web UI
Visit the Space directly to use the interactive interface.

## API Access

Call the prediction API from your application:

```bash
curl https://iofocus-stocks-price-prediction.hf.space/api/predict/AAPL
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

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Gradio Web UI |
| `/api/predict/{ticker}` | Get prediction for a stock |
| `/api/health` | Health check |

## Integration with frameworx.site

```javascript
const response = await fetch('https://iofocus-stocks-price-prediction.hf.space/api/predict/AAPL');
const prediction = await response.json();
console.log(`Predicted close: $${prediction.close}`);
```
