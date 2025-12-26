# Stock Prediction Integration Guide for frameworx.site

This guide explains how to integrate the LSTM Stock Price Prediction functionality into your frameworx.site project.

## Overview

The Stock-path project provides AI-powered stock price predictions using an LSTM neural network trained on historical data from Yahoo Finance. There are multiple integration approaches depending on your architecture.

## Integration Options

### Option 1: Embed Streamlit App (Quickest)

If you want to quickly add stock predictions, embed the existing Streamlit app:

```html
<!-- Add to your frameworx.site page -->
<iframe
  src="https://frameworx.streamlit.app/?embed=true&embed_options=dark_theme"
  width="100%"
  height="800px"
  frameborder="0"
  style="border-radius: 12px;"
></iframe>
```

Or from Hugging Face:
```html
<iframe
  src="https://huggingface.co/spaces/iofocus/stocks_price_prediction"
  width="100%"
  height="800px"
></iframe>
```

### Option 2: Deploy Prediction API (Recommended)

Deploy the FastAPI backend for full control and integration.

#### 1. Deploy the API

**Using Docker:**
```bash
cd Stock-path/src/api
docker build -t stock-prediction-api .
docker run -p 8000:8000 stock-prediction-api
```

**Using Python directly:**
```bash
cd Stock-path/src/api
pip install -r requirements.txt fastapi uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Deploy to Railway/Render/Fly.io:**
```yaml
# railway.json or render.yaml
services:
  - name: stock-prediction-api
    root: src/api
    env: python
    buildCommand: pip install -r requirements.txt fastapi uvicorn
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

#### 2. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict/{ticker}` | GET | Get prediction for single stock |
| `/api/predict` | POST | Get prediction (body params) |
| `/api/predict/batch` | POST | Get predictions for multiple stocks |
| `/api/supported-tickers` | GET | List popular tickers |
| `/api/health` | GET | Health check |

**Example API Call:**
```bash
curl https://your-api.frameworx.site/api/predict/AAPL
```

**Response:**
```json
{
  "ticker": "AAPL",
  "open": 185.42,
  "high": 187.89,
  "low": 184.15,
  "close": 186.73,
  "volume": 52340000,
  "rmse": 0.0423,
  "confidence": "High",
  "prediction_date": "2025-01-15"
}
```

### Option 3: Use TypeScript Client in frameworx.site

#### Installation

Copy the client files to your frameworx.site project:
```bash
cp Stock-path/src/client/stockPredictionClient.ts frameworxsite/src/lib/
cp Stock-path/src/client/StockPredictionWidget.tsx frameworxsite/src/components/
```

#### Usage in React/Next.js

```tsx
import { StockPredictionClient } from '@/lib/stockPredictionClient';
import StockPredictionWidget from '@/components/StockPredictionWidget';

// Option A: Use the widget directly
function StockPage() {
  return (
    <StockPredictionWidget
      initialTicker="AAPL"
      showHistorical
      onPrediction={(pred) => console.log('Prediction:', pred)}
    />
  );
}

// Option B: Use the client directly
const client = new StockPredictionClient('https://api.frameworx.site');

async function getPrediction() {
  const prediction = await client.predict('GOOGL', {
    period: '1y',
    includeHistorical: true
  });
  console.log(`Predicted close: $${prediction.close.toFixed(2)}`);
}

// Option C: Use the React hook
import { useStockPrediction } from '@/components/StockPredictionWidget';

function PredictionDisplay({ ticker }: { ticker: string }) {
  const { prediction, loading, error } = useStockPrediction(ticker);

  if (loading) return <div>Training model...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <h2>{prediction?.ticker}</h2>
      <p>Predicted Close: ${prediction?.close.toFixed(2)}</p>
    </div>
  );
}
```

### Option 4: Direct Python Integration

If frameworx.site has a Python backend, import the prediction module directly:

```python
from stock_predictor import StockPredictor, quick_predict

# Quick prediction
result = quick_predict("AAPL")
print(f"Predicted close: ${result['close']:.2f}")

# Or use the class for more control
predictor = StockPredictor(epochs=10, sequence_length=60)
prediction = predictor.predict("GOOGL", period="2y", include_historical=True)

print(f"""
Ticker: {prediction.ticker}
Predicted Close: ${prediction.close:.2f}
Confidence: {prediction.confidence}
RMSE: {prediction.rmse:.4f}
""")
```

## Integration with Neural Dashboard

Based on your frameworx.site Neural Dashboard (shown in screenshots), here's how to add stock predictions:

### Add to Market Command Center

```tsx
// In your Neural Dashboard component
import { useStockPrediction } from '@/lib/StockPredictionWidget';

function MarketCommandCenter() {
  // Add predictions for tracked stocks
  const applePredict = useStockPrediction('AAPL');
  const googlePredict = useStockPrediction('GOOGL');

  return (
    <div className="command-center">
      {/* Existing crypto cards */}
      <CryptoCard symbol="BTC" />
      <CryptoCard symbol="ETH" />

      {/* Add stock prediction cards */}
      <StockPredictionCard
        ticker="AAPL"
        prediction={applePredict.prediction}
        loading={applePredict.loading}
      />
    </div>
  );
}
```

### Add AI Predictions Tab

In your Neural Market Command Center, add an "AI Predictions" tab:

```tsx
<Tabs defaultValue="command">
  <TabsList>
    <TabsTrigger value="command">Command View</TabsTrigger>
    <TabsTrigger value="correlation">Correlation</TabsTrigger>
    <TabsTrigger value="predictions">AI Predictions</TabsTrigger> {/* New */}
  </TabsList>

  <TabsContent value="predictions">
    <StockPredictionWidget initialTicker="AAPL" showInput />
  </TabsContent>
</Tabs>
```

### Integrate with Stock Screener

Add predictions to your screener results:

```tsx
// Enhance screener with AI predictions
async function enhanceScreenerResults(stocks: Stock[]) {
  const client = new StockPredictionClient();
  const tickers = stocks.map(s => s.ticker);

  const { predictions } = await client.predictBatch(tickers);

  return stocks.map(stock => ({
    ...stock,
    aiPrediction: predictions[stock.ticker],
    predictedChange: predictions[stock.ticker]
      ? ((predictions[stock.ticker].close - stock.price) / stock.price * 100)
      : null
  }));
}
```

## Environment Variables

Set these in your frameworx.site project:

```env
# .env.local
REACT_APP_PREDICTION_API_URL=https://api.frameworx.site
# or
NEXT_PUBLIC_PREDICTION_API_URL=https://api.frameworx.site
```

## Performance Considerations

1. **Caching**: The API caches predictions. Use `useCache: false` for fresh predictions.
2. **Timeout**: Model training takes 10-30 seconds. Set appropriate timeouts.
3. **Batch requests**: Use `/api/predict/batch` for multiple stocks to reduce overhead.
4. **Rate limiting**: Consider adding rate limiting in production.

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    frameworx.site                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Neural    │  │   Stock     │  │    AI Insights      │ │
│  │  Dashboard  │  │  Screener   │  │       Page          │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │ StockPredictionClient │                      │
│              └───────────┬───────────┘                      │
└──────────────────────────┼──────────────────────────────────┘
                           │ HTTPS
                           ▼
              ┌───────────────────────┐
              │  Stock Prediction API │
              │    (FastAPI/uvicorn)  │
              │   api.frameworx.site  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Yahoo Finance API   │
              │   (Historical Data)   │
              └───────────────────────┘
```

## Files Reference

| File | Purpose |
|------|---------|
| `src/api/stock_predictor.py` | Core LSTM prediction module |
| `src/api/main.py` | FastAPI endpoints |
| `src/client/stockPredictionClient.ts` | TypeScript API client |
| `src/client/StockPredictionWidget.tsx` | React component |
| `src/streamlit_app.py` | Standalone Streamlit app |

## Support

- API Documentation: `/api/docs` (Swagger UI)
- Streamlit Demo: https://frameworx.streamlit.app
- HuggingFace Space: https://huggingface.co/spaces/iofocus/stocks_price_prediction
