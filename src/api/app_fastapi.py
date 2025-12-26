"""
Hugging Face Spaces - Pure FastAPI (no Gradio dependency issues)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from torch.utils.data import TensorDataset, DataLoader

# LSTM Model
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=5, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.relu(self.fc1(out[:, -1, :]))
        out = torch.relu(self.fc2(out))
        return self.fc3(out)


def predict_stock(ticker: str) -> dict:
    """Core prediction function"""
    ticker = ticker.upper().strip()

    try:
        data = yf.download(ticker, period='1y', progress=False)[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.dropna(inplace=True)
    except Exception as e:
        return {"error": f"Failed to fetch data for {ticker}: {str(e)}"}

    if len(data) < 60:
        return {"error": f"Insufficient data for {ticker}. Need at least 60 days."}

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = StockPriceLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

    model.train()
    for _ in range(5):
        for bx, by in loader:
            loss = criterion(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        recent = yf.download(ticker, period='70d', progress=False)[['Open', 'High', 'Low', 'Close', 'Volume']]
        last_60 = scaler.transform(recent.dropna()[-60:])
        input_t = torch.tensor(last_60, dtype=torch.float32).unsqueeze(0)
        pred_scaled = model(input_t)
        pred = scaler.inverse_transform(pred_scaled.numpy())[0]

    with torch.no_grad():
        preds = model(X_t[-100:]).numpy()
        rmse = float(np.sqrt(np.mean((preds - y[-100:]) ** 2)))

    return {
        "ticker": ticker,
        "open": round(float(pred[0]), 2),
        "high": round(float(pred[1]), 2),
        "low": round(float(pred[2]), 2),
        "close": round(float(pred[3]), 2),
        "volume": int(pred[4]),
        "rmse": round(rmse, 4),
        "confidence": "High" if rmse < 0.05 else "Medium" if rmse < 0.1 else "Low"
    }


# FastAPI App
app = FastAPI(title="Stock Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple HTML UI
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Stock Price Predictor</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 30px; }
        .input-group { display: flex; gap: 10px; margin-bottom: 20px; }
        input {
            flex: 1; padding: 15px; font-size: 18px;
            border: none; border-radius: 8px;
            background: #0f3460; color: #fff;
        }
        button {
            padding: 15px 30px; font-size: 18px;
            background: #e94560; color: #fff;
            border: none; border-radius: 8px; cursor: pointer;
        }
        button:hover { background: #ff6b6b; }
        button:disabled { background: #666; cursor: wait; }
        .result {
            background: #0f3460; border-radius: 12px;
            padding: 20px; margin-top: 20px;
        }
        .result h2 { margin-top: 0; color: #e94560; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .metric { background: #16213e; padding: 15px; border-radius: 8px; }
        .metric-label { color: #888; font-size: 14px; }
        .metric-value { font-size: 24px; font-weight: bold; margin-top: 5px; }
        .confidence { text-align: center; margin-top: 20px; padding: 10px; border-radius: 8px; }
        .confidence.High { background: #00b894; }
        .confidence.Medium { background: #fdcb6e; color: #000; }
        .confidence.Low { background: #d63031; }
        .error { background: #d63031; padding: 15px; border-radius: 8px; }
        .loading { text-align: center; padding: 40px; }
        .api-info { margin-top: 40px; padding: 20px; background: #0f3460; border-radius: 12px; }
        code { background: #16213e; padding: 2px 8px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 LSTM Stock Price Predictor</h1>

        <div class="input-group">
            <input type="text" id="ticker" placeholder="Enter ticker (e.g., AAPL)" value="AAPL">
            <button onclick="predict()" id="btn">Predict</button>
        </div>

        <div id="result"></div>

        <div class="api-info">
            <h3>API Access</h3>
            <p>Call the API directly from your app:</p>
            <code>GET /api/predict/{ticker}</code>
            <p style="margin-top:10px">Example: <code>/api/predict/AAPL</code></p>
        </div>
    </div>

    <script>
        async function predict() {
            const ticker = document.getElementById('ticker').value.toUpperCase();
            const btn = document.getElementById('btn');
            const result = document.getElementById('result');

            btn.disabled = true;
            btn.textContent = 'Predicting...';
            result.innerHTML = '<div class="loading">Training LSTM model... (10-30 seconds)</div>';

            try {
                const res = await fetch(`/api/predict/${ticker}`);
                const data = await res.json();

                if (data.error) {
                    result.innerHTML = `<div class="error">${data.error}</div>`;
                } else {
                    result.innerHTML = `
                        <div class="result">
                            <h2>${data.ticker} - Next Day Prediction</h2>
                            <div class="grid">
                                <div class="metric">
                                    <div class="metric-label">Open</div>
                                    <div class="metric-value">$${data.open.toLocaleString()}</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-label">Close</div>
                                    <div class="metric-value" style="color:#00b894">$${data.close.toLocaleString()}</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-label">High</div>
                                    <div class="metric-value">$${data.high.toLocaleString()}</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-label">Low</div>
                                    <div class="metric-value">$${data.low.toLocaleString()}</div>
                                </div>
                            </div>
                            <div class="confidence ${data.confidence}">
                                ${data.confidence} Confidence (RMSE: ${data.rmse})
                            </div>
                        </div>
                    `;
                }
            } catch (e) {
                result.innerHTML = `<div class="error">Error: ${e.message}</div>`;
            }

            btn.disabled = false;
            btn.textContent = 'Predict';
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE

@app.get("/api/predict/{ticker}")
async def api_predict(ticker: str):
    result = predict_stock(ticker)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/api/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
