"""
Hugging Face Spaces compatible API
Combines Gradio UI + FastAPI endpoints
"""

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

    # Download data
    data = yf.download(ticker, period='1y', progress=False)[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.dropna(inplace=True)

    if len(data) < 60:
        return {"error": f"Insufficient data for {ticker}"}

    # Scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    # Train model
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

    # Predict
    model.eval()
    with torch.no_grad():
        recent = yf.download(ticker, period='70d', progress=False)[['Open', 'High', 'Low', 'Close', 'Volume']]
        last_60 = scaler.transform(recent.dropna()[-60:])
        input_t = torch.tensor(last_60, dtype=torch.float32).unsqueeze(0)
        pred_scaled = model(input_t)
        pred = scaler.inverse_transform(pred_scaled.numpy())[0]

    # Calculate RMSE
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


def gradio_predict(ticker: str) -> str:
    """Gradio wrapper that returns formatted string"""
    result = predict_stock(ticker)
    if "error" in result:
        return result["error"]
    return f"""
**{result['ticker']} - Next Day Prediction**

| Metric | Value |
|--------|-------|
| Open | ${result['open']:,.2f} |
| High | ${result['high']:,.2f} |
| Low | ${result['low']:,.2f} |
| Close | ${result['close']:,.2f} |
| Volume | {result['volume']:,} |

**Model Confidence:** {result['confidence']} (RMSE: {result['rmse']})
"""


# Create Gradio app
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(label="Stock Ticker", placeholder="AAPL", value="AAPL"),
    outputs=gr.Markdown(label="Prediction"),
    title="🧠 LSTM Stock Price Predictor",
    description="Enter a stock ticker to predict next day's prices using an LSTM neural network.",
    examples=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
    cache_examples=False
)

# Mount FastAPI for API access
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/predict/{ticker}")
async def api_predict(ticker: str):
    """API endpoint for programmatic access"""
    return predict_stock(ticker)

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

# Mount Gradio
app = gr.mount_gradio_app(app, demo, path="/")

# For local testing: uvicorn app:app --reload
