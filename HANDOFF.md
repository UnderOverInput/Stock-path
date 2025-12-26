# Stock Prediction API Integration Guide

## Overview

Integrate LSTM-based stock price predictions into frameworx.site from a private Hugging Face Space.

## API Details

**Base URL:** `https://iofocus-stocks-price-prediction.hf.space`

**Authentication:** Private Space - requires HF token in header

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict/{ticker}` | GET | Get next-day price prediction |
| `/api/health` | GET | Health check |

## Environment Variable

Add to `.env.local`:
```
NEXT_PUBLIC_HF_TOKEN=<your-huggingface-token>
```

Generate a token at https://huggingface.co/settings/tokens with read access to the Space.

## Response Format

```typescript
interface StockPrediction {
  ticker: string;      // "AAPL"
  open: number;        // 242.17
  high: number;        // 254.01
  low: number;         // 221.01
  close: number;       // 225.89
  volume: number;      // 23240906
  rmse: number;        // 0.2649 (model error)
  confidence: string;  // "High" | "Medium" | "Low"
}
```

## Option 1: Simple Fetch (Quick Integration)

```typescript
async function getStockPrediction(ticker: string) {
  const response = await fetch(
    `https://iofocus-stocks-price-prediction.hf.space/api/predict/${ticker}`,
    {
      headers: {
        'Authorization': `Bearer ${process.env.NEXT_PUBLIC_HF_TOKEN}`
      }
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch prediction for ${ticker}`);
  }

  return response.json();
}

// Usage
const prediction = await getStockPrediction('AAPL');
console.log(`Predicted close: $${prediction.close}`);
```

## Option 2: TypeScript Client (Full Featured)

Create `lib/stockPredictionClient.ts`:

```typescript
export interface StockPrediction {
  ticker: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rmse: number | null;
  confidence: 'High' | 'Medium' | 'Low' | null;
}

export interface ClientConfig {
  baseUrl?: string;
  timeout?: number;
  authToken?: string;
}

export class StockPredictionClient {
  private baseUrl: string;
  private defaultTimeout: number;
  private authToken?: string;

  constructor(config: string | ClientConfig = 'http://localhost:8000') {
    if (typeof config === 'string') {
      this.baseUrl = config.replace(/\/$/, '');
      this.defaultTimeout = 120000;
    } else {
      this.baseUrl = (config.baseUrl || 'http://localhost:8000').replace(/\/$/, '');
      this.defaultTimeout = config.timeout || 120000;
      this.authToken = config.authToken;
    }
  }

  setAuthToken(token: string | undefined): void {
    this.authToken = token;
  }

  private async request<T>(endpoint: string, timeout: number = this.defaultTimeout): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        signal: controller.signal,
        headers,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }

      return response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async predict(ticker: string): Promise<StockPrediction> {
    return this.request<StockPrediction>(`/api/predict/${ticker.toUpperCase()}`);
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.request('/api/health', 5000);
      return true;
    } catch {
      return false;
    }
  }
}

// Create singleton instance
export const stockPredictionClient = new StockPredictionClient({
  baseUrl: 'https://iofocus-stocks-price-prediction.hf.space',
  authToken: process.env.NEXT_PUBLIC_HF_TOKEN,
});
```

## Option 3: React Hook

Create `hooks/useStockPrediction.ts`:

```typescript
import { useState, useEffect, useCallback } from 'react';
import { stockPredictionClient, StockPrediction } from '@/lib/stockPredictionClient';

export function useStockPrediction(ticker: string | null) {
  const [prediction, setPrediction] = useState<StockPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPrediction = useCallback(async () => {
    if (!ticker) {
      setPrediction(null);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await stockPredictionClient.predict(ticker);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch prediction');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  useEffect(() => {
    fetchPrediction();
  }, [fetchPrediction]);

  return { prediction, loading, error, refetch: fetchPrediction };
}
```

## Integration Examples

### Stock Screener Enhancement

Add AI prediction column to screener results:

```typescript
// In your Stock Screener component
import { useStockPrediction } from '@/hooks/useStockPrediction';

function StockRow({ ticker, currentPrice }: { ticker: string; currentPrice: number }) {
  const { prediction, loading } = useStockPrediction(ticker);

  const predictedChange = prediction
    ? ((prediction.close - currentPrice) / currentPrice * 100).toFixed(2)
    : null;

  return (
    <tr>
      <td>{ticker}</td>
      <td>${currentPrice}</td>
      <td>
        {loading ? 'Loading...' : prediction ? (
          <span className={predictedChange > 0 ? 'text-green-500' : 'text-red-500'}>
            ${prediction.close.toFixed(2)} ({predictedChange}%)
          </span>
        ) : 'N/A'}
      </td>
      <td>{prediction?.confidence}</td>
    </tr>
  );
}
```

### Neural Dashboard Card

```typescript
function AIPredictionCard({ ticker }: { ticker: string }) {
  const { prediction, loading, error } = useStockPrediction(ticker);

  if (loading) return <div className="animate-pulse">Training model...</div>;
  if (error) return <div className="text-red-500">{error}</div>;
  if (!prediction) return null;

  return (
    <div className="bg-slate-800 rounded-xl p-4">
      <h3 className="text-lg font-bold">{prediction.ticker} AI Prediction</h3>
      <div className="grid grid-cols-2 gap-2 mt-2">
        <div>
          <span className="text-slate-400 text-sm">Predicted Close</span>
          <p className="text-xl text-green-400">${prediction.close.toFixed(2)}</p>
        </div>
        <div>
          <span className="text-slate-400 text-sm">Confidence</span>
          <p className={`text-xl ${
            prediction.confidence === 'High' ? 'text-green-400' :
            prediction.confidence === 'Medium' ? 'text-yellow-400' : 'text-red-400'
          }`}>{prediction.confidence}</p>
        </div>
      </div>
    </div>
  );
}
```

## Important Notes

1. **Timeout:** Predictions take 10-30 seconds (model trains on-the-fly)
2. **Rate Limiting:** Be mindful of HF Space limits on free tier
3. **Caching:** Consider caching predictions client-side to reduce API calls
4. **Token Security:** Store HF token in environment variables, never commit to repo

## Test Command

```bash
curl -H "Authorization: Bearer $NEXT_PUBLIC_HF_TOKEN" \
  https://iofocus-stocks-price-prediction.hf.space/api/predict/AAPL
```

## Source Repository

Full client code available at: `Stock-path/src/client/stockPredictionClient.ts`
