/**
 * Stock Prediction API Client for frameworx.site
 * TypeScript/JavaScript client for integrating LSTM stock predictions
 */

export interface StockPrediction {
  ticker: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rmse: number | null;
  confidence: 'High' | 'Medium' | 'Low' | null;
  prediction_date: string | null;
  historical_data?: HistoricalDataPoint[];
  cached: boolean;
}

export interface HistoricalDataPoint {
  Date: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
}

export interface BatchPredictionResponse {
  predictions: Record<string, StockPrediction | null>;
  success_count: number;
  failed_tickers: string[];
}

export interface SupportedTickers {
  popular: string[];
  tech: string[];
  crypto_related: string[];
  finance: string[];
}

export interface PredictionOptions {
  period?: '1mo' | '3mo' | '6mo' | '1y' | '2y';
  includeHistorical?: boolean;
  useCache?: boolean;
}

export class StockPredictionClient {
  private baseUrl: string;
  private defaultTimeout: number;

  /**
   * Create a new Stock Prediction API client
   * @param baseUrl - API base URL (e.g., "https://api.frameworx.site" or "http://localhost:8000")
   * @param timeout - Request timeout in milliseconds (default: 120000 for model training)
   */
  constructor(baseUrl: string = 'http://localhost:8000', timeout: number = 120000) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.defaultTimeout = timeout;
  }

  /**
   * Make an API request with timeout
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    timeout: number = this.defaultTimeout
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
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

  /**
   * Get stock price prediction for a single ticker
   * @param ticker - Stock symbol (e.g., "AAPL", "GOOGL")
   * @param options - Prediction options
   * @returns Promise<StockPrediction>
   *
   * @example
   * const client = new StockPredictionClient('https://api.frameworx.site');
   * const prediction = await client.predict('AAPL');
   * console.log(`Predicted close: $${prediction.close.toFixed(2)}`);
   */
  async predict(ticker: string, options: PredictionOptions = {}): Promise<StockPrediction> {
    const params = new URLSearchParams();
    if (options.period) params.set('period', options.period);
    if (options.includeHistorical !== undefined) {
      params.set('include_historical', String(options.includeHistorical));
    }
    if (options.useCache !== undefined) {
      params.set('use_cache', String(options.useCache));
    }

    const queryString = params.toString();
    const endpoint = `/api/predict/${ticker.toUpperCase()}${queryString ? `?${queryString}` : ''}`;

    return this.request<StockPrediction>(endpoint);
  }

  /**
   * Get predictions for multiple stocks
   * @param tickers - Array of stock symbols
   * @param period - Historical data period
   * @returns Promise<BatchPredictionResponse>
   *
   * @example
   * const results = await client.predictBatch(['AAPL', 'GOOGL', 'MSFT']);
   * Object.entries(results.predictions).forEach(([ticker, pred]) => {
   *   if (pred) console.log(`${ticker}: $${pred.close.toFixed(2)}`);
   * });
   */
  async predictBatch(
    tickers: string[],
    period: PredictionOptions['period'] = '1y'
  ): Promise<BatchPredictionResponse> {
    return this.request<BatchPredictionResponse>('/api/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ tickers, period }),
    });
  }

  /**
   * Get list of commonly supported stock tickers
   */
  async getSupportedTickers(): Promise<SupportedTickers> {
    return this.request<SupportedTickers>('/api/supported-tickers');
  }

  /**
   * Check if the API is healthy
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.request('/api/health', {}, 5000);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Clear the prediction cache
   */
  async clearCache(): Promise<void> {
    await this.request('/api/cache', { method: 'DELETE' });
  }
}

/**
 * React Hook for stock predictions (for use with frameworx.site React components)
 *
 * @example
 * // In a React component:
 * const { prediction, loading, error, refetch } = useStockPrediction('AAPL');
 *
 * if (loading) return <Spinner />;
 * if (error) return <Error message={error} />;
 * return <PriceDisplay close={prediction?.close} />;
 */
export function createStockPredictionHook(client: StockPredictionClient) {
  return function useStockPrediction(ticker: string, options?: PredictionOptions) {
    // This is a factory function - actual React implementation would be:
    // const [prediction, setPrediction] = useState<StockPrediction | null>(null);
    // const [loading, setLoading] = useState(false);
    // const [error, setError] = useState<string | null>(null);
    //
    // useEffect(() => {
    //   if (!ticker) return;
    //   setLoading(true);
    //   client.predict(ticker, options)
    //     .then(setPrediction)
    //     .catch(e => setError(e.message))
    //     .finally(() => setLoading(false));
    // }, [ticker, JSON.stringify(options)]);
    //
    // return { prediction, loading, error, refetch: () => {...} };

    return {
      hookFactory: true,
      message: 'Use this in your React app with useState/useEffect'
    };
  };
}

// Export default instance for quick usage
export const defaultClient = new StockPredictionClient();

// Convenience function
export async function quickPredict(ticker: string): Promise<StockPrediction> {
  return defaultClient.predict(ticker);
}
