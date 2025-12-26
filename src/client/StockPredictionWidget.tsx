/**
 * Stock Prediction Widget Component for frameworx.site
 * Drop-in React component for displaying LSTM stock predictions
 */

import React, { useState, useEffect, useCallback } from 'react';
import { StockPredictionClient, StockPrediction, PredictionOptions } from './stockPredictionClient';

// Initialize client with your API URL
const API_URL = process.env.REACT_APP_PREDICTION_API_URL || 'http://localhost:8000';
const client = new StockPredictionClient(API_URL);

interface StockPredictionWidgetProps {
  /** Initial ticker symbol */
  initialTicker?: string;
  /** Show input field for ticker */
  showInput?: boolean;
  /** Show historical data */
  showHistorical?: boolean;
  /** Custom styling */
  className?: string;
  /** Callback when prediction completes */
  onPrediction?: (prediction: StockPrediction) => void;
  /** Callback on error */
  onError?: (error: string) => void;
}

/**
 * Stock Prediction Widget
 *
 * @example
 * // Basic usage
 * <StockPredictionWidget initialTicker="AAPL" />
 *
 * @example
 * // With callbacks
 * <StockPredictionWidget
 *   initialTicker="GOOGL"
 *   onPrediction={(pred) => console.log('Got prediction:', pred)}
 *   showHistorical
 * />
 */
export const StockPredictionWidget: React.FC<StockPredictionWidgetProps> = ({
  initialTicker = 'AAPL',
  showInput = true,
  showHistorical = false,
  className = '',
  onPrediction,
  onError,
}) => {
  const [ticker, setTicker] = useState(initialTicker);
  const [inputValue, setInputValue] = useState(initialTicker);
  const [prediction, setPrediction] = useState<StockPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const fetchPrediction = useCallback(async () => {
    if (!ticker) return;

    setLoading(true);
    setError(null);
    setProgress(0);

    // Simulate progress during model training
    const progressInterval = setInterval(() => {
      setProgress((p) => Math.min(p + 10, 90));
    }, 1000);

    try {
      const result = await client.predict(ticker, {
        includeHistorical: showHistorical,
        useCache: true,
      });

      setPrediction(result);
      setProgress(100);
      onPrediction?.(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Prediction failed';
      setError(message);
      onError?.(message);
    } finally {
      clearInterval(progressInterval);
      setLoading(false);
    }
  }, [ticker, showHistorical, onPrediction, onError]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setTicker(inputValue.toUpperCase());
  };

  useEffect(() => {
    fetchPrediction();
  }, [ticker]);

  // Confidence badge color
  const getConfidenceColor = (confidence: string | null) => {
    switch (confidence) {
      case 'High':
        return 'bg-green-500';
      case 'Medium':
        return 'bg-yellow-500';
      case 'Low':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  // Format volume
  const formatVolume = (value: number) => {
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
    return value.toFixed(0);
  };

  return (
    <div className={`stock-prediction-widget rounded-xl bg-slate-800 p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <span className="text-2xl">🧠</span>
          AI Price Prediction
        </h3>
        {prediction?.cached && (
          <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded">
            Cached
          </span>
        )}
      </div>

      {/* Ticker Input */}
      {showInput && (
        <form onSubmit={handleSubmit} className="mb-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value.toUpperCase())}
              placeholder="Enter ticker (e.g., AAPL)"
              className="flex-1 bg-slate-700 text-white px-4 py-2 rounded-lg border border-slate-600 focus:border-blue-500 focus:outline-none"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !inputValue}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white rounded-lg font-medium transition-colors"
            >
              {loading ? 'Predicting...' : 'Predict'}
            </button>
          </div>
        </form>
      )}

      {/* Loading State */}
      {loading && (
        <div className="mb-4">
          <div className="flex justify-between text-sm text-slate-400 mb-1">
            <span>Training LSTM model...</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Prediction Results */}
      {prediction && !loading && (
        <div className="space-y-4">
          {/* Ticker Header */}
          <div className="flex items-center justify-between">
            <div>
              <span className="text-2xl font-bold text-white">{prediction.ticker}</span>
              <span className="ml-2 text-sm text-slate-400">Next Day Prediction</span>
            </div>
            {prediction.confidence && (
              <span
                className={`px-2 py-1 rounded text-xs font-medium text-white ${getConfidenceColor(
                  prediction.confidence
                )}`}
              >
                {prediction.confidence} Confidence
              </span>
            )}
          </div>

          {/* Price Grid */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-700/50 rounded-lg p-3">
              <div className="text-slate-400 text-xs mb-1">Open</div>
              <div className="text-white font-semibold">{formatCurrency(prediction.open)}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-3">
              <div className="text-slate-400 text-xs mb-1">Close</div>
              <div className="text-green-400 font-semibold">{formatCurrency(prediction.close)}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-3">
              <div className="text-slate-400 text-xs mb-1">High</div>
              <div className="text-white font-semibold">{formatCurrency(prediction.high)}</div>
            </div>
            <div className="bg-slate-700/50 rounded-lg p-3">
              <div className="text-slate-400 text-xs mb-1">Low</div>
              <div className="text-white font-semibold">{formatCurrency(prediction.low)}</div>
            </div>
          </div>

          {/* Volume */}
          <div className="bg-slate-700/50 rounded-lg p-3">
            <div className="text-slate-400 text-xs mb-1">Predicted Volume</div>
            <div className="text-white font-semibold">{formatVolume(prediction.volume)}</div>
          </div>

          {/* Model Metrics */}
          {prediction.rmse !== null && (
            <div className="flex items-center justify-between text-sm text-slate-400 pt-2 border-t border-slate-700">
              <span>Model RMSE: {prediction.rmse.toFixed(4)}</span>
              <span>Date: {prediction.prediction_date}</span>
            </div>
          )}

          {/* Historical Data */}
          {showHistorical && prediction.historical_data && (
            <div className="mt-4">
              <h4 className="text-sm font-medium text-slate-300 mb-2">Recent History</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-slate-400">
                      <th className="text-left py-1">Date</th>
                      <th className="text-right py-1">Close</th>
                      <th className="text-right py-1">Volume</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prediction.historical_data.slice(-5).map((row, i) => (
                      <tr key={i} className="text-slate-300">
                        <td className="py-1">{new Date(row.Date).toLocaleDateString()}</td>
                        <td className="text-right">{formatCurrency(row.Close)}</td>
                        <td className="text-right">{formatVolume(row.Volume)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <p className="text-xs text-slate-500 mt-4">
            Predictions are generated using an LSTM neural network trained on historical data.
            This is not financial advice.
          </p>
        </div>
      )}
    </div>
  );
};

/**
 * Custom hook for stock predictions
 */
export function useStockPrediction(ticker: string, options?: PredictionOptions) {
  const [prediction, setPrediction] = useState<StockPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPrediction = useCallback(async () => {
    if (!ticker) return;

    setLoading(true);
    setError(null);

    try {
      const result = await client.predict(ticker, options);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }, [ticker, options?.period, options?.includeHistorical, options?.useCache]);

  useEffect(() => {
    fetchPrediction();
  }, [fetchPrediction]);

  return { prediction, loading, error, refetch: fetchPrediction };
}

export default StockPredictionWidget;
