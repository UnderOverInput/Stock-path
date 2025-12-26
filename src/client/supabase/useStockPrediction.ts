/**
 * React Hooks for Stock Predictions with Supabase
 * For frameworx.site integration
 */

import { useState, useEffect, useCallback } from 'react';
import { useSupabaseClient, useUser } from '@supabase/auth-helpers-react';
import { StockPredictionSupabaseClient, StockPrediction } from './stockPredictionSupabase';

// Initialize client (will use environment variables)
let predictionClient: StockPredictionSupabaseClient | null = null;

function getPredictionClient(): StockPredictionSupabaseClient {
  if (!predictionClient) {
    predictionClient = new StockPredictionSupabaseClient();
  }
  return predictionClient;
}

/**
 * Hook for fetching a single stock prediction
 *
 * @example
 * const { prediction, loading, error, refetch } = useStockPrediction('AAPL');
 */
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
      const client = getPredictionClient();
      const result = await client.predict(ticker);
      setPrediction(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch prediction';
      setError(message);
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  useEffect(() => {
    fetchPrediction();
  }, [fetchPrediction]);

  return {
    prediction,
    loading,
    error,
    refetch: fetchPrediction,
  };
}

/**
 * Hook for fetching multiple stock predictions
 *
 * @example
 * const { predictions, loading } = useMultipleStockPredictions(['AAPL', 'GOOGL', 'MSFT']);
 */
export function useMultipleStockPredictions(tickers: string[]) {
  const [predictions, setPredictions] = useState<Record<string, StockPrediction | null>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPredictions = useCallback(async () => {
    if (!tickers.length) {
      setPredictions({});
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const client = getPredictionClient();
      const results: Record<string, StockPrediction | null> = {};

      await Promise.all(
        tickers.map(async (ticker) => {
          try {
            results[ticker] = await client.predict(ticker);
          } catch {
            results[ticker] = null;
          }
        })
      );

      setPredictions(results);
    } catch (err) {
      setError('Failed to fetch predictions');
    } finally {
      setLoading(false);
    }
  }, [tickers.join(',')]);

  useEffect(() => {
    fetchPredictions();
  }, [fetchPredictions]);

  return {
    predictions,
    loading,
    error,
    refetch: fetchPredictions,
  };
}

/**
 * Hook for user's watchlist with predictions
 *
 * @example
 * const { watchlist, addTicker, removeTicker, loading } = useWatchlistPredictions();
 */
export function useWatchlistPredictions() {
  const supabase = useSupabaseClient();
  const user = useUser();
  const [watchlist, setWatchlist] = useState<Array<{ ticker: string; prediction: StockPrediction | null }>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWatchlist = useCallback(async () => {
    if (!user) {
      setWatchlist([]);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Get user's watchlist
      const { data, error: fetchError } = await supabase
        .from('user_watchlist')
        .select('ticker')
        .eq('user_id', user.id)
        .order('added_at', { ascending: false });

      if (fetchError) throw fetchError;

      // Fetch predictions for each ticker
      const client = getPredictionClient();
      const watchlistWithPredictions = await Promise.all(
        (data || []).map(async (item) => ({
          ticker: item.ticker,
          prediction: await client.predict(item.ticker).catch(() => null),
        }))
      );

      setWatchlist(watchlistWithPredictions);
    } catch (err) {
      setError('Failed to fetch watchlist');
    } finally {
      setLoading(false);
    }
  }, [user, supabase]);

  const addTicker = useCallback(
    async (ticker: string) => {
      if (!user) return;

      const { error } = await supabase.from('user_watchlist').upsert({
        user_id: user.id,
        ticker: ticker.toUpperCase(),
      });

      if (!error) {
        fetchWatchlist();
      }
    },
    [user, supabase, fetchWatchlist]
  );

  const removeTicker = useCallback(
    async (ticker: string) => {
      if (!user) return;

      const { error } = await supabase
        .from('user_watchlist')
        .delete()
        .eq('user_id', user.id)
        .eq('ticker', ticker.toUpperCase());

      if (!error) {
        setWatchlist((prev) => prev.filter((item) => item.ticker !== ticker.toUpperCase()));
      }
    },
    [user, supabase]
  );

  useEffect(() => {
    fetchWatchlist();
  }, [fetchWatchlist]);

  return {
    watchlist,
    loading,
    error,
    addTicker,
    removeTicker,
    refetch: fetchWatchlist,
    isAuthenticated: !!user,
  };
}

/**
 * Hook for prediction history
 *
 * @example
 * const { history, loading } = usePredictionHistory('AAPL');
 */
export function usePredictionHistory(ticker: string, limit: number = 10) {
  const supabase = useSupabaseClient();
  const [history, setHistory] = useState<StockPrediction[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!ticker) return;

    setLoading(true);

    supabase
      .from('stock_predictions_history')
      .select('*')
      .eq('ticker', ticker.toUpperCase())
      .order('created_at', { ascending: false })
      .limit(limit)
      .then(({ data }) => {
        setHistory(data || []);
        setLoading(false);
      });
  }, [ticker, limit, supabase]);

  return { history, loading };
}

/**
 * Hook for real-time prediction updates via Supabase Realtime
 *
 * @example
 * usePredictionSubscription('AAPL', (prediction) => {
 *   console.log('New prediction:', prediction);
 * });
 */
export function usePredictionSubscription(
  ticker: string,
  onUpdate: (prediction: StockPrediction) => void
) {
  const supabase = useSupabaseClient();

  useEffect(() => {
    if (!ticker) return;

    const channel = supabase
      .channel(`prediction-${ticker}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'stock_predictions',
          filter: `ticker=eq.${ticker.toUpperCase()}`,
        },
        (payload) => {
          if (payload.new) {
            onUpdate(payload.new as StockPrediction);
          }
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [ticker, supabase, onUpdate]);
}
