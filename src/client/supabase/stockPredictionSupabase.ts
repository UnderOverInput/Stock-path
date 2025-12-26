/**
 * Stock Prediction Client with Supabase Caching
 * For frameworx.site integration
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';

// Types
export interface StockPrediction {
  ticker: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rmse: number | null;
  confidence: 'High' | 'Medium' | 'Low' | null;
  prediction_date: string;
}

interface CachedPrediction extends StockPrediction {
  id: string;
  created_at: string;
  expires_at: string;
}

// Configuration
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const PREDICTION_API_URL = process.env.NEXT_PUBLIC_PREDICTION_API_URL || 'http://localhost:8000';

// Cache duration in hours
const CACHE_DURATION_HOURS = 4;

export class StockPredictionSupabaseClient {
  private supabase: SupabaseClient;
  private apiUrl: string;

  constructor(
    supabaseUrl: string = SUPABASE_URL,
    supabaseKey: string = SUPABASE_ANON_KEY,
    predictionApiUrl: string = PREDICTION_API_URL
  ) {
    this.supabase = createClient(supabaseUrl, supabaseKey);
    this.apiUrl = predictionApiUrl;
  }

  /**
   * Get prediction with Supabase caching
   */
  async predict(ticker: string): Promise<StockPrediction> {
    ticker = ticker.toUpperCase().trim();

    // 1. Check Supabase cache first
    const cached = await this.getCachedPrediction(ticker);
    if (cached) {
      console.log(`Cache hit for ${ticker}`);
      return cached;
    }

    // 2. Fetch fresh prediction from API
    console.log(`Cache miss for ${ticker}, fetching from API...`);
    const prediction = await this.fetchFromApi(ticker);

    // 3. Store in Supabase cache
    await this.cachePrediction(prediction);

    return prediction;
  }

  /**
   * Check Supabase for cached prediction
   */
  private async getCachedPrediction(ticker: string): Promise<StockPrediction | null> {
    const { data, error } = await this.supabase
      .from('stock_predictions')
      .select('*')
      .eq('ticker', ticker)
      .gt('expires_at', new Date().toISOString())
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    if (error || !data) {
      return null;
    }

    return {
      ticker: data.ticker,
      open: data.open,
      high: data.high,
      low: data.low,
      close: data.close,
      volume: data.volume,
      rmse: data.rmse,
      confidence: data.confidence,
      prediction_date: data.prediction_date,
    };
  }

  /**
   * Fetch prediction from FastAPI backend
   */
  private async fetchFromApi(ticker: string): Promise<StockPrediction> {
    const response = await fetch(`${this.apiUrl}/api/predict/${ticker}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Failed to fetch prediction for ${ticker}`);
    }

    return response.json();
  }

  /**
   * Store prediction in Supabase cache
   */
  private async cachePrediction(prediction: StockPrediction): Promise<void> {
    const expiresAt = new Date();
    expiresAt.setHours(expiresAt.getHours() + CACHE_DURATION_HOURS);

    const { error } = await this.supabase.from('stock_predictions').upsert(
      {
        ticker: prediction.ticker,
        open: prediction.open,
        high: prediction.high,
        low: prediction.low,
        close: prediction.close,
        volume: prediction.volume,
        rmse: prediction.rmse,
        confidence: prediction.confidence,
        prediction_date: prediction.prediction_date,
        expires_at: expiresAt.toISOString(),
      },
      {
        onConflict: 'ticker',
      }
    );

    if (error) {
      console.error('Failed to cache prediction:', error);
    }
  }

  /**
   * Get prediction history for a ticker
   */
  async getPredictionHistory(ticker: string, limit: number = 10): Promise<CachedPrediction[]> {
    const { data, error } = await this.supabase
      .from('stock_predictions_history')
      .select('*')
      .eq('ticker', ticker.toUpperCase())
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) {
      console.error('Failed to fetch history:', error);
      return [];
    }

    return data || [];
  }

  /**
   * Get user's watchlist predictions
   */
  async getWatchlistPredictions(userId: string): Promise<StockPrediction[]> {
    // Get user's watchlist from Supabase
    const { data: watchlist } = await this.supabase
      .from('user_watchlist')
      .select('ticker')
      .eq('user_id', userId);

    if (!watchlist || watchlist.length === 0) {
      return [];
    }

    // Get predictions for all watchlist items
    const predictions = await Promise.all(
      watchlist.map((item) => this.predict(item.ticker).catch(() => null))
    );

    return predictions.filter((p): p is StockPrediction => p !== null);
  }

  /**
   * Add ticker to user's watchlist
   */
  async addToWatchlist(userId: string, ticker: string): Promise<void> {
    const { error } = await this.supabase.from('user_watchlist').upsert({
      user_id: userId,
      ticker: ticker.toUpperCase(),
    });

    if (error) {
      throw new Error('Failed to add to watchlist');
    }
  }

  /**
   * Save prediction feedback (for model improvement)
   */
  async savePredictionFeedback(
    ticker: string,
    predictionDate: string,
    actualClose: number,
    predictedClose: number
  ): Promise<void> {
    const accuracy = 1 - Math.abs(actualClose - predictedClose) / actualClose;

    await this.supabase.from('prediction_feedback').insert({
      ticker,
      prediction_date: predictionDate,
      predicted_close: predictedClose,
      actual_close: actualClose,
      accuracy_percentage: accuracy * 100,
    });
  }
}

// Export singleton instance
export const predictionClient = new StockPredictionSupabaseClient();

// Convenience function
export async function quickPredict(ticker: string): Promise<StockPrediction> {
  return predictionClient.predict(ticker);
}
