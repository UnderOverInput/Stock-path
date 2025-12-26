-- Supabase Schema for Stock Predictions
-- Run this in Supabase SQL Editor or as a migration

-- Main predictions cache table
CREATE TABLE IF NOT EXISTS stock_predictions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  ticker VARCHAR(10) NOT NULL UNIQUE,
  open DECIMAL(12, 2) NOT NULL,
  high DECIMAL(12, 2) NOT NULL,
  low DECIMAL(12, 2) NOT NULL,
  close DECIMAL(12, 2) NOT NULL,
  volume DECIMAL(20, 0) NOT NULL,
  rmse DECIMAL(10, 6),
  confidence VARCHAR(10) CHECK (confidence IN ('High', 'Medium', 'Low')),
  prediction_date DATE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,

  -- Index for faster lookups
  CONSTRAINT valid_prices CHECK (
    open > 0 AND high > 0 AND low > 0 AND close > 0 AND volume >= 0
  )
);

-- Index for cache lookups
CREATE INDEX idx_predictions_ticker_expires ON stock_predictions(ticker, expires_at);

-- Historical predictions (for tracking accuracy over time)
CREATE TABLE IF NOT EXISTS stock_predictions_history (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  ticker VARCHAR(10) NOT NULL,
  open DECIMAL(12, 2) NOT NULL,
  high DECIMAL(12, 2) NOT NULL,
  low DECIMAL(12, 2) NOT NULL,
  close DECIMAL(12, 2) NOT NULL,
  volume DECIMAL(20, 0) NOT NULL,
  rmse DECIMAL(10, 6),
  confidence VARCHAR(10),
  prediction_date DATE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_history_ticker ON stock_predictions_history(ticker, created_at DESC);

-- User watchlist
CREATE TABLE IF NOT EXISTS user_watchlist (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  ticker VARCHAR(10) NOT NULL,
  added_at TIMESTAMPTZ DEFAULT NOW(),
  notes TEXT,

  UNIQUE(user_id, ticker)
);

CREATE INDEX idx_watchlist_user ON user_watchlist(user_id);

-- Prediction feedback (for tracking accuracy)
CREATE TABLE IF NOT EXISTS prediction_feedback (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  ticker VARCHAR(10) NOT NULL,
  prediction_date DATE NOT NULL,
  predicted_close DECIMAL(12, 2) NOT NULL,
  actual_close DECIMAL(12, 2) NOT NULL,
  accuracy_percentage DECIMAL(5, 2),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feedback_ticker ON prediction_feedback(ticker, prediction_date);

-- Function to auto-archive predictions before update
CREATE OR REPLACE FUNCTION archive_prediction()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO stock_predictions_history (
    ticker, open, high, low, close, volume, rmse, confidence, prediction_date
  )
  VALUES (
    OLD.ticker, OLD.open, OLD.high, OLD.low, OLD.close, OLD.volume,
    OLD.rmse, OLD.confidence, OLD.prediction_date
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to archive before update
CREATE TRIGGER archive_before_update
BEFORE UPDATE ON stock_predictions
FOR EACH ROW
EXECUTE FUNCTION archive_prediction();

-- Function to clean expired predictions
CREATE OR REPLACE FUNCTION clean_expired_predictions()
RETURNS void AS $$
BEGIN
  DELETE FROM stock_predictions WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- RLS (Row Level Security) Policies

-- Enable RLS
ALTER TABLE stock_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE stock_predictions_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_feedback ENABLE ROW LEVEL SECURITY;

-- Public read access to predictions (anyone can view)
CREATE POLICY "Public read access" ON stock_predictions
  FOR SELECT USING (true);

CREATE POLICY "Public read history" ON stock_predictions_history
  FOR SELECT USING (true);

-- Service role can insert/update predictions
CREATE POLICY "Service role insert" ON stock_predictions
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Service role update" ON stock_predictions
  FOR UPDATE USING (true);

-- Users can only access their own watchlist
CREATE POLICY "Users own watchlist" ON user_watchlist
  FOR ALL USING (auth.uid() = user_id);

-- Anyone can submit feedback
CREATE POLICY "Public feedback" ON prediction_feedback
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Public read feedback" ON prediction_feedback
  FOR SELECT USING (true);

-- View for prediction accuracy stats
CREATE OR REPLACE VIEW prediction_accuracy_stats AS
SELECT
  ticker,
  COUNT(*) as total_predictions,
  AVG(accuracy_percentage) as avg_accuracy,
  MIN(accuracy_percentage) as min_accuracy,
  MAX(accuracy_percentage) as max_accuracy,
  STDDEV(accuracy_percentage) as accuracy_stddev
FROM prediction_feedback
GROUP BY ticker;

-- Grant access to the view
GRANT SELECT ON prediction_accuracy_stats TO anon, authenticated;
