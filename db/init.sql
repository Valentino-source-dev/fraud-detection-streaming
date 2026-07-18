-- ============================================================
-- Fraud Detection Streaming Pipeline — Database Schema
-- ============================================================

-- Predictions table: every inference result from the consumer
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    original_index  INTEGER,
    amount          DECIMAL(12, 2),
    score           FLOAT NOT NULL,
    is_fraud_pred   BOOLEAN NOT NULL,
    is_fraud_actual BOOLEAN,
    model_version   VARCHAR(50),
    latency_ms      FLOAT,
    features_json   JSONB
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_is_fraud_pred ON predictions (is_fraud_pred);
CREATE INDEX IF NOT EXISTS idx_predictions_score ON predictions (score);

-- Materialized view for hourly aggregation (useful for Grafana)
-- Refresh manually or via cron: REFRESH MATERIALIZED VIEW hourly_fraud_stats;
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_fraud_stats AS
SELECT
    date_trunc('hour', created_at)              AS hour,
    COUNT(*)                                     AS total_predictions,
    COUNT(*) FILTER (WHERE is_fraud_pred)        AS fraud_predictions,
    COUNT(*) FILTER (WHERE is_fraud_actual)      AS actual_frauds,
    AVG(score)                                   AS avg_score,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency_ms,
    AVG(latency_ms)                              AS avg_latency_ms
FROM predictions
GROUP BY 1
ORDER BY 1;

-- Model deployment log
CREATE TABLE IF NOT EXISTS model_deployments (
    id            SERIAL PRIMARY KEY,
    deployed_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name    VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    stage         VARCHAR(20) NOT NULL,
    metrics_json  JSONB
);
