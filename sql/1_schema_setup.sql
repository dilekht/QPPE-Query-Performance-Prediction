-- =================================================================
-- QPPE Database Schema Setup
-- File: 1_schema_setup.sql
-- =================================================================

-- Training data table
CREATE TABLE IF NOT EXISTS qppe_training_data (
    id SERIAL PRIMARY KEY,
    query_hash TEXT NOT NULL,
    num_joins INT DEFAULT 0,
    num_relations INT DEFAULT 0,
    join_depth INT DEFAULT 0,
    est_rows BIGINT DEFAULT 0,
    est_cost FLOAT DEFAULT 0.0,
    join_types TEXT,
    scan_types TEXT,
    has_subquery BOOLEAN DEFAULT FALSE,
    has_aggregation BOOLEAN DEFAULT FALSE,
    actual_time FLOAT NOT NULL,
    performance_class INT NOT NULL CHECK (performance_class IN (0, 1, 2)),
    execution_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_actual_time CHECK (actual_time >= 0)
);

CREATE INDEX idx_qppe_query_hash ON qppe_training_data(query_hash);
CREATE INDEX idx_qppe_perf_class ON qppe_training_data(performance_class);
CREATE INDEX idx_qppe_timestamp ON qppe_training_data(execution_timestamp);

-- Model metadata
CREATE TABLE IF NOT EXISTS qppe_model_metadata (
    id SERIAL PRIMARY KEY,
    model_version INT NOT NULL,
    training_samples INT,
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE
);

-- Configuration
CREATE TABLE IF NOT EXISTS qppe_config (
    parameter TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO qppe_config (parameter, value, description) VALUES 
    ('enabled', 'true', 'Enable/disable QPPE optimization'),
    ('penalty_fast', '0.95', 'Cost multiplier for fast queries'),
    ('penalty_medium', '1.0', 'Cost multiplier for medium queries'),
    ('penalty_slow', '1.3', 'Cost multiplier for slow queries'),
    ('prediction_timeout_ms', '5', 'Timeout for predictions in ms'),
    ('retrain_threshold', '1000', 'Min samples before retraining'),
    ('fast_threshold_ms', '100', 'Threshold for fast classification'),
    ('slow_threshold_ms', '1000', 'Threshold for slow classification')
ON CONFLICT (parameter) DO NOTHING;

-- Helper function for performance classification
CREATE OR REPLACE FUNCTION classify_performance(exec_time_ms FLOAT)
RETURNS INT AS $$
DECLARE
    fast_threshold FLOAT;
    slow_threshold FLOAT;
BEGIN
    SELECT value::FLOAT INTO fast_threshold 
    FROM qppe_config WHERE parameter = 'fast_threshold_ms';
    
    SELECT value::FLOAT INTO slow_threshold 
    FROM qppe_config WHERE parameter = 'slow_threshold_ms';
    
    IF exec_time_ms < fast_threshold THEN
        RETURN 0;  -- Fast
    ELSIF exec_time_ms < slow_threshold THEN
        RETURN 1;  -- Medium
    ELSE
        RETURN 2;  -- Slow
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Stats view
CREATE OR REPLACE VIEW qppe_training_stats AS
SELECT 
    COUNT(*) as total_samples,
    COUNT(CASE WHEN performance_class = 0 THEN 1 END) as fast_count,
    COUNT(CASE WHEN performance_class = 1 THEN 1 END) as medium_count,
    COUNT(CASE WHEN performance_class = 2 THEN 1 END) as slow_count,
    AVG(actual_time) as avg_execution_time,
    MIN(execution_timestamp) as first_sample,
    MAX(execution_timestamp) as last_sample
FROM qppe_training_data;
