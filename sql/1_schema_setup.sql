-- =================================================================
-- QPPE Database Schema Setup for Windows/PostgreSQL 18+
-- File: 1_schema_setup.sql
-- =================================================================

-- Training data table with enhanced features
CREATE TABLE IF NOT EXISTS qppe_training_data (
    id SERIAL PRIMARY KEY,
    query_hash TEXT NOT NULL,
    query_template TEXT,
    
    -- Basic features
    num_joins INT DEFAULT 0,
    num_relations INT DEFAULT 0,
    join_depth INT DEFAULT 0,
    est_rows BIGINT DEFAULT 0,
    est_cost FLOAT DEFAULT 0.0,
    
    -- Join characteristics
    join_types TEXT,
    scan_types TEXT,
    num_hash_joins INT DEFAULT 0,
    num_merge_joins INT DEFAULT 0,
    num_nested_loops INT DEFAULT 0,
    
    -- Scan characteristics
    num_seq_scans INT DEFAULT 0,
    num_index_scans INT DEFAULT 0,
    
    -- Query complexity
    has_subquery BOOLEAN DEFAULT FALSE,
    has_aggregation BOOLEAN DEFAULT FALSE,
    has_sort BOOLEAN DEFAULT FALSE,
    has_hash BOOLEAN DEFAULT FALSE,
    selectivity FLOAT DEFAULT 1.0,
    
    -- Configuration used
    work_mem_kb INT,
    random_page_cost FLOAT,
    effective_cache_size_kb INT,
    
    -- Performance metrics
    actual_time FLOAT NOT NULL,
    planning_time FLOAT,
    peak_memory_kb INT,
    performance_class INT NOT NULL CHECK (performance_class IN (0, 1, 2)),
    
    -- Metadata
    execution_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_synthetic BOOLEAN DEFAULT FALSE,
    
    CONSTRAINT valid_actual_time CHECK (actual_time >= 0)
);

CREATE INDEX IF NOT EXISTS idx_qppe_query_hash ON qppe_training_data(query_hash);
CREATE INDEX IF NOT EXISTS idx_qppe_perf_class ON qppe_training_data(performance_class);
CREATE INDEX IF NOT EXISTS idx_qppe_timestamp ON qppe_training_data(execution_timestamp);
CREATE INDEX IF NOT EXISTS idx_qppe_template ON qppe_training_data(query_template);

-- Model metadata with per-class metrics
CREATE TABLE IF NOT EXISTS qppe_model_metadata (
    id SERIAL PRIMARY KEY,
    model_version INT NOT NULL,
    training_samples INT,
    
    -- Overall metrics
    accuracy FLOAT,
    weighted_f1 FLOAT,
    
    -- Per-class metrics
    fast_precision FLOAT,
    fast_recall FLOAT,
    fast_f1 FLOAT,
    medium_precision FLOAT,
    medium_recall FLOAT,
    medium_f1 FLOAT,
    slow_precision FLOAT,
    slow_recall FLOAT,
    slow_f1 FLOAT,
    
    -- Training details
    used_smote BOOLEAN DEFAULT FALSE,
    used_cost_sensitive BOOLEAN DEFAULT FALSE,
    class_weights TEXT,
    
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE
);

-- Configuration with detailed parameters
CREATE TABLE IF NOT EXISTS qppe_config (
    parameter TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO qppe_config (parameter, value, description) VALUES 
    ('enabled', 'true', 'Enable/disable QPPE optimization'),
    ('penalty_fast', '0.95', 'Cost multiplier for fast queries (favor these plans)'),
    ('penalty_medium', '1.0', 'Cost multiplier for medium queries (neutral)'),
    ('penalty_slow', '1.3', 'Cost multiplier for slow queries (penalize these plans)'),
    ('prediction_timeout_ms', '5', 'Timeout for predictions in ms'),
    ('retrain_threshold', '1000', 'Min samples before retraining'),
    ('fast_threshold_ms', '100', 'Threshold for fast classification'),
    ('slow_threshold_ms', '1000', 'Threshold for slow classification'),
    ('min_samples_per_class', '300', 'Minimum samples per class for training'),
    ('use_smote', 'true', 'Use SMOTE for balancing'),
    ('use_cost_sensitive', 'true', 'Use cost-sensitive learning')
ON CONFLICT (parameter) DO NOTHING;

-- Enhanced performance classification function
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

-- Enhanced stats view with class balance
CREATE OR REPLACE VIEW qppe_training_stats AS
SELECT 
    COUNT(*) as total_samples,
    COUNT(CASE WHEN performance_class = 0 THEN 1 END) as fast_count,
    COUNT(CASE WHEN performance_class = 1 THEN 1 END) as medium_count,
    COUNT(CASE WHEN performance_class = 2 THEN 1 END) as slow_count,
    
    -- Class percentages
    ROUND(100.0 * COUNT(CASE WHEN performance_class = 0 THEN 1 END) / NULLIF(COUNT(*), 0), 1) as fast_pct,
    ROUND(100.0 * COUNT(CASE WHEN performance_class = 1 THEN 1 END) / NULLIF(COUNT(*), 0), 1) as medium_pct,
    ROUND(100.0 * COUNT(CASE WHEN performance_class = 2 THEN 1 END) / NULLIF(COUNT(*), 0), 1) as slow_pct,
    
    -- Balance ratio (ideally close to 1.0)
    ROUND(
        CASE WHEN COUNT(*) > 0 THEN
            LEAST(
                COUNT(CASE WHEN performance_class = 0 THEN 1 END)::FLOAT / NULLIF(COUNT(CASE WHEN performance_class = 1 THEN 1 END), 0),
                COUNT(CASE WHEN performance_class = 1 THEN 1 END)::FLOAT / NULLIF(COUNT(CASE WHEN performance_class = 2 THEN 1 END), 0),
                COUNT(CASE WHEN performance_class = 2 THEN 1 END)::FLOAT / NULLIF(COUNT(CASE WHEN performance_class = 0 THEN 1 END), 0)
            )
        ELSE 0
        END
    , 2) as balance_ratio,
    
    AVG(actual_time) as avg_execution_time,
    MIN(execution_timestamp) as first_sample,
    MAX(execution_timestamp) as last_sample,
    
    -- Synthetic data info
    COUNT(CASE WHEN is_synthetic THEN 1 END) as synthetic_count
FROM qppe_training_data;

-- View for identifying underrepresented classes
CREATE OR REPLACE VIEW qppe_class_targets AS
WITH class_counts AS (
    SELECT 
        performance_class,
        COUNT(*) as current_count
    FROM qppe_training_data
    GROUP BY performance_class
),
max_count AS (
    SELECT COALESCE(MAX(current_count), 0) as max_val FROM class_counts
)
SELECT 
    cc.performance_class,
    CASE cc.performance_class
        WHEN 0 THEN 'FAST'
        WHEN 1 THEN 'MEDIUM'
        WHEN 2 THEN 'SLOW'
    END as class_name,
    cc.current_count,
    mc.max_val as target_count,
    mc.max_val - cc.current_count as samples_needed,
    ROUND(100.0 * cc.current_count / NULLIF(mc.max_val, 0), 1) as balance_pct
FROM class_counts cc
CROSS JOIN max_count mc
ORDER BY cc.performance_class;

-- Function to get class-specific statistics
CREATE OR REPLACE FUNCTION get_class_stats(class_id INT)
RETURNS TABLE(
    class_name TEXT,
    sample_count BIGINT,
    avg_time FLOAT,
    min_time FLOAT,
    max_time FLOAT,
    avg_joins FLOAT,
    avg_cost FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CASE class_id
            WHEN 0 THEN 'FAST'
            WHEN 1 THEN 'MEDIUM'
            WHEN 2 THEN 'SLOW'
        END as class_name,
        COUNT(*) as sample_count,
        AVG(actual_time) as avg_time,
        MIN(actual_time) as min_time,
        MAX(actual_time) as max_time,
        AVG(num_joins)::FLOAT as avg_joins,
        AVG(est_cost) as avg_cost
    FROM qppe_training_data
    WHERE performance_class = class_id;
END;
$$ LANGUAGE plpgsql;

-- Success message
DO $$ BEGIN RAISE NOTICE 'QPPE Schema setup complete!'; END $$;
