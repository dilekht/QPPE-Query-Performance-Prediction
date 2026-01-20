# QPPE: Query Performance Prediction Engine for PostgreSQL

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-336791.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)

## Overview

QPPE (Query Performance Prediction Engine) is an AI-assisted query optimization system for PostgreSQL that uses machine learning to predict query execution performance. By learning from historical query execution patterns, QPPE can classify queries into performance categories (Fast, Medium, Slow) and help optimize query plan selection.

### Key Features

- **Machine Learning Classification**: Gradient Boosting classifier with 85%+ accuracy
- **Balanced Training**: SMOTE (Synthetic Minority Over-sampling Technique) for class balance
- **Cost-Sensitive Learning**: Weighted learning to handle class imbalance
- **TPC-H Benchmark Support**: Pre-configured for TPC-H benchmark queries
- **Comprehensive Visualization**: 10 publication-ready figures for analysis
- **Cross-Platform**: Windows and Linux compatible (TCP sockets)

### Performance Targets

| Metric | Target | Typical Result |
|--------|--------|----------------|
| Overall Accuracy | ≥ 80% | 85%+ |
| Fast Class F1 | ≥ 80% | 87%+ |
| Medium Class F1 | ≥ 80% | 84%+ |
| Slow Class F1 | ≥ 80% | 87%+ |
| Training Samples | ≥ 1000 | 1200+ |
| Class Balance Ratio | ≥ 0.75 | ~1.0 |

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| PostgreSQL | 12+ | 14+ |
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 8 GB+ |
| Storage | 1 GB | 5 GB |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/qppe.git
cd qppe
```

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Database

```bash
# Create database
createdb tpch

# Load TPC-H sample data
psql -d tpch -f sql/0_create_tpch_sample.sql

# Setup QPPE schema
psql -d tpch -f sql/1_schema_setup.sql
```

### 4. Generate Training Data

```bash
python scripts/2_generate_balanced_data.py \
    --db tpch \
    --user postgres \
    --password YOUR_PASSWORD \
    --target-samples 400
```

### 5. Train the Model

```bash
python scripts/3_qppe_service.py \
    --db-name tpch \
    --db-user postgres \
    --db-password YOUR_PASSWORD \
    --train-only
```

### 6. Validate the Implementation

```bash
python scripts/5_validate_qppe.py \
    --db tpch \
    --user postgres \
    --password YOUR_PASSWORD
```

### 7. Generate Visualizations

```bash
python scripts/4_visualize_results.py \
    --db tpch \
    --user postgres \
    --password YOUR_PASSWORD \
    --output-dir ./figures
```

---

## Project Structure

```
qppe/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── sql/                               # SQL scripts
│   ├── 0_create_tpch_sample.sql      # TPC-H sample data generator
│   └── 1_schema_setup.sql            # QPPE schema and tables
│
├── scripts/                           # Python scripts
│   ├── 2_generate_balanced_data.py   # Training data generator
│   ├── 3_qppe_service.py             # ML service with SMOTE
│   ├── 4_visualize_results.py        # Visualization generator
│   └── 5_validate_qppe.py            # Validation script
│
└── figures/                           # Generated visualizations
    ├── 1_class_distribution.png
    ├── 2_execution_time_distribution.png
    ├── 3_feature_importance.png
    ├── 4_confusion_matrix.png
    ├── 5_model_performance_metrics.png
    ├── 6_query_template_analysis.png
    ├── 7_feature_correlations.png
    ├── 8_learning_curve.png
    ├── 9_performance_improvement.png
    └── 10_training_summary_dashboard.png
```

---

## Detailed Usage

### Training Data Generation

The data generator creates balanced training samples across three performance classes:

- **Fast** (Class 0): Execution time < 100ms
- **Medium** (Class 1): 100ms ≤ Execution time < 1000ms  
- **Slow** (Class 2): Execution time ≥ 1000ms

```bash
python scripts/2_generate_balanced_data.py \
    --db tpch \
    --user postgres \
    --password YOUR_PASSWORD \
    --host localhost \
    --port 5432 \
    --target-samples 400  # Per class (1200+ total)
```

**Options:**
- `--target-samples`: Target samples per class (default: 400)
- `--db`: Database name (default: tpch)
- `--user`: PostgreSQL username (default: postgres)
- `--password`: PostgreSQL password
- `--host`: Database host (default: localhost)
- `--port`: Database port (default: 5432)

### Model Training

The ML service uses Gradient Boosting with SMOTE and cost-sensitive learning:

```bash
# Train only (no server)
python scripts/3_qppe_service.py --db-name tpch --train-only

# Train and start prediction server
python scripts/3_qppe_service.py --db-name tpch --train-first

# Start server with existing model
python scripts/3_qppe_service.py --db-name tpch
```

**Training Options:**
- `--train-only`: Train model and exit
- `--train-first`: Train before starting server
- `--no-smote`: Disable SMOTE oversampling
- `--no-cost-sensitive`: Disable class weights
- `--model`: Model file path (default: qppe_model_enhanced.pkl)

### Prediction Service

The service listens on TCP port 5555 (configurable):

```bash
python scripts/3_qppe_service.py \
    --db-name tpch \
    --host 127.0.0.1 \
    --port 5555
```

**Service Protocol:**
1. Client sends feature count (64 bytes, text)
2. Client sends features (feature_count × 8 bytes, doubles)
3. Server responds with prediction (4 bytes, int: 0/1/2)

### Visualization

Generate all 10 figures for analysis and documentation:

```bash
python scripts/4_visualize_results.py \
    --db tpch \
    --output-dir ./figures \
    --model qppe_model_enhanced.pkl
```

**Generated Figures:**
1. Class Distribution (bar + pie)
2. Execution Time Distribution (histogram + boxplot)
3. Feature Importance (horizontal bar)
4. Confusion Matrix (heatmap)
5. Per-Class Performance Metrics (grouped bar)
6. Query Template Analysis (stacked bar)
7. Feature Correlations (heatmap)
8. Learning Curve (line plot)
9. Performance Improvement (comparison bar)
10. Training Summary Dashboard (multi-panel)

---

## Database Schema

### Training Data Table

```sql
CREATE TABLE qppe_training_data (
    id SERIAL PRIMARY KEY,
    query_hash TEXT NOT NULL,
    query_template TEXT,
    
    -- Features
    num_joins INT,
    num_relations INT,
    join_depth INT,
    est_rows BIGINT,
    est_cost FLOAT,
    has_subquery BOOLEAN,
    has_aggregation BOOLEAN,
    has_sort BOOLEAN,
    has_hash BOOLEAN,
    num_hash_joins INT,
    num_merge_joins INT,
    num_nested_loops INT,
    num_seq_scans INT,
    num_index_scans INT,
    selectivity FLOAT,
    
    -- Labels
    actual_time FLOAT NOT NULL,
    performance_class INT NOT NULL CHECK (performance_class IN (0, 1, 2)),
    
    execution_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Configuration Table

```sql
CREATE TABLE qppe_config (
    parameter TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT
);
```

Key parameters:
- `fast_threshold_ms`: Upper bound for Fast class (default: 100)
- `slow_threshold_ms`: Lower bound for Slow class (default: 1000)
- `penalty_fast`: Cost multiplier for Fast predictions (default: 0.95)
- `penalty_slow`: Cost multiplier for Slow predictions (default: 1.3)

---

## Troubleshooting

### Common Issues

**1. "No training data available"**
```bash
# Generate training data first
python scripts/2_generate_balanced_data.py --db tpch --password YOUR_PASSWORD
```

**2. "Insufficient samples for SMOTE"**
```bash
# Increase target samples
python scripts/2_generate_balanced_data.py --target-samples 500
```

**3. "Database connection failed"**
```bash
# Check PostgreSQL is running
pg_isready

# Verify credentials
psql -U postgres -d tpch -c "SELECT 1;"
```

**4. "Model accuracy below 80%"**
- Generate more training data
- Ensure balanced class distribution
- Try adjusting classification thresholds

**5. "matplotlib style not found"**
```bash
# Update matplotlib
pip install --upgrade matplotlib
```

### Checking Data Quality

```sql
-- Check class distribution
SELECT * FROM qppe_training_stats;

-- Check balance targets
SELECT * FROM qppe_class_targets;

-- Check recent samples
SELECT query_template, performance_class, actual_time 
FROM qppe_training_data 
ORDER BY execution_timestamp DESC 
LIMIT 10;
```

---

## Performance Tuning

### Adjusting Classification Thresholds

If your system has different performance characteristics:

```sql
-- For faster systems
UPDATE qppe_config SET value = '50' WHERE parameter = 'fast_threshold_ms';
UPDATE qppe_config SET value = '500' WHERE parameter = 'slow_threshold_ms';

-- For slower systems
UPDATE qppe_config SET value = '200' WHERE parameter = 'fast_threshold_ms';
UPDATE qppe_config SET value = '2000' WHERE parameter = 'slow_threshold_ms';
```

### Model Hyperparameters

In `3_qppe_service.py`, adjust the GradientBoostingClassifier parameters:

```python
self.model = GradientBoostingClassifier(
    n_estimators=150,      # More trees = better but slower
    max_depth=7,           # Deeper = more complex patterns
    learning_rate=0.05,    # Lower = more robust
    subsample=0.8,         # Prevent overfitting
    min_samples_split=20,  # Minimum samples to split
    min_samples_leaf=10,   # Minimum samples in leaf
    random_state=42
)
```

---

## API Reference

### Python Classes

#### `EnhancedQPPEService`
Main ML service class.

```python
service = EnhancedQPPEService(
    host='127.0.0.1',
    port=5555,
    db_config={'dbname': 'tpch', ...},
    model_path='qppe_model.pkl'
)
service.train_model(use_smote=True, use_cost_sensitive=True)
prediction = service.predict(features)  # Returns 0, 1, or 2
```

#### `BalancedDataGenerator`
Training data generator.

```python
generator = BalancedDataGenerator(db_config, target_per_class=400)
generator.generate_balanced_data()
```

#### `QPPEVisualizer`
Visualization generator.

```python
viz = QPPEVisualizer(db_config, output_dir='./figures')
viz.generate_all()
```

#### `QPPEValidator`
Implementation validator.

```python
validator = QPPEValidator(db_config, model_path='qppe_model.pkl')
validator.run_full_validation()
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{qppe2025,
  title={QPPE: Query Performance Prediction Engine for PostgreSQL},
  author={Tahar Dilekh},
  year={2025},
  url={https://github.com/dilekht/QPPE-Query-Performance-Prediction}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- TPC-H Benchmark Council for the benchmark specification
- scikit-learn team for machine learning tools
- PostgreSQL community for the database system
