# QPPE: Query Performance Prediction Engine

AI-Assisted Query Optimization for PostgreSQL using Machine Learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL 12+](https://img.shields.io/badge/postgresql-12%2B-blue.svg)](https://www.postgresql.org/)

## 📄 Paper

**Title:** AI-Assisted Query Optimization in PostgreSQL: A Machine Learning Approach to Performance Prediction

**Published in:** The VLDB Journal (Springer), 2025

**Authors:** Tahar Dilekh

**Abstract:** QPPE is a practical machine learning system that enhances PostgreSQL's query optimizer through performance-aware cost adjustment. Using Gradient Boosting trained on balanced data with SMOTE, we achieve 86% accuracy with 80%+ precision/recall across all performance classes, demonstrating 18-29% improvements on TPC-H benchmarks.

## 🌟 Key Features

- ✅ **86% Classification Accuracy** with balanced performance across all query types
- ✅ **18-29% Performance Improvement** on TPC-H benchmarks (SF1-SF100)
- ✅ **Minimal Overhead** (1.5ms average planning time)
- ✅ **Easy Integration** via PostgreSQL planner hooks
- ✅ **Balanced Training** using SMOTE + cost-sensitive learning
- ✅ **Production-Ready** with graceful degradation

## 🚀 Quick Start

### Prerequisites

- PostgreSQL 12+ (tested on 14.5)
- Python 3.8+
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/dilekht/QPPE-Query-Performance-Prediction.git
cd QPPE-Query-Performance-Prediction
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup PostgreSQL database**
```bash
createdb qppe_demo
psql -d qppe_demo -f sql/1_schema_setup.sql
```

4. **Generate training data**
```bash
python src/generate_training_data.py --db qppe_demo --iterations 10
```

5. **Train the model**
```bash
python src/qppe_service.py --db-name qppe_demo --train-only
```

6. **Start the prediction service**
```bash
python src/qppe_service.py --db-name qppe_demo
```

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PostgreSQL                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Planner Hook (Feature Extraction)        │  │
│  └────────────────┬─────────────────────────────────┘  │
└───────────────────┼─────────────────────────────────────┘
                    │ Unix Socket
                    ▼
┌─────────────────────────────────────────────────────────┐
│          ML Prediction Service (Python)                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │    Gradient Boosting Classifier (150 trees)      │  │
│  │    - 86% accuracy, 80%+ per-class metrics        │  │
│  │    - <1ms prediction latency                     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                    ▲
                    │ Training Data
┌─────────────────────────────────────────────────────────┐
│            Training Data Pipeline                       │
│  - Strategic configuration variation                    │
│  - Query template exploration                           │
│  - SMOTE for class balancing                           │
│  - 1,209 balanced samples (403 per class)              │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
QPPE/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── sql/
│   └── 1_schema_setup.sql            # Database schema and configuration
├── src/
│   ├── qppe_service.py               # ML prediction service
│   ├── generate_training_data.py     # Training data generator
│   └── visualize_results.py          # Result visualization
├── data/
│   └── sample_training_data.csv      # Example training data
├── models/
│   └── .gitkeep                      # Model files stored here
└── figures/
    ├── fig1_overall_performance.png
    ├── fig2_training_distribution.png
    ├── fig3_confusion_matrix.png
    └── ... (10 figures total)
```

## 🔧 Configuration

Edit `qppe_config` table in PostgreSQL:

```sql
-- Adjust cost penalties
UPDATE qppe_config SET value = '0.95' WHERE parameter = 'penalty_fast';
UPDATE qppe_config SET value = '1.3' WHERE parameter = 'penalty_slow';

-- Adjust classification thresholds
UPDATE qppe_config SET value = '100' WHERE parameter = 'fast_threshold_ms';
UPDATE qppe_config SET value = '1000' WHERE parameter = 'slow_threshold_ms';
```

## 📈 Performance Results

### Overall Improvements

| Scale Factor | Database Size | Baseline | QPPE | Improvement |
|--------------|--------------|----------|------|-------------|
| SF1          | 1 GB         | 1.24s    | 1.02s| **18%**     |
| SF10         | 10 GB        | 8.67s    | 6.58s| **24%**     |
| SF100        | 100 GB       | 94.3s    | 67.1s| **29%**     |

### Per-Query Improvements (SF10)

| Query | Baseline | QPPE  | Improvement |
|-------|----------|-------|-------------|
| Q9    | 18.3s    | 9.7s  | **47%**     |
| Q21   | 31.2s    | 18.1s | **42%**     |
| Q17   | 14.7s    | 9.0s  | **39%**     |
| Q5    | 12.4s    | 8.6s  | **31%**     |
| Q3    | 6.2s     | 4.5s  | **28%**     |

### Model Performance

- **Overall Accuracy:** 86.0%
- **Fast Queries:** Precision 87.4%, Recall 86.0%, F1 86.7%
- **Medium Queries:** Precision 83.1%, Recall 84.8%, F1 84.0%
- **Slow Queries:** Precision 87.5%, Recall 86.9%, F1 87.2%
- **Planning Overhead:** 1.5ms average

## 🎯 Usage Examples

### Basic Usage

```python
# Start the service
python src/qppe_service.py --db-name mydb

# In another terminal, run queries in PostgreSQL
psql -d mydb -c "SELECT * FROM large_table WHERE ..."
```

### Training with Custom Data

```python
# Generate training data from your workload
python src/generate_training_data.py \
    --db mydb \
    --iterations 20 \
    --queries custom_queries.sql

# Train model
python src/qppe_service.py --db-name mydb --train-only

# Start service with new model
python src/qppe_service.py --db-name mydb
```

### Monitoring

```sql
-- View training statistics
SELECT * FROM qppe_training_stats;

-- Check model metadata
SELECT * FROM qppe_model_metadata 
ORDER BY trained_at DESC LIMIT 5;

-- View recent predictions
SELECT query_hash, num_joins, performance_class, actual_time
FROM qppe_training_data
ORDER BY execution_timestamp DESC
LIMIT 10;
```

## 📊 Visualizations

Generate all visualizations:

```bash
python src/visualize_results.py --output-dir ./figures
```

This creates 10 publication-quality figures:
1. Overall performance comparison
2. Training data distribution (before/after SMOTE)
3. Confusion matrix
4. Per-class metrics
5. Feature importance
6. Performance distribution and CDF
7. Per-query improvements
8. Planning overhead analysis
9. Scalability analysis
10. System architecture diagram

## 🧪 Running Experiments

### TPC-H Benchmark

1. **Setup TPC-H database**
```bash
# Generate TPC-H data (using dbgen)
./dbgen -s 1  # Scale factor 1 (1GB)

# Load into PostgreSQL
./tpch_load.sh
```

2. **Run experiments**
```bash
# Generate training data
python src/generate_training_data.py --db tpch --iterations 10

# Train model
python src/qppe_service.py --db-name tpch --train-only

# Run TPC-H queries and measure performance
./run_tpch_benchmark.sh
```

## 🔬 Technical Details

### Feature Engineering

QPPE extracts 15 features from query plans:

**Basic Features:**
- `est_cost`: Optimizer's estimated cost
- `est_rows`: Estimated output rows
- `num_joins`: Number of joins
- `num_relations`: Number of tables
- `join_depth`: Join tree depth

**Join/Scan Methods:**
- `num_hash_joins`, `num_merge_joins`, `num_nested_loops`
- `num_seq_scans`, `num_index_scans`

**Complexity Indicators:**
- `has_subquery`, `has_aggregation`, `has_sort`, `has_hash`, `selectivity`

### Machine Learning Model

- **Algorithm:** Gradient Boosting Classifier
- **Trees:** 150
- **Max Depth:** 7
- **Learning Rate:** 0.05
- **Training:** Cost-sensitive with balanced class weights
- **Balancing:** SMOTE (Synthetic Minority Over-sampling)
- **Training Data:** 1,209 samples (403 per class)

### Performance Classification

| Class  | Execution Time   | Cost Multiplier |
|--------|------------------|-----------------|
| Fast   | < 100ms          | 0.95×          |
| Medium | 100ms - 1000ms   | 1.0×           |
| Slow   | ≥ 1000ms         | 1.3×           |

## 🛠️ Troubleshooting

### Service won't start

```bash
# Check if socket exists
ls -l /tmp/qppe_service.sock

# Remove stale socket
rm /tmp/qppe_service.sock

# Check logs
tail -f qppe_service.log
```

### No training data

```bash
# Verify table
psql -d mydb -c "SELECT COUNT(*) FROM qppe_training_data;"

# Re-run data generator
python src/generate_training_data.py --db mydb --iterations 5
```

### Low accuracy

- Increase training data iterations
- Check class balance in data
- Adjust feature extraction
- Try different hyperparameters

## 📚 Citation

If you use QPPE in your research, please cite:

```bibtex
@article{qppe2025,
  title={AI-Assisted Query Optimization in PostgreSQL: A Machine Learning Approach to Performance Prediction},
  author={Tahar Dilekh},
  journal={The VLDB Journal},
  publisher={Springer},
  year={2025},
  doi={10.1007/xxxxx}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PostgreSQL Global Development Group
- TPC-H Benchmark
- scikit-learn contributors
- SMOTE algorithm by Chawla et al.

## 📧 Contact

For questions or issues:
- **Email:** tahar.dilekh@univ-batna2.dz
- **Issues:** [GitHub Issues](https://github.com/dilekht/QPPE/issues)
- **Paper:** [Link to paper]

## 🔗 Related Work

- [Neo: A Learned Query Optimizer](https://dl.acm.org/doi/10.14778/3342263.3342644)
- [Bao: Making Learned Query Optimization Practical](https://dl.acm.org/doi/10.1145/3448016.3452838)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

**Built with ❤️ for the database community**
