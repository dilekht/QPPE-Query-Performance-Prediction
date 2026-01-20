#!/usr/bin/env python3
"""
QPPE Validation Script
Validates training data quality and model performance

Usage:
    python 5_validate_qppe.py --db tpch
"""

import psycopg2
import argparse
import logging
import sys
import pickle
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class QPPEValidator:
    """Validate QPPE implementation"""
    
    def __init__(self, db_config, model_path='qppe_model_enhanced.pkl'):
        self.db_config = db_config
        self.model_path = model_path
        self.issues = []
        self.warnings = []
        
    def connect(self):
        return psycopg2.connect(**self.db_config)
    
    def check_mark(self, passed):
        return "[OK]" if passed else "[X]"
    
    def validate_database_connection(self):
        """Check database connection"""
        logger.info("\n" + "=" * 80)
        logger.info("0. DATABASE CONNECTION")
        logger.info("=" * 80)
        
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            cur.close()
            conn.close()
            
            logger.info(f"  [OK] Connected successfully")
            logger.info(f"  PostgreSQL Version: {version[:50]}...")
            return True
        except Exception as e:
            logger.error(f"  [X] Connection failed: {e}")
            self.issues.append(f"Database connection failed: {e}")
            return False
    
    def validate_schema(self):
        """Check if QPPE schema exists"""
        logger.info("\n" + "=" * 80)
        logger.info("1. SCHEMA VALIDATION")
        logger.info("=" * 80)
        
        required_tables = ['qppe_training_data', 'qppe_model_metadata', 'qppe_config']
        
        conn = self.connect()
        cur = conn.cursor()
        
        all_exist = True
        for table in required_tables:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table,))
            exists = cur.fetchone()[0]
            
            status = self.check_mark(exists)
            logger.info(f"  {status} Table '{table}': {'Found' if exists else 'MISSING'}")
            
            if not exists:
                all_exist = False
                self.issues.append(f"Table '{table}' not found")
        
        cur.close()
        conn.close()
        
        return all_exist
    
    def validate_data_quantity(self):
        """Check if we have enough training data"""
        logger.info("\n" + "=" * 80)
        logger.info("2. DATA QUANTITY VALIDATION")
        logger.info("=" * 80)
        
        conn = self.connect()
        cur = conn.cursor()
        
        # Overall count
        cur.execute("SELECT COUNT(*) FROM qppe_training_data")
        total = cur.fetchone()[0]
        
        # Per-class counts
        cur.execute("""
            SELECT performance_class, COUNT(*)
            FROM qppe_training_data
            GROUP BY performance_class
            ORDER BY performance_class
        """)
        class_counts = {row[0]: row[1] for row in cur.fetchall()}
        
        cur.close()
        conn.close()
        
        # Validate totals
        min_total = 1000
        min_per_class = 300
        recommended_per_class = 400
        
        logger.info(f"  Total samples: {total}")
        passed = total >= min_total
        logger.info(f"    {self.check_mark(passed)} Minimum 1000 samples: {'PASS' if passed else 'FAIL'}")
        if not passed:
            self.issues.append(f"Only {total} total samples (need {min_total})")
        
        logger.info(f"\n  Per-class samples:")
        for cls in [0, 1, 2]:
            cls_name = ['Fast', 'Medium', 'Slow'][cls]
            count = class_counts.get(cls, 0)
            
            passed = count >= min_per_class
            recommended = count >= recommended_per_class
            
            status = "PASS" if passed else "FAIL"
            if passed and not recommended:
                status = "PASS (below recommended)"
            
            logger.info(f"    {cls_name:8s}: {count:4d} {self.check_mark(passed)} {status}")
            
            if not passed:
                self.issues.append(f"{cls_name} class has only {count} samples (need {min_per_class})")
            elif not recommended:
                self.warnings.append(f"{cls_name} class has {count} samples (recommend {recommended_per_class})")
        
        return len([c for c in class_counts.values() if c >= min_per_class]) == 3
    
    def validate_data_balance(self):
        """Check class balance"""
        logger.info("\n" + "=" * 80)
        logger.info("3. DATA BALANCE VALIDATION")
        logger.info("=" * 80)
        
        conn = self.connect()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT performance_class, COUNT(*)
            FROM qppe_training_data
            GROUP BY performance_class
        """)
        class_counts = {row[0]: row[1] for row in cur.fetchall()}
        
        cur.close()
        conn.close()
        
        if not class_counts:
            logger.info("  [X] No data available")
            self.issues.append("No training data")
            return False
        
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        logger.info(f"  Class distribution:")
        for cls in [0, 1, 2]:
            cls_name = ['Fast', 'Medium', 'Slow'][cls]
            count = class_counts.get(cls, 0)
            pct = 100 * count / sum(class_counts.values())
            logger.info(f"    {cls_name:8s}: {count:4d} ({pct:5.1f}%)")
        
        logger.info(f"\n  Balance ratio: {balance_ratio:.3f}")
        
        if balance_ratio >= 0.90:
            logger.info(f"    [OK] EXCELLENT balance (>= 0.90)")
            return True
        elif balance_ratio >= 0.75:
            logger.info(f"    [OK] GOOD balance (>= 0.75)")
            self.warnings.append(f"Balance ratio {balance_ratio:.3f} is acceptable but could be improved")
            return True
        else:
            logger.info(f"    [X] POOR balance (< 0.75)")
            self.issues.append(f"Balance ratio {balance_ratio:.3f} too low (need >= 0.75)")
            return False
    
    def validate_data_diversity(self):
        """Check query diversity"""
        logger.info("\n" + "=" * 80)
        logger.info("4. DATA DIVERSITY VALIDATION")
        logger.info("=" * 80)
        
        conn = self.connect()
        cur = conn.cursor()
        
        # Unique query templates
        cur.execute("""
            SELECT COUNT(DISTINCT query_template)
            FROM qppe_training_data
        """)
        unique_templates = cur.fetchone()[0]
        
        # Feature diversity
        cur.execute("""
            SELECT 
                COUNT(DISTINCT num_joins) as unique_joins,
                COUNT(DISTINCT num_relations) as unique_relations,
                AVG(num_joins) as avg_joins,
                COALESCE(STDDEV(num_joins), 0) as stddev_joins,
                AVG(est_cost) as avg_cost,
                COALESCE(STDDEV(est_cost), 0) as stddev_cost
            FROM qppe_training_data
        """)
        diversity = cur.fetchone()
        
        cur.close()
        conn.close()
        
        logger.info(f"  Query templates: {unique_templates}")
        passed = unique_templates >= 5
        logger.info(f"    {self.check_mark(passed)} Minimum 5 templates: {'PASS' if passed else 'FAIL'}")
        if not passed:
            self.issues.append(f"Only {unique_templates} query templates (need >= 5)")
        
        logger.info(f"\n  Feature diversity:")
        logger.info(f"    Unique join counts: {diversity[0]}")
        logger.info(f"    Unique relation counts: {diversity[1]}")
        logger.info(f"    Avg joins: {diversity[2]:.2f} (std={diversity[3]:.2f})")
        logger.info(f"    Avg cost: {diversity[4]:.2f} (std={diversity[5]:.2f})")
        
        # Check for good variation
        passed = diversity[0] >= 3 and diversity[3] > 0.5
        logger.info(f"    {self.check_mark(passed)} Adequate join variation: {'PASS' if passed else 'WARNING'}")
        if not passed:
            self.warnings.append("Low join count variation - may hurt generalization")
        
        return unique_templates >= 5
    
    def validate_model_file(self):
        """Check if model file exists and is valid"""
        logger.info("\n" + "=" * 80)
        logger.info("5. MODEL FILE VALIDATION")
        logger.info("=" * 80)
        
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            logger.info(f"  [X] Model file not found: {self.model_path}")
            self.issues.append(f"Model file not found - run training first")
            return False
        
        logger.info(f"  [OK] Model file found: {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            required_keys = ['model', 'scaler', 'accuracy']
            for key in required_keys:
                exists = key in model_data
                logger.info(f"    {self.check_mark(exists)} Key '{key}': {'Found' if exists else 'MISSING'}")
                if not exists:
                    self.issues.append(f"Model missing required key: {key}")
            
            logger.info(f"\n  Model details:")
            logger.info(f"    Accuracy: {model_data.get('accuracy', 'N/A'):.4f}")
            logger.info(f"    Weighted F1: {model_data.get('weighted_f1', 'N/A'):.4f}")
            logger.info(f"    Used SMOTE: {model_data.get('used_smote', 'N/A')}")
            logger.info(f"    Used Cost-Sensitive: {model_data.get('used_cost_sensitive', 'N/A')}")
            logger.info(f"    Timestamp: {model_data.get('timestamp', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"  [X] Error loading model: {e}")
            self.issues.append(f"Error loading model file: {e}")
            return False
    
    def validate_model_metrics(self):
        """Check model performance metrics from file"""
        logger.info("\n" + "=" * 80)
        logger.info("6. MODEL PERFORMANCE VALIDATION")
        logger.info("=" * 80)
        
        if not Path(self.model_path).exists():
            logger.info("  [X] No model file - skipping metric validation")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
        except:
            return False
        
        accuracy = model_data.get('accuracy', 0)
        logger.info(f"  Overall accuracy: {accuracy:.4f}")
        passed = accuracy >= 0.80
        logger.info(f"    {self.check_mark(passed)} Minimum 80% accuracy: {'PASS' if passed else 'FAIL'}")
        if not passed:
            self.issues.append(f"Overall accuracy {accuracy:.1%} < 80%")
        
        if 'per_class_metrics' not in model_data:
            logger.info("  [X] No per-class metrics available")
            return passed
        
        metrics = model_data['per_class_metrics']
        
        logger.info(f"\n  Per-class metrics:")
        all_pass = True
        
        for i, cls_name in enumerate(['Fast', 'Medium', 'Slow']):
            p = metrics['precision'][i]
            r = metrics['recall'][i]
            f = metrics['f1'][i]
            
            prec_pass = p >= 0.80
            rec_pass = r >= 0.80
            f1_pass = f >= 0.80
            
            logger.info(f"\n    {cls_name}:")
            logger.info(f"      Precision: {p:.3f} {self.check_mark(prec_pass)}")
            logger.info(f"      Recall:    {r:.3f} {self.check_mark(rec_pass)}")
            logger.info(f"      F1:        {f:.3f} {self.check_mark(f1_pass)}")
            
            if not (prec_pass and rec_pass and f1_pass):
                all_pass = False
                self.issues.append(f"{cls_name} class below 80% threshold")
        
        if all_pass:
            logger.info(f"\n  [OK] ALL CLASSES MEET 80%+ THRESHOLD!")
        else:
            logger.info(f"\n  [X] Some classes below 80% threshold")
        
        return all_pass
    
    def generate_report(self):
        """Generate validation report"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        if not self.issues and not self.warnings:
            logger.info("\n[SUCCESS] ALL CHECKS PASSED!")
            logger.info("\nYour QPPE implementation is ready for use.")
            logger.info("\nNext steps:")
            logger.info("  1. Start prediction service: python 3_qppe_service_enhanced.py --db-name tpch")
            logger.info("  2. Generate visualizations: python 4_visualize_results.py --db tpch")
            return True
        
        if self.warnings:
            logger.info("\n[!] WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                logger.info(f"    {i}. {warning}")
        
        if self.issues:
            logger.info("\n[X] CRITICAL ISSUES:")
            for i, issue in enumerate(self.issues, 1):
                logger.info(f"    {i}. {issue}")
            
            logger.info("\n  RECOMMENDED ACTIONS:")
            
            if any("samples" in issue.lower() for issue in self.issues):
                logger.info("    -> Generate more training data:")
                logger.info("       python 2_generate_balanced_data.py --db tpch --target-samples 500")
            
            if any("balance" in issue.lower() for issue in self.issues):
                logger.info("    -> Focus on underrepresented classes")
                logger.info("    -> Adjust classification thresholds")
            
            if any("accuracy" in issue.lower() or "threshold" in issue.lower() for issue in self.issues):
                logger.info("    -> Retrain with SMOTE and cost-sensitive learning:")
                logger.info("       python 3_qppe_service_enhanced.py --db-name tpch --train-only")
            
            if any("model" in issue.lower() and "not found" in issue.lower() for issue in self.issues):
                logger.info("    -> Train a model first:")
                logger.info("       python 3_qppe_service_enhanced.py --db-name tpch --train-only")
            
            return False
        
        logger.info("\n[OK] Validation passed with warnings")
        logger.info("Consider addressing warnings before production deployment")
        return True
    
    def run_full_validation(self):
        """Run all validation checks"""
        logger.info("=" * 80)
        logger.info("QPPE VALIDATION REPORT")
        logger.info("=" * 80)
        
        if not self.validate_database_connection():
            return self.generate_report()
        
        checks = [
            self.validate_schema(),
            self.validate_data_quantity(),
            self.validate_data_balance(),
            self.validate_data_diversity(),
            self.validate_model_file(),
            self.validate_model_metrics()
        ]
        
        return self.generate_report()


def main():
    parser = argparse.ArgumentParser(description='Validate QPPE implementation')
    parser.add_argument('--db', default='tpch', help='Database name')
    parser.add_argument('--user', default='postgres', help='DB user')
    parser.add_argument('--password', default='postgres', help='DB password')
    parser.add_argument('--host', default='localhost', help='DB host')
    parser.add_argument('--port', type=int, default=5432, help='DB port')
    parser.add_argument('--model', default='qppe_model_enhanced.pkl', help='Model file path')
    
    args = parser.parse_args()
    
    db_config = {
        'dbname': args.db,
        'user': args.user,
        'password': args.password,
        'host': args.host,
        'port': args.port
    }
    
    validator = QPPEValidator(db_config, args.model)
    success = validator.run_full_validation()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
