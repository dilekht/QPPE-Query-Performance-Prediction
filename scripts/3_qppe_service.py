#!/usr/bin/env python3
"""
Enhanced QPPE ML Service with SMOTE and Cost-Sensitive Learning
Windows-Compatible Version (TCP Socket instead of Unix Socket)

Requirements:
    pip install psycopg2-binary scikit-learn numpy imbalanced-learn

Usage:
    python 3_qppe_service_enhanced.py --db-name tpch --train-only
    python 3_qppe_service_enhanced.py --db-name tpch --train-first
"""

import socket
import struct
import pickle
import numpy as np
import argparse
import logging
import threading
import sys
import time
import os
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import psycopg2
from datetime import datetime
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('qppe_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EnhancedQPPEService:
    """Enhanced QPPE with SMOTE and Cost-Sensitive Learning (Windows-Compatible)"""
    
    def __init__(self, host='127.0.0.1', port=5555, db_config=None, model_path='qppe_model_enhanced.pkl'):
        self.host = host
        self.port = port
        self.db_config = db_config or {}
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.class_weights = None
        self.feature_names = [
            'est_cost', 'est_rows', 'num_joins', 'num_relations',
            'join_depth', 'has_subquery', 'has_aggregation', 'has_sort',
            'has_hash', 'num_hash_joins', 'num_merge_joins', 'num_nested_loops',
            'num_seq_scans', 'num_index_scans', 'selectivity'
        ]
        self.lock = threading.Lock()
        self.running = True
        self.stats = {
            'predictions': 0,
            'errors': 0,
            'start_time': time.time(),
            'class_predictions': {0: 0, 1: 0, 2: 0}
        }
        
    def get_db_connection(self):
        """Create database connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def load_training_data(self, limit=50000):
        """Load training data with enhanced features"""
        logger.info("Loading training data...")
        
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        # Get all enhanced features
        query = """
            SELECT 
                COALESCE(est_cost, 0) as est_cost,
                COALESCE(est_rows, 0) as est_rows,
                COALESCE(num_joins, 0) as num_joins,
                COALESCE(num_relations, 1) as num_relations,
                COALESCE(join_depth, 0) as join_depth,
                COALESCE(has_subquery::int, 0) as has_subquery,
                COALESCE(has_aggregation::int, 0) as has_aggregation,
                COALESCE(has_sort::int, 0) as has_sort,
                COALESCE(has_hash::int, 0) as has_hash,
                COALESCE(num_hash_joins, 0) as num_hash_joins,
                COALESCE(num_merge_joins, 0) as num_merge_joins,
                COALESCE(num_nested_loops, 0) as num_nested_loops,
                COALESCE(num_seq_scans, 0) as num_seq_scans,
                COALESCE(num_index_scans, 0) as num_index_scans,
                COALESCE(selectivity, 1.0) as selectivity,
                performance_class
            FROM qppe_training_data
            WHERE actual_time > 0 
            ORDER BY execution_timestamp DESC
            LIMIT %s
        """
        
        cur.execute(query, (limit,))
        data = cur.fetchall()
        cur.close()
        conn.close()
        
        if not data:
            logger.warning("No training data available")
            return None, None
        
        logger.info(f"Loaded {len(data)} training samples")
        
        X = np.array([row[:-1] for row in data], dtype=np.float64)
        y = np.array([row[-1] for row in data], dtype=np.int32)
        
        # Clean data
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=0.0)
        
        # Check class distribution
        class_counts = Counter(y)
        logger.info(f"Feature matrix: {X.shape}")
        logger.info(f"Class distribution:")
        for cls in sorted(class_counts.keys()):
            cls_name = ['FAST', 'MEDIUM', 'SLOW'][cls]
            logger.info(f"  {cls_name}: {class_counts[cls]} ({100*class_counts[cls]/len(y):.1f}%)")
        
        # Check if we have enough samples
        min_samples = min(class_counts.values())
        if min_samples < 50:
            logger.warning(f"Low sample count for some classes: {min_samples}")
        
        return X, y
    
    def apply_smote(self, X, y):
        """Apply SMOTE to balance classes"""
        logger.info("\nApplying SMOTE for class balancing...")
        
        original_counts = Counter(y)
        logger.info("Original distribution:")
        for cls, count in sorted(original_counts.items()):
            logger.info(f"  Class {cls}: {count}")
        
        try:
            # Use SMOTE with careful parameters
            min_samples = min(original_counts.values())
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            
            if k_neighbors < 1:
                logger.warning("Not enough samples for SMOTE, skipping...")
                return X, y
            
            smote = SMOTE(
                sampling_strategy='auto',
                k_neighbors=k_neighbors,
                random_state=42
            )
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            resampled_counts = Counter(y_resampled)
            logger.info("After SMOTE:")
            for cls, count in sorted(resampled_counts.items()):
                added = count - original_counts.get(cls, 0)
                logger.info(f"  Class {cls}: {count} (+{added})")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"SMOTE failed: {e}")
            logger.info("Continuing without SMOTE...")
            return X, y
    
    def compute_class_weights(self, y):
        """Compute class weights for cost-sensitive learning"""
        logger.info("\nComputing class weights for cost-sensitive learning...")
        
        # Compute balanced class weights
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # Create weight dictionary
        class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        
        logger.info("Class weights:")
        for cls, weight in sorted(class_weight_dict.items()):
            cls_name = ['FAST', 'MEDIUM', 'SLOW'][cls]
            logger.info(f"  {cls_name}: {weight:.3f}")
        
        return class_weight_dict
    
    def train_model(self, use_smote=True, use_cost_sensitive=True):
        """Train the ML model with enhanced techniques"""
        logger.info("=" * 80)
        logger.info("Enhanced Model Training with SMOTE and Cost-Sensitive Learning")
        logger.info("=" * 80)
        
        # Load data
        X, y = self.load_training_data()
        
        if X is None or len(X) < 100:
            logger.error(f"Insufficient training data: {len(X) if X is not None else 0} samples")
            logger.error("Need at least 100 samples to train")
            return False
        
        # Apply SMOTE if requested
        if use_smote:
            X, y = self.apply_smote(X, y)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        logger.info(f"\nTraining set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Fit scaler
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Compute class weights if using cost-sensitive learning
        sample_weights = None
        if use_cost_sensitive:
            self.class_weights = self.compute_class_weights(y_train)
            sample_weights = np.array([self.class_weights[cls] for cls in y_train])
        
        # Train model with Gradient Boosting
        logger.info("\n" + "=" * 80)
        logger.info("Training Gradient Boosting Classifier...")
        logger.info("=" * 80)
        
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            verbose=0
        )
        
        # Train with sample weights if using cost-sensitive
        if use_cost_sensitive and sample_weights is not None:
            self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"\nTraining accuracy: {train_acc:.4f}")
        logger.info(f"Testing accuracy: {test_acc:.4f}")
        
        # Cross-validation with stratified folds
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        logger.info(f"Cross-validation (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Weighted F1 score
        y_pred = self.model.predict(X_test_scaled)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        logger.info(f"Weighted F1 Score: {weighted_f1:.4f}")
        
        # Detailed metrics
        logger.info("\n" + "=" * 80)
        logger.info("Classification Report:")
        logger.info("=" * 80)
        print(classification_report(y_test, y_pred, 
                                   target_names=['Fast (0)', 'Medium (1)', 'Slow (2)'],
                                   digits=3))
        
        # Check if all classes meet 80% threshold
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        logger.info("\nPer-Class Performance Analysis:")
        all_above_threshold = True
        for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
            cls_name = ['FAST', 'MEDIUM', 'SLOW'][i]
            p_status = '[OK]' if p >= 0.80 else '[LOW]'
            r_status = '[OK]' if r >= 0.80 else '[LOW]'
            f_status = '[OK]' if f >= 0.80 else '[LOW]'
            logger.info(f"{cls_name}:")
            logger.info(f"  Precision: {p:.3f} {p_status}")
            logger.info(f"  Recall:    {r:.3f} {r_status}")
            logger.info(f"  F1:        {f:.3f} {f_status}")
            logger.info(f"  Support:   {s}")
            if p < 0.80 or r < 0.80:
                all_above_threshold = False
        
        if all_above_threshold:
            logger.info("\n[SUCCESS] All classes meet 80%+ threshold!")
        else:
            logger.warning("\n[WARNING] Some classes below 80% threshold")
            logger.info("Consider: more training data, feature engineering, or hyperparameter tuning")
        
        # Confusion Matrix
        logger.info("\n" + "=" * 80)
        logger.info("Confusion Matrix:")
        logger.info("=" * 80)
        cm = confusion_matrix(y_test, y_pred)
        logger.info("          Predicted")
        logger.info("          Fast  Med  Slow")
        for i, row in enumerate(cm):
            cls_name = ['Fast', 'Med ', 'Slow'][i]
            logger.info(f"Actual {cls_name}: {row[0]:4d} {row[1]:4d} {row[2]:4d}")
        
        # Per-class accuracy
        logger.info("\nPer-Class Accuracy:")
        for i in range(3):
            cls_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            cls_name = ['Fast', 'Medium', 'Slow'][i]
            logger.info(f"  {cls_name}: {cls_acc:.1%}")
        
        # Feature importance
        logger.info("\n" + "=" * 80)
        logger.info("Feature Importance:")
        logger.info("=" * 80)
        importance = self.model.feature_importances_
        feature_importance = sorted(zip(self.feature_names, importance), 
                                   key=lambda x: x[1], reverse=True)
        for name, imp in feature_importance:
            bar = '#' * int(imp * 50)
            logger.info(f"  {name:25s}: {imp:.4f} {bar}")
        
        # Save model
        self.save_model(test_acc, weighted_f1, precision, recall, f1, use_smote, use_cost_sensitive)
        
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        
        return True
    
    def save_model(self, accuracy, weighted_f1, precision, recall, f1, 
                   used_smote, used_cost_sensitive):
        """Save model and update metadata"""
        # Save to file
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'class_weights': self.class_weights,
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist()
            },
            'used_smote': used_smote,
            'used_cost_sensitive': used_cost_sensitive,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"\nModel saved to {self.model_path}")
        
        # Update database metadata
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()
            
            # Deactivate old models
            cur.execute("UPDATE qppe_model_metadata SET is_active = FALSE")
            
            # Get sample count
            cur.execute("SELECT COUNT(*) FROM qppe_training_data")
            sample_count = cur.fetchone()[0]
            
            # Insert new model with per-class metrics
            cur.execute("""
                INSERT INTO qppe_model_metadata 
                (model_version, training_samples, accuracy, weighted_f1,
                 fast_precision, fast_recall, fast_f1,
                 medium_precision, medium_recall, medium_f1,
                 slow_precision, slow_recall, slow_f1,
                 used_smote, used_cost_sensitive, class_weights, is_active)
                VALUES 
                ((SELECT COALESCE(MAX(model_version), 0) + 1 FROM qppe_model_metadata),
                 %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
            """, (
                sample_count, accuracy, weighted_f1,
                float(precision[0]), float(recall[0]), float(f1[0]),
                float(precision[1]), float(recall[1]), float(f1[1]),
                float(precision[2]), float(recall[2]), float(f1[2]),
                used_smote, used_cost_sensitive,
                str(self.class_weights) if self.class_weights else None
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            logger.info("Model metadata saved to database")
        except Exception as e:
            logger.warning(f"Could not update model metadata: {e}")
    
    def load_model(self):
        """Load model from disk"""
        if not Path(self.model_path).exists():
            logger.warning(f"Model file not found: {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.class_weights = data.get('class_weights')
                
                logger.info(f"Model loaded from {self.model_path}")
                logger.info(f"  Accuracy: {data.get('accuracy', 'unknown'):.4f}")
                logger.info(f"  Weighted F1: {data.get('weighted_f1', 'unknown'):.4f}")
                
                if 'per_class_metrics' in data:
                    metrics = data['per_class_metrics']
                    logger.info("  Per-class F1:")
                    for i, f1 in enumerate(metrics['f1']):
                        cls_name = ['Fast', 'Medium', 'Slow'][i]
                        logger.info(f"    {cls_name}: {f1:.3f}")
                
                return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, features):
        """Make prediction"""
        with self.lock:
            if self.model is None:
                logger.warning("No model loaded, returning default class")
                return 1
            
            try:
                features_array = np.array(features).reshape(1, -1)
                features_array = np.nan_to_num(features_array, nan=0.0, 
                                              posinf=1e10, neginf=0.0)
                features_scaled = self.scaler.transform(features_array)
                prediction = self.model.predict(features_scaled)[0]
                
                self.stats['predictions'] += 1
                self.stats['class_predictions'][int(prediction)] += 1
                
                return int(prediction)
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                self.stats['errors'] += 1
                return 1
    
    def handle_client(self, client_socket):
        """Handle prediction request via TCP"""
        try:
            # Read feature count
            count_data = client_socket.recv(64)
            if not count_data:
                return
            
            feature_count = int(count_data.decode().strip())
            
            if feature_count <= 0 or feature_count > 20:
                logger.warning(f"Invalid feature count: {feature_count}")
                return
            
            # Read features
            feature_bytes = client_socket.recv(feature_count * 8)
            if len(feature_bytes) != feature_count * 8:
                return
            
            features = struct.unpack(f'{feature_count}d', feature_bytes)
            prediction = self.predict(features)
            
            # Send response
            client_socket.send(struct.pack('i', prediction))
            
        except Exception as e:
            logger.error(f"Client handling error: {e}")
            self.stats['errors'] += 1
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def print_stats(self):
        """Print service statistics"""
        uptime = time.time() - self.stats['start_time']
        logger.info(f"\nService Statistics:")
        logger.info(f"  Uptime: {uptime:.0f}s")
        logger.info(f"  Total predictions: {self.stats['predictions']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        if self.stats['predictions'] > 0:
            logger.info(f"  Success rate: {100*(1-self.stats['errors']/self.stats['predictions']):.1f}%")
            logger.info(f"  Predictions by class:")
            for cls in [0, 1, 2]:
                cls_name = ['Fast', 'Medium', 'Slow'][cls]
                count = self.stats['class_predictions'][cls]
                pct = 100 * count / self.stats['predictions']
                logger.info(f"    {cls_name}: {count} ({pct:.1f}%)")
    
    def start_server(self):
        """Start TCP socket server (Windows-compatible)"""
        # Load or train model
        if not self.load_model():
            logger.warning("No model found, please train first with --train-first")
            logger.info("Starting anyway with default predictions...")
        
        # Create TCP socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(10)
        server.settimeout(1.0)
        
        logger.info("=" * 80)
        logger.info(f"Enhanced QPPE Service Started (TCP)")
        logger.info(f"Listening on: {self.host}:{self.port}")
        logger.info(f"Model: {'Loaded' if self.model else 'Not loaded'}")
        logger.info("=" * 80)
        
        try:
            while self.running:
                try:
                    client, addr = server.accept()
                    logger.debug(f"Connection from {addr}")
                    thread = threading.Thread(
                        target=self.handle_client,
                        args=(client,),
                        daemon=True
                    )
                    thread.start()
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            logger.info("\nShutdown requested...")
        finally:
            server.close()
            self.print_stats()
            logger.info("Service stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced QPPE ML Service with SMOTE and Cost-Sensitive Learning (Windows)'
    )
    parser.add_argument('--host', default='127.0.0.1',
                       help='TCP host address')
    parser.add_argument('--port', type=int, default=5555,
                       help='TCP port number')
    parser.add_argument('--db-name', default='tpch', help='Database name')
    parser.add_argument('--db-user', default='postgres', help='Database user')
    parser.add_argument('--db-password', default='postgres', help='Database password')
    parser.add_argument('--db-host', default='localhost', help='Database host')
    parser.add_argument('--db-port', type=int, default=5432, help='Database port')
    parser.add_argument('--model', default='qppe_model_enhanced.pkl', help='Model file path')
    parser.add_argument('--train-first', action='store_true',
                       help='Train model before starting server')
    parser.add_argument('--train-only', action='store_true',
                       help='Train model and exit')
    parser.add_argument('--no-smote', action='store_true',
                       help='Disable SMOTE')
    parser.add_argument('--no-cost-sensitive', action='store_true',
                       help='Disable cost-sensitive learning')
    
    args = parser.parse_args()
    
    db_config = {
        'dbname': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'host': args.db_host,
        'port': args.db_port
    }
    
    service = EnhancedQPPEService(
        host=args.host,
        port=args.port,
        db_config=db_config,
        model_path=args.model
    )
    
    if args.train_only or args.train_first:
        use_smote = not args.no_smote
        use_cost_sensitive = not args.no_cost_sensitive
        
        if not service.train_model(use_smote=use_smote, 
                                   use_cost_sensitive=use_cost_sensitive):
            logger.error("Training failed")
            sys.exit(1)
        
        if args.train_only:
            return
    
    try:
        service.start_server()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
