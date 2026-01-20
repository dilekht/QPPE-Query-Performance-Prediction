#!/usr/bin/env python3
"""
Enhanced Training Data Generator with Strategic Sampling
Windows-Compatible Version
Targets balanced 1000+ samples across 3 performance classes

Usage:
    python 2_generate_balanced_data.py --db tpch --target-samples 400
"""

import psycopg2
import time
import hashlib
import re
import argparse
import logging
import random
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# Expanded TPC-H query templates
TPCH_QUERIES = {
    'Q1': {
        'sql': """
            SELECT l_returnflag, l_linestatus,
                   sum(l_quantity) as sum_qty,
                   count(*) as count_order
            FROM lineitem
            WHERE l_shipdate <= date '1998-12-01' - interval '{days} days'
            GROUP BY l_returnflag, l_linestatus
            ORDER BY l_returnflag, l_linestatus;
        """,
        'params': {'days': [30, 60, 90, 120, 180]},
        'expected_class': 0  # Fast
    },
    
    'Q3': {
        'sql': """
            SELECT l_orderkey,
                   sum(l_extendedprice * (1 - l_discount)) as revenue,
                   o_orderdate, o_shippriority
            FROM customer, orders, lineitem
            WHERE c_mktsegment = '{segment}'
              AND c_custkey = o_custkey
              AND l_orderkey = o_orderkey
              AND o_orderdate < date '{date}'
              AND l_shipdate > date '{date}'
            GROUP BY l_orderkey, o_orderdate, o_shippriority
            ORDER BY revenue DESC
            LIMIT {limit};
        """,
        'params': {
            'segment': ['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE'],
            'date': ['1995-01-15', '1995-03-15', '1995-06-15', '1995-09-15'],
            'limit': [10, 50, 100]
        },
        'expected_class': 1  # Medium
    },
    
    'Q5': {
        'sql': """
            SELECT n_name,
                   sum(l_extendedprice * (1 - l_discount)) as revenue
            FROM customer, orders, lineitem, supplier, nation, region
            WHERE c_custkey = o_custkey
              AND l_orderkey = o_orderkey
              AND l_suppkey = s_suppkey
              AND c_nationkey = s_nationkey
              AND s_nationkey = n_nationkey
              AND n_regionkey = r_regionkey
              AND r_name = '{region}'
              AND o_orderdate >= date '{year}-01-01'
              AND o_orderdate < date '{year}-12-31'
            GROUP BY n_name
            ORDER BY revenue DESC;
        """,
        'params': {
            'region': ['ASIA', 'EUROPE', 'AMERICA', 'AFRICA', 'MIDDLE EAST'],
            'year': [1993, 1994, 1995, 1996, 1997]
        },
        'expected_class': 2  # Slow
    },
    
    'Q6': {
        'sql': """
            SELECT sum(l_extendedprice * l_discount) as revenue
            FROM lineitem
            WHERE l_shipdate >= date '{year}-01-01'
              AND l_shipdate < date '{year}-12-31'
              AND l_discount between {disc_min} and {disc_max}
              AND l_quantity < {qty};
        """,
        'params': {
            'year': [1993, 1994, 1995, 1996, 1997],
            'disc_min': [0.02, 0.04, 0.05],
            'disc_max': [0.06, 0.07, 0.08],
            'qty': [24, 25, 26]
        },
        'expected_class': 0  # Fast
    },
    
    'Q10': {
        'sql': """
            SELECT c_custkey, c_name,
                   sum(l_extendedprice * (1 - l_discount)) as revenue,
                   c_acctbal, n_name, c_address, c_phone, c_comment
            FROM customer, orders, lineitem, nation
            WHERE c_custkey = o_custkey
              AND l_orderkey = o_orderkey
              AND o_orderdate >= date '{year}-01-01'
              AND o_orderdate < date '{year}-04-01'
              AND l_returnflag = 'R'
              AND c_nationkey = n_nationkey
            GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
            ORDER BY revenue DESC
            LIMIT {limit};
        """,
        'params': {
            'year': [1993, 1994, 1995],
            'limit': [10, 20, 50]
        },
        'expected_class': 1  # Medium
    },
    
    'Q12': {
        'sql': """
            SELECT l_shipmode,
                   sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH'
                       then 1 else 0 end) as high_priority,
                   sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH'
                       then 1 else 0 end) as low_priority
            FROM orders, lineitem
            WHERE o_orderkey = l_orderkey
              AND l_shipmode in ('{mode1}', '{mode2}')
              AND l_commitdate < l_receiptdate
              AND l_shipdate < l_commitdate
              AND l_receiptdate >= date '{year}-01-01'
              AND l_receiptdate < date '{year}-12-31'
            GROUP BY l_shipmode
            ORDER BY l_shipmode;
        """,
        'params': {
            'mode1': ['MAIL', 'SHIP', 'AIR'],
            'mode2': ['TRUCK', 'REG AIR', 'FOB'],
            'year': [1994, 1995, 1996]
        },
        'expected_class': 1  # Medium
    },
    
    'Q14': {
        'sql': """
            SELECT 100.00 * sum(case when p_type like 'PROMO%%'
                    then l_extendedprice * (1 - l_discount) else 0 end)
                   / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
            FROM lineitem, part
            WHERE l_partkey = p_partkey
              AND l_shipdate >= date '{year}-{month}-01'
              AND l_shipdate < date '{year}-{month}-01' + interval '1 month';
        """,
        'params': {
            'year': [1994, 1995, 1996],
            'month': ['01', '04', '07', '10']
        },
        'expected_class': 1  # Medium
    },
}


class BalancedDataGenerator:
    """Generate balanced training data targeting 1000+ samples"""
    
    def __init__(self, db_config, target_per_class=400):
        self.db_config = db_config
        self.target_per_class = target_per_class
        self.class_counts = defaultdict(int)
        self.generated_hashes = set()
        
    def connect(self):
        return psycopg2.connect(**self.db_config)
    
    def get_current_class_counts(self):
        """Get current sample counts per class"""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT performance_class, COUNT(*)
            FROM qppe_training_data
            GROUP BY performance_class
        """)
        counts = {0: 0, 1: 0, 2: 0}
        for row in cur.fetchall():
            counts[row[0]] = row[1]
        cur.close()
        conn.close()
        return counts
    
    def extract_plan_features(self, plan_text):
        """Enhanced feature extraction"""
        features = {
            'num_joins': len(re.findall(r'Join', plan_text)),
            'num_relations': len(re.findall(r'Scan on', plan_text)),
            'join_depth': plan_text.count('->') // 2 if '->' in plan_text else 0,
            'has_subquery': 'SubPlan' in plan_text or 'InitPlan' in plan_text,
            'has_aggregation': 'Aggregate' in plan_text or 'HashAggregate' in plan_text,
            'has_sort': 'Sort' in plan_text,
            'has_hash': 'Hash' in plan_text,
            
            # Join type counts
            'num_hash_joins': len(re.findall(r'Hash Join', plan_text)),
            'num_merge_joins': len(re.findall(r'Merge Join', plan_text)),
            'num_nested_loops': len(re.findall(r'Nested Loop', plan_text)),
            
            # Scan type counts
            'num_seq_scans': len(re.findall(r'Seq Scan', plan_text)),
            'num_index_scans': len(re.findall(r'Index Scan', plan_text)),
        }
        
        # Extract cost and rows
        cost_match = re.search(r'cost=([0-9.]+)\.\.([0-9.]+) rows=([0-9]+)', plan_text)
        if cost_match:
            features['est_cost'] = float(cost_match.group(2))
            features['est_rows'] = int(cost_match.group(3))
        else:
            features['est_cost'] = 0.0
            features['est_rows'] = 0
        
        # Estimate selectivity
        if features['est_rows'] > 0 and features['num_relations'] > 0:
            features['selectivity'] = min(1.0, features['est_rows'] / 1000000.0)
        else:
            features['selectivity'] = 1.0
        
        return features
    
    def classify_performance(self, exec_time_ms):
        """Classify based on execution time"""
        if exec_time_ms < 100:
            return 0  # Fast
        elif exec_time_ms < 1000:
            return 1  # Medium
        else:
            return 2  # Slow
    
    def generate_config_variations(self):
        """Generate diverse configuration sets to create class diversity"""
        # Fast query configurations (favor hash joins, large work_mem)
        fast_configs = [
            {
                'work_mem': "'256MB'",
                'random_page_cost': 1.0,
                'effective_cache_size': "'4GB'",
                'enable_hashjoin': "'on'",
                'enable_mergejoin': "'on'",
                'enable_nestloop': "'off'",
            },
            {
                'work_mem': "'512MB'",
                'random_page_cost': 1.0,
                'effective_cache_size': "'8GB'",
                'enable_hashjoin': "'on'",
                'enable_mergejoin': "'on'",
                'enable_nestloop': "'on'",
            }
        ]
        
        # Medium query configurations (balanced)
        medium_configs = [
            {
                'work_mem': "'64MB'",
                'random_page_cost': 2.0,
                'effective_cache_size': "'2GB'",
                'enable_hashjoin': "'on'",
                'enable_mergejoin': "'on'",
                'enable_nestloop': "'on'",
            },
            {
                'work_mem': "'128MB'",
                'random_page_cost': 1.5,
                'effective_cache_size': "'3GB'",
                'enable_hashjoin': "'on'",
                'enable_mergejoin': "'off'",
                'enable_nestloop': "'on'",
            },
            {
                'work_mem': "'32MB'",
                'random_page_cost': 2.5,
                'effective_cache_size': "'1GB'",
                'enable_hashjoin': "'on'",
                'enable_mergejoin': "'on'",
                'enable_nestloop': "'on'",
            }
        ]
        
        # Slow query configurations (small work_mem, favor nested loops)
        slow_configs = [
            {
                'work_mem': "'4MB'",
                'random_page_cost': 4.0,
                'effective_cache_size': "'512MB'",
                'enable_hashjoin': "'off'",
                'enable_mergejoin': "'off'",
                'enable_nestloop': "'on'",
            },
            {
                'work_mem': "'8MB'",
                'random_page_cost': 4.0,
                'effective_cache_size': "'1GB'",
                'enable_hashjoin': "'on'",
                'enable_mergejoin': "'off'",
                'enable_nestloop': "'on'",
            },
            {
                'work_mem': "'2MB'",
                'random_page_cost': 5.0,
                'effective_cache_size': "'256MB'",
                'enable_hashjoin': "'off'",
                'enable_mergejoin': "'off'",
                'enable_nestloop': "'on'",
            }
        ]
        
        return {
            0: fast_configs,   # For generating fast queries
            1: medium_configs, # For generating medium queries
            2: slow_configs    # For generating slow queries
        }
    
    def execute_and_collect(self, query_name, query_template, params, config, target_class):
        """Execute query and collect training data"""
        # Format query with parameters
        try:
            query = query_template.format(**params)
        except KeyError as e:
            logger.error(f"Missing parameter: {e}")
            return None
        
        # Generate hash
        config_str = str(sorted(config.items()))
        query_hash = hashlib.md5(
            f"{query_name}_{str(params)}_{config_str}".encode()
        ).hexdigest()[:16]
        
        # Skip if already generated
        if query_hash in self.generated_hashes:
            return None
        
        conn = self.connect()
        conn.autocommit = True
        cur = conn.cursor()
        
        try:
            # Apply configuration
            for param, value in config.items():
                cur.execute(f"SET {param} = {value}")
            
            # Get plan
            cur.execute(f"EXPLAIN {query}")
            plan = '\n'.join([row[0] for row in cur.fetchall()])
            
            # Execute with timing
            cur.execute(f"EXPLAIN (ANALYZE, TIMING ON, BUFFERS ON) {query}")
            result = cur.fetchall()
            result_text = '\n'.join([r[0] for r in result])
            
            # Extract execution time
            exec_time_match = re.search(r'Execution Time: ([0-9.]+) ms', result_text)
            if not exec_time_match:
                return None
            
            exec_time = float(exec_time_match.group(1))
            
            # Extract planning time
            plan_time_match = re.search(r'Planning Time: ([0-9.]+) ms', result_text)
            plan_time = float(plan_time_match.group(1)) if plan_time_match else 0.0
            
            # Extract features
            features = self.extract_plan_features(plan)
            actual_class = self.classify_performance(exec_time)
            
            # Only keep if it matches target class (with some flexibility)
            if abs(actual_class - target_class) > 1:
                return None
            
            # Calculate work_mem in KB
            work_mem_str = config['work_mem'].strip("'")
            if 'MB' in work_mem_str:
                work_mem_kb = int(work_mem_str.replace('MB', '')) * 1024
            elif 'GB' in work_mem_str:
                work_mem_kb = int(work_mem_str.replace('GB', '')) * 1024 * 1024
            else:
                work_mem_kb = 0
            
            features.update({
                'query_hash': query_hash,
                'query_template': query_name,
                'actual_time': exec_time,
                'planning_time': plan_time,
                'performance_class': actual_class,
                'work_mem_kb': work_mem_kb,
                'random_page_cost': config['random_page_cost'],
            })
            
            # Insert to database
            cur.execute("""
                INSERT INTO qppe_training_data
                (query_hash, query_template, num_joins, num_relations, join_depth,
                 est_rows, est_cost, has_subquery, has_aggregation, has_sort, has_hash,
                 num_hash_joins, num_merge_joins, num_nested_loops,
                 num_seq_scans, num_index_scans, selectivity,
                 work_mem_kb, random_page_cost, actual_time, planning_time, performance_class)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                features['query_hash'], features['query_template'],
                features['num_joins'], features['num_relations'], features['join_depth'],
                features['est_rows'], features['est_cost'],
                features['has_subquery'], features['has_aggregation'],
                features['has_sort'], features['has_hash'],
                features['num_hash_joins'], features['num_merge_joins'], features['num_nested_loops'],
                features['num_seq_scans'], features['num_index_scans'], features['selectivity'],
                features['work_mem_kb'], features['random_page_cost'],
                features['actual_time'], features['planning_time'], features['performance_class']
            ))
            
            self.generated_hashes.add(query_hash)
            self.class_counts[actual_class] += 1
            
            class_name = ['FAST', 'MEDIUM', 'SLOW'][actual_class]
            logger.info(f"[OK] {query_name:6s} [{class_name:6s}] {exec_time:7.1f}ms | "
                       f"Classes: F={self.class_counts[0]} M={self.class_counts[1]} S={self.class_counts[2]}")
            
            return features
            
        except Exception as e:
            logger.debug(f"Skipped {query_name}: {e}")
            return None
        finally:
            cur.close()
            conn.close()
    
    def generate_balanced_data(self):
        """Generate balanced training data"""
        logger.info("=" * 80)
        logger.info("QPPE Balanced Training Data Generation")
        logger.info("=" * 80)
        
        # Get current counts
        current_counts = self.get_current_class_counts()
        self.class_counts = defaultdict(int, current_counts)
        
        logger.info(f"Current samples: Fast={current_counts[0]}, "
                   f"Medium={current_counts[1]}, Slow={current_counts[2]}")
        
        # Calculate targets
        targets = {
            0: self.target_per_class,
            1: self.target_per_class,
            2: self.target_per_class
        }
        
        logger.info(f"Target samples per class: {self.target_per_class}")
        logger.info("=" * 80)
        
        # Get configuration variations
        config_sets = self.generate_config_variations()
        
        # Generate data for each class
        max_attempts = 5000
        attempts = 0
        
        while attempts < max_attempts and any(self.class_counts[c] < targets[c] for c in [0, 1, 2]):
            # Prioritize underrepresented classes
            needs_more = [c for c in [0, 1, 2] if self.class_counts[c] < targets[c]]
            if not needs_more:
                break
            target_class = min(needs_more, key=lambda c: self.class_counts[c])
            
            # Find queries likely to produce this class
            candidate_queries = [
                (name, info) for name, info in TPCH_QUERIES.items()
                if info['expected_class'] == target_class
            ]
            
            if not candidate_queries:
                candidate_queries = list(TPCH_QUERIES.items())
            
            # Select random query and parameters
            query_name, query_info = random.choice(candidate_queries)
            
            # Generate random parameters
            params = {}
            for param, values in query_info['params'].items():
                params[param] = random.choice(values)
            
            # Select appropriate config
            config = random.choice(config_sets[target_class])
            
            # Execute
            self.execute_and_collect(
                query_name,
                query_info['sql'],
                params,
                config,
                target_class
            )
            
            attempts += 1
            
            if attempts % 100 == 0:
                logger.info(f"Attempts: {attempts}/{max_attempts}")
                logger.info(f"Progress: F={self.class_counts[0]}/{targets[0]}, "
                           f"M={self.class_counts[1]}/{targets[1]}, "
                           f"S={self.class_counts[2]}/{targets[2]}")
        
        # Final stats
        logger.info("=" * 80)
        logger.info("Generation Complete!")
        logger.info(f"Total generated: {sum(self.class_counts.values())} samples")
        logger.info(f"  Fast: {self.class_counts[0]}")
        logger.info(f"  Medium: {self.class_counts[1]}")
        logger.info(f"  Slow: {self.class_counts[2]}")
        
        # Show database stats
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute("SELECT * FROM qppe_training_stats")
            stats = cur.fetchone()
            cur.close()
            conn.close()
            
            if stats:
                logger.info("\nDatabase Statistics:")
                logger.info(f"  Total samples: {stats[0]}")
                logger.info(f"  Fast: {stats[1]} ({stats[4]}%)")
                logger.info(f"  Medium: {stats[2]} ({stats[5]}%)")
                logger.info(f"  Slow: {stats[3]} ({stats[6]}%)")
                logger.info(f"  Balance ratio: {stats[7]}")
        except Exception as e:
            logger.warning(f"Could not fetch stats: {e}")
        
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate balanced QPPE training data'
    )
    parser.add_argument('--db', default='tpch', help='Database name')
    parser.add_argument('--user', default='postgres', help='DB user')
    parser.add_argument('--password', default='postgres', help='DB password')
    parser.add_argument('--host', default='localhost', help='DB host')
    parser.add_argument('--port', type=int, default=5432, help='DB port')
    parser.add_argument('--target-samples', type=int, default=400,
                       help='Target samples per class')
    
    args = parser.parse_args()
    
    db_config = {
        'dbname': args.db,
        'user': args.user,
        'password': args.password,
        'host': args.host,
        'port': args.port
    }
    
    generator = BalancedDataGenerator(db_config, args.target_samples)
    generator.generate_balanced_data()


if __name__ == '__main__':
    main()
