-- =================================================================
-- TPC-H Sample Schema for QPPE Testing
-- Run this if you don't have full TPC-H data
-- File: 0_create_tpch_sample.sql
-- =================================================================

-- Region table
CREATE TABLE IF NOT EXISTS region (
    r_regionkey INTEGER PRIMARY KEY,
    r_name CHAR(25) NOT NULL,
    r_comment VARCHAR(152)
);

-- Nation table
CREATE TABLE IF NOT EXISTS nation (
    n_nationkey INTEGER PRIMARY KEY,
    n_name CHAR(25) NOT NULL,
    n_regionkey INTEGER NOT NULL REFERENCES region(r_regionkey),
    n_comment VARCHAR(152)
);

-- Part table
CREATE TABLE IF NOT EXISTS part (
    p_partkey INTEGER PRIMARY KEY,
    p_name VARCHAR(55) NOT NULL,
    p_mfgr CHAR(25) NOT NULL,
    p_brand CHAR(10) NOT NULL,
    p_type VARCHAR(25) NOT NULL,
    p_size INTEGER NOT NULL,
    p_container CHAR(10) NOT NULL,
    p_retailprice DECIMAL(15,2) NOT NULL,
    p_comment VARCHAR(23) NOT NULL
);

-- Supplier table
CREATE TABLE IF NOT EXISTS supplier (
    s_suppkey INTEGER PRIMARY KEY,
    s_name CHAR(25) NOT NULL,
    s_address VARCHAR(40) NOT NULL,
    s_nationkey INTEGER NOT NULL REFERENCES nation(n_nationkey),
    s_phone CHAR(15) NOT NULL,
    s_acctbal DECIMAL(15,2) NOT NULL,
    s_comment VARCHAR(101) NOT NULL
);

-- Customer table
CREATE TABLE IF NOT EXISTS customer (
    c_custkey INTEGER PRIMARY KEY,
    c_name VARCHAR(25) NOT NULL,
    c_address VARCHAR(40) NOT NULL,
    c_nationkey INTEGER NOT NULL REFERENCES nation(n_nationkey),
    c_phone CHAR(15) NOT NULL,
    c_acctbal DECIMAL(15,2) NOT NULL,
    c_mktsegment CHAR(10) NOT NULL,
    c_comment VARCHAR(117) NOT NULL
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    o_orderkey INTEGER PRIMARY KEY,
    o_custkey INTEGER NOT NULL REFERENCES customer(c_custkey),
    o_orderstatus CHAR(1) NOT NULL,
    o_totalprice DECIMAL(15,2) NOT NULL,
    o_orderdate DATE NOT NULL,
    o_orderpriority CHAR(15) NOT NULL,
    o_clerk CHAR(15) NOT NULL,
    o_shippriority INTEGER NOT NULL,
    o_comment VARCHAR(79) NOT NULL
);

-- Partsupp table
CREATE TABLE IF NOT EXISTS partsupp (
    ps_partkey INTEGER NOT NULL REFERENCES part(p_partkey),
    ps_suppkey INTEGER NOT NULL REFERENCES supplier(s_suppkey),
    ps_availqty INTEGER NOT NULL,
    ps_supplycost DECIMAL(15,2) NOT NULL,
    ps_comment VARCHAR(199) NOT NULL,
    PRIMARY KEY (ps_partkey, ps_suppkey)
);

-- Lineitem table
CREATE TABLE IF NOT EXISTS lineitem (
    l_orderkey INTEGER NOT NULL REFERENCES orders(o_orderkey),
    l_partkey INTEGER NOT NULL REFERENCES part(p_partkey),
    l_suppkey INTEGER NOT NULL REFERENCES supplier(s_suppkey),
    l_linenumber INTEGER NOT NULL,
    l_quantity DECIMAL(15,2) NOT NULL,
    l_extendedprice DECIMAL(15,2) NOT NULL,
    l_discount DECIMAL(15,2) NOT NULL,
    l_tax DECIMAL(15,2) NOT NULL,
    l_returnflag CHAR(1) NOT NULL,
    l_linestatus CHAR(1) NOT NULL,
    l_shipdate DATE NOT NULL,
    l_commitdate DATE NOT NULL,
    l_receiptdate DATE NOT NULL,
    l_shipinstruct CHAR(25) NOT NULL,
    l_shipmode CHAR(10) NOT NULL,
    l_comment VARCHAR(44) NOT NULL,
    PRIMARY KEY (l_orderkey, l_linenumber)
);

-- Insert sample data

-- Regions
INSERT INTO region VALUES (0, 'AFRICA', 'special requests about the carefully bold packages wake');
INSERT INTO region VALUES (1, 'AMERICA', 'furiously regular deposits');
INSERT INTO region VALUES (2, 'ASIA', 'deposits boost after the carefully');
INSERT INTO region VALUES (3, 'EUROPE', 'furiously express accounts');
INSERT INTO region VALUES (4, 'MIDDLE EAST', 'unusual accounts. furiously regular');

-- Nations (25 nations)
INSERT INTO nation VALUES (0, 'ALGERIA', 0, 'special express requests');
INSERT INTO nation VALUES (1, 'ARGENTINA', 1, 'pending theodolites cajole');
INSERT INTO nation VALUES (2, 'BRAZIL', 1, 'y express requests above');
INSERT INTO nation VALUES (3, 'CANADA', 1, 'bold requests. pending theodolites');
INSERT INTO nation VALUES (4, 'EGYPT', 4, 'deposits cajole. pending');
INSERT INTO nation VALUES (5, 'ETHIOPIA', 0, 'lar deposits. careful');
INSERT INTO nation VALUES (6, 'FRANCE', 3, 'express depths. bold');
INSERT INTO nation VALUES (7, 'GERMANY', 3, 'quickly final accounts');
INSERT INTO nation VALUES (8, 'INDIA', 2, 'slyly express requests');
INSERT INTO nation VALUES (9, 'INDONESIA', 2, 'slyly bold requests');
INSERT INTO nation VALUES (10, 'IRAN', 4, 'ironic pinto beans among');
INSERT INTO nation VALUES (11, 'IRAQ', 4, 'ounts cajole slyly');
INSERT INTO nation VALUES (12, 'JAPAN', 2, 'efully final depths');
INSERT INTO nation VALUES (13, 'JORDAN', 4, 'lar deposits boost blithely');
INSERT INTO nation VALUES (14, 'KENYA', 0, 'pending excuses haggle');
INSERT INTO nation VALUES (15, 'MOROCCO', 0, 'slyly unusual pinto beans');
INSERT INTO nation VALUES (16, 'MOZAMBIQUE', 0, 'deposits boost furiously');
INSERT INTO nation VALUES (17, 'PERU', 1, 'platelets haggle express');
INSERT INTO nation VALUES (18, 'CHINA', 2, 'pending foxes sleep slyly');
INSERT INTO nation VALUES (19, 'ROMANIA', 3, 'final pinto beans detect');
INSERT INTO nation VALUES (20, 'SAUDI ARABIA', 4, 'furiously unusual forges');
INSERT INTO nation VALUES (21, 'VIETNAM', 2, 'accounts sleep furiously');
INSERT INTO nation VALUES (22, 'RUSSIA', 3, 'blithely regular deposits');
INSERT INTO nation VALUES (23, 'UNITED KINGDOM', 3, 'final dolphins boost');
INSERT INTO nation VALUES (24, 'UNITED STATES', 1, 'pinto beans boost carefully');

-- Generate sample data using PL/pgSQL
DO $$
DECLARE
    i INTEGER;
    j INTEGER;
    k INTEGER;
    v_part_key INTEGER;
    v_supp_key INTEGER;
    v_cust_key INTEGER;
    v_order_key INTEGER;
    v_nation INTEGER;
    v_price DECIMAL(15,2);
    v_qty DECIMAL(15,2);
    v_discount DECIMAL(15,2);
    v_date DATE;
    v_segment VARCHAR(10);
    v_priority VARCHAR(15);
    v_shipmode VARCHAR(10);
    v_shipinstruct VARCHAR(25);
    v_status CHAR(1);
    v_rflag CHAR(1);
    v_segments VARCHAR(10)[] := ARRAY['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE'];
    v_priorities VARCHAR(15)[] := ARRAY['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW'];
    v_shipmodes VARCHAR(10)[] := ARRAY['MAIL', 'SHIP', 'AIR', 'TRUCK', 'REG AIR', 'FOB', 'RAIL'];
    v_shipinst VARCHAR(25)[] := ARRAY['DELIVER IN PERSON', 'TAKE BACK RETURN', 'COLLECT COD', 'NONE'];
    v_types VARCHAR(25)[] := ARRAY['PROMO BURNISHED COPPER', 'STANDARD POLISHED TIN', 'SMALL PLATED STEEL', 'ECONOMY ANODIZED BRASS', 'MEDIUM BRUSHED NICKEL'];
    v_brands VARCHAR(10)[] := ARRAY['Brand#11', 'Brand#12', 'Brand#13', 'Brand#21', 'Brand#22', 'Brand#23', 'Brand#31', 'Brand#32', 'Brand#33'];
    v_containers VARCHAR(10)[] := ARRAY['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG', 'MED BAG', 'MED BOX', 'LG CASE', 'LG BOX'];
BEGIN
    -- Generate Parts (2000 parts)
    RAISE NOTICE 'Generating parts...';
    FOR i IN 1..2000 LOOP
        v_price := 900 + (random() * 1000)::DECIMAL(15,2);
        INSERT INTO part VALUES (
            i,
            'Part ' || i || ' ' || v_types[1 + (random() * 4)::int],
            'Manufacturer#' || (1 + (random() * 4)::int),
            v_brands[1 + (random() * 8)::int],
            v_types[1 + (random() * 4)::int],
            (1 + random() * 49)::int,
            v_containers[1 + (random() * 7)::int],
            v_price,
            'comment ' || i
        ) ON CONFLICT DO NOTHING;
    END LOOP;
    
    -- Generate Suppliers (100 suppliers)
    RAISE NOTICE 'Generating suppliers...';
    FOR i IN 1..100 LOOP
        v_nation := (random() * 24)::int;
        INSERT INTO supplier VALUES (
            i,
            'Supplier#' || lpad(i::text, 9, '0'),
            'Address ' || i,
            v_nation,
            '25-' || lpad((100 + random() * 899)::int::text, 3, '0') || '-' || lpad((1000 + random() * 8999)::int::text, 4, '0'),
            (-1000 + random() * 11000)::DECIMAL(15,2),
            'supplier comment ' || i
        ) ON CONFLICT DO NOTHING;
    END LOOP;
    
    -- Generate Customers (1500 customers)
    RAISE NOTICE 'Generating customers...';
    FOR i IN 1..1500 LOOP
        v_nation := (random() * 24)::int;
        v_segment := v_segments[1 + (random() * 4)::int];
        INSERT INTO customer VALUES (
            i,
            'Customer#' || lpad(i::text, 9, '0'),
            'Address ' || i,
            v_nation,
            '25-' || lpad((100 + random() * 899)::int::text, 3, '0') || '-' || lpad((1000 + random() * 8999)::int::text, 4, '0'),
            (-1000 + random() * 10000)::DECIMAL(15,2),
            v_segment,
            'customer comment ' || i
        ) ON CONFLICT DO NOTHING;
    END LOOP;
    
    -- Generate PartSupp (cross product sample)
    RAISE NOTICE 'Generating partsupp...';
    FOR i IN 1..2000 LOOP
        FOR j IN 1..4 LOOP
            v_supp_key := 1 + ((i + j - 1) % 100);
            INSERT INTO partsupp VALUES (
                i,
                v_supp_key,
                (1 + random() * 9998)::int,
                (1 + random() * 999)::DECIMAL(15,2),
                'partsupp comment ' || i || '-' || j
            ) ON CONFLICT DO NOTHING;
        END LOOP;
    END LOOP;
    
    -- Generate Orders (15000 orders)
    RAISE NOTICE 'Generating orders...';
    FOR i IN 1..15000 LOOP
        v_cust_key := 1 + (random() * 1499)::int;
        v_date := '1992-01-01'::date + ((random() * 2557)::int);  -- 7 years
        v_priority := v_priorities[1 + (random() * 4)::int];
        
        IF random() < 0.5 THEN v_status := 'O';
        ELSIF random() < 0.75 THEN v_status := 'F';
        ELSE v_status := 'P';
        END IF;
        
        INSERT INTO orders VALUES (
            i,
            v_cust_key,
            v_status,
            0,  -- will update later
            v_date,
            v_priority,
            'Clerk#' || lpad((1 + random() * 999)::int::text, 9, '0'),
            0,
            'order comment ' || i
        ) ON CONFLICT DO NOTHING;
    END LOOP;
    
    -- Generate Lineitem (60000 lineitems, ~4 per order)
    RAISE NOTICE 'Generating lineitems (this may take a minute)...';
    FOR i IN 1..15000 LOOP
        -- Get order date
        SELECT o_orderdate INTO v_date FROM orders WHERE o_orderkey = i;
        IF v_date IS NULL THEN
            CONTINUE;
        END IF;
        
        FOR j IN 1..4 LOOP
            v_part_key := 1 + (random() * 1999)::int;
            v_supp_key := 1 + (random() * 99)::int;
            v_qty := (1 + random() * 49)::DECIMAL(15,2);
            v_price := (900 + random() * 100000)::DECIMAL(15,2) / 100;
            v_discount := (random() * 0.1)::DECIMAL(15,2);
            v_shipmode := v_shipmodes[1 + (random() * 6)::int];
            v_shipinstruct := v_shipinst[1 + (random() * 3)::int];
            
            IF random() < 0.5 THEN v_rflag := 'N';
            ELSIF random() < 0.75 THEN v_rflag := 'R';
            ELSE v_rflag := 'A';
            END IF;
            
            IF random() < 0.5 THEN v_status := 'O';
            ELSE v_status := 'F';
            END IF;
            
            BEGIN
                INSERT INTO lineitem VALUES (
                    i,                                          -- l_orderkey
                    v_part_key,                                 -- l_partkey
                    v_supp_key,                                 -- l_suppkey
                    j,                                          -- l_linenumber
                    v_qty,                                      -- l_quantity
                    v_price * v_qty,                            -- l_extendedprice
                    v_discount,                                 -- l_discount
                    (random() * 0.08)::DECIMAL(15,2),          -- l_tax
                    v_rflag,                                    -- l_returnflag
                    v_status,                                   -- l_linestatus
                    v_date + ((random() * 120)::int),          -- l_shipdate
                    v_date + ((random() * 90)::int),           -- l_commitdate
                    v_date + ((random() * 150)::int),          -- l_receiptdate
                    v_shipinstruct,                             -- l_shipinstruct
                    v_shipmode,                                 -- l_shipmode
                    'lineitem comment ' || i || '-' || j        -- l_comment
                );
            EXCEPTION WHEN OTHERS THEN
                -- Skip on constraint violation
                NULL;
            END;
        END LOOP;
    END LOOP;
    
    -- Update order totals
    RAISE NOTICE 'Updating order totals...';
    UPDATE orders o 
    SET o_totalprice = (
        SELECT COALESCE(SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)), 0)
        FROM lineitem l 
        WHERE l.l_orderkey = o.o_orderkey
    );
    
    RAISE NOTICE 'TPC-H sample data generation complete!';
END $$;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_lineitem_shipdate ON lineitem(l_shipdate);
CREATE INDEX IF NOT EXISTS idx_lineitem_orderkey ON lineitem(l_orderkey);
CREATE INDEX IF NOT EXISTS idx_lineitem_partkey ON lineitem(l_partkey);
CREATE INDEX IF NOT EXISTS idx_lineitem_suppkey ON lineitem(l_suppkey);
CREATE INDEX IF NOT EXISTS idx_orders_orderdate ON orders(o_orderdate);
CREATE INDEX IF NOT EXISTS idx_orders_custkey ON orders(o_custkey);
CREATE INDEX IF NOT EXISTS idx_customer_nationkey ON customer(c_nationkey);
CREATE INDEX IF NOT EXISTS idx_customer_mktsegment ON customer(c_mktsegment);
CREATE INDEX IF NOT EXISTS idx_supplier_nationkey ON supplier(s_nationkey);
CREATE INDEX IF NOT EXISTS idx_nation_regionkey ON nation(n_regionkey);

-- Analyze tables for better query planning
ANALYZE region;
ANALYZE nation;
ANALYZE part;
ANALYZE supplier;
ANALYZE customer;
ANALYZE orders;
ANALYZE partsupp;
ANALYZE lineitem;

-- Report table sizes
SELECT 
    relname as table_name,
    n_live_tup as row_count
FROM pg_stat_user_tables 
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;

DO $$ BEGIN RAISE NOTICE 'TPC-H Sample Schema Created Successfully!'; END $$;
