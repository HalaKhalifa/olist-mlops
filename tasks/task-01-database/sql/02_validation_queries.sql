-- Olist MLOps - Task 1
-- Database Validation Queries
-- PostgreSQL

-- ============================================================
-- 1. List all tables
-- ============================================================

SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;


-- ============================================================
-- 2. Count rows in each table
-- ============================================================

SELECT 'customers' AS table_name, COUNT(*) AS rows FROM customers
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL
SELECT 'order_payments', COUNT(*) FROM order_payments
UNION ALL
SELECT 'order_reviews', COUNT(*) FROM order_reviews
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'sellers', COUNT(*) FROM sellers
UNION ALL
SELECT 'geolocation', COUNT(*) FROM geolocation
UNION ALL
SELECT 'product_category_translation', COUNT(*)
FROM product_category_translation;


-- ============================================================
-- 3. Sample customer records
-- ============================================================

SELECT *
FROM customers
LIMIT 5;


-- ============================================================
-- 4. Customer + Order JOIN
-- ============================================================

SELECT
    o.order_id,
    o.order_status,
    o.order_purchase_timestamp,
    c.customer_city,
    c.customer_state
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
LIMIT 10;


-- ============================================================
-- 5. Order + Items + Products JOIN
-- ============================================================

SELECT
    o.order_id,
    o.order_purchase_timestamp,
    oi.order_item_id,
    oi.price,
    p.product_category_name
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id
JOIN products p
    ON oi.product_id = p.product_id
LIMIT 10;


-- ============================================================
-- 6. Multi-table JOIN: Orders + Products + Sellers
-- ============================================================

SELECT
    o.order_id,
    p.product_category_name,
    s.seller_city,
    s.seller_state,
    oi.price
FROM orders o
JOIN order_items oi
    ON o.order_id = oi.order_id
JOIN products p
    ON oi.product_id = p.product_id
JOIN sellers s
    ON oi.seller_id = s.seller_id
LIMIT 10;


-- ============================================================
-- 7. Delivery dates relevant to the ML problem
-- ============================================================

SELECT
    order_id,
    order_purchase_timestamp,
    order_delivered_customer_date,
    order_estimated_delivery_date
FROM orders
WHERE order_delivered_customer_date IS NOT NULL
LIMIT 10;
