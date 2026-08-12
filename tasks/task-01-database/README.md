# Task 1 — Get the Data Into a Database

## Overview

This task is the first hands-on stage of the MLOps Training 2026/2027.

The objective is to take the Brazilian E-Commerce Public Dataset by Olist, understand its relational structure, load the data into a PostgreSQL database, and verify that the database can be queried and joined successfully.

This task establishes the data foundation for the rest of the MLOps project.

## Dataset

The project uses the **Brazilian E-Commerce Public Dataset by Olist**, a public dataset containing information about orders made through the Olist Brazilian e-commerce platform.

The dataset contains information about:

* Customers
* Orders
* Order items
* Products
* Sellers
* Payments
* Reviews
* Geolocation
* Product category translations

The raw CSV files are not stored in this repository.

## Problem Definition

The long-term goal of this project is to build a machine learning system that predicts whether an order will be **delivered late or on time**.

The key dates for defining the delivery outcome are:

* `order_estimated_delivery_date` — the estimated delivery date
* `order_delivered_customer_date` — the actual delivery date

Conceptually:

```text
Actual delivery date > Estimated delivery date → LATE

Actual delivery date <= Estimated delivery date → ON TIME
```

The target variable and feature engineering will be developed in later tasks.

## Database

**Technology:**

* PostgreSQL 14
* pgAdmin 4
* Local PostgreSQL server

**Database:**

* Database: `olist`
* Host: `localhost`
* Port: `5432`

## Database Tables

| Table                          | Description                                              |
| ------------------------------ | -------------------------------------------------------- |
| `customers`                    | Customer information                                     |
| `orders`                       | Order-level information and delivery dates               |
| `order_items`                  | Products and sellers associated with each order          |
| `products`                     | Product information                                      |
| `sellers`                      | Seller information                                       |
| `order_payments`               | Payment information for orders                           |
| `order_reviews`                | Customer reviews associated with orders                  |
| `geolocation`                  | Geographic information associated with ZIP code prefixes |
| `product_category_translation` | Product category translations                            |

## Main Relationships

The main relationships are:

```text
customers
    │
    │ customer_id
    ▼
orders
    │
    │ order_id
    ▼
order_items
    │
    ├── product_id ──► products
    │
    └── seller_id ───► sellers

orders
    ├── order_id ────► order_payments
    │
    └── order_id ────► order_reviews
```

The complete Entity Relationship Diagram will be added to the project documentation.

## Task Implementation

### Step 1 — Read & Understand

* Reviewed the dataset documentation.
* Studied the dataset structure and tables.
* Identified the relationships between customers, orders, products, and sellers.
* Understood the relationship between estimated and actual delivery dates.
* Understood the eventual late-delivery prediction problem.

### Step 2 — Download & Ingest

* Downloaded the Olist CSV dataset.
* Set up and verified a local PostgreSQL server.
* Created the `olist` database.
* Created relational database tables.
* Imported the CSV data into PostgreSQL.
* Added primary-key and foreign-key constraints.

### Step 3 — Test

* Verified that all tables exist.
* Checked the number of records in each table.
* Queried individual tables.
* Tested relationships using SQL JOINs.
* Tested multi-table JOINs.
* Verified the delivery-related fields required for the future prediction task.

## SQL Files

The SQL used during this task is stored in:

* [`01_create_tables.sql`](sql/01_create_tables.sql)
* [`02_validation_queries.sql`](sql/02_validation_queries.sql)

## Issue Encountered During Ingestion

During the import of the `order_reviews` table, PostgreSQL reported a duplicate primary-key violation for `review_id`.

The schema was adjusted to use a composite primary key:

```text
(review_id, order_id)
```

This allowed the dataset to be imported while maintaining a uniqueness constraint.

## Deliverables

The Task 1 deliverables include:

* Database schema SQL
* Database validation SQL
* Entity Relationship Diagram (ERD)
* Screenshots demonstrating successful database queries
* Task completion report

## Task Status

**Completed**

The Olist dataset has been successfully loaded into a local PostgreSQL relational database and validated through SQL queries and table JOINs.

EDA, feature engineering, and machine learning development will be addressed in later tasks.

