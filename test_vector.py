import psycopg
from pgvector.psycopg import register_vector

# Connect to your active chatvectorv1 database
with psycopg.connect("postgresql://postgres:871031@localhost:5432/chatvectorv1") as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT 1;")
        print("Database and pgvector connection successful!")