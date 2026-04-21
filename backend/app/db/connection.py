import mysql.connector
from mysql.connector import pooling

# Using a connection pool for better performance in a real-time app
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "attendance_system"
}
connection_pool = None
try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="attendance_pool",
        pool_size=5,
        **db_config
    )
except mysql.connector.Error as err:
    print(f"❌ Failed to create Connection Pool: {err}")

def get_connection():
    if connection_pool is None:
        raise Exception("Database connection pool was never initialized. Check your MySQL status.")
    return connection_pool.get_connection()