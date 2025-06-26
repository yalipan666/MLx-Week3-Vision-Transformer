import psycopg2
from datetime import datetime
import os

def create_connection():
    """Create a database connection."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('POSTGRES_DB', 'mnist_db'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432')
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def create_table():
    """Create the predictions and model_versions tables if they don't exist."""
    conn = create_connection()
    if conn:
        try:
            cur = conn.cursor()
            # Create predictions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    predicted_digit INTEGER,
                    true_label INTEGER,
                    image_data BYTEA,
                    model_version INTEGER
                )
            """)
            
            # Create model_versions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    version INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    accuracy FLOAT,
                    training_samples INTEGER
                )
            """)
            
            conn.commit()
        except Exception as e:
            print(f"Error creating tables: {e}")
        finally:
            cur.close()
            conn.close()

def log_prediction(predicted_digit, true_label, image_data, model_version):
    """Log a prediction to the database."""
    conn = create_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO predictions (timestamp, predicted_digit, true_label, image_data, model_version)
                VALUES (%s, %s, %s, %s, %s)
            """, (datetime.now(), predicted_digit, true_label, image_data, model_version))
            conn.commit()
        except Exception as e:
            print(f"Error logging prediction: {e}")
        finally:
            cur.close()
            conn.close()

def get_latest_model_version():
    """Get the latest model version number."""
    conn = create_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT MAX(version) FROM model_versions")
            version = cur.fetchone()[0]
            return version if version is not None else 0
        except Exception as e:
            print(f"Error getting model version: {e}")
            return 0
        finally:
            cur.close()
            conn.close()

def get_model_accuracy():
    """Calculate current model sequence-level accuracy."""
    conn = create_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT predicted_digit, true_label
                FROM predictions
                WHERE true_label IS NOT NULL
            """)
            rows = cur.fetchall()
            total = 0
            correct = 0
            for pred, true in rows:
                # Both pred and true are stored as strings (e.g., '1234')
                if pred is not None and true is not None:
                    if pred == true:
                        correct += 1
                    total += 1
            return correct / total if total > 0 else 0
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            return 0
        finally:
            cur.close()
            conn.close() 