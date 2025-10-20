# Archivo: modules/db_manager.py

import sqlite3
import json

DATABASE_FILE = "preferences.db"

def create_connection():
    """Crea y retorna una conexión a la base de datos SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table():
    """Crea la tabla de preferencias si no existe."""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    widget_type TEXT NOT NULL,
                    widget_params TEXT NOT NULL
                );
            """)
            conn.commit()
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()

def save_preference(username, widget_type, params):
    """Guarda una nueva preferencia de widget para un usuario."""
    conn = create_connection()
    if conn is not None:
        try:
            params_json = json.dumps(params)
            sql = ''' INSERT INTO preferences(username, widget_type, widget_params)
                      VALUES(?,?,?) '''
            cursor = conn.cursor()
            cursor.execute(sql, (username, widget_type, params_json))
            conn.commit()
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()

def get_preferences(username):
    """Recupera todas las preferencias de widgets para un usuario."""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT widget_type, widget_params FROM preferences WHERE username=?", (username,))
            rows = cursor.fetchall()
            # Convierte los parámetros de JSON string a diccionario
            preferences = [{'type': row[0], 'params': json.loads(row[1])} for row in rows]
            return preferences
        except sqlite3.Error as e:
            print(e)
            return []
        finally:
            conn.close()
    return []
