# Archivo: modules/db_manager.py

import streamlit as st
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Obtener la URL de conexión desde secrets.toml
# Asegúrate de que en .streamlit/secrets.toml tengas:
# DATABASE_URL = "postgresql://postgres..."
try:
    DATABASE_URL = st.secrets["DATABASE_URL"]
except Exception:
    # Fallback para desarrollo local si no hay secrets (opcional)
    DATABASE_URL = None

def get_engine():
    """Crea y retorna el motor de conexión SQLAlchemy."""
    if not DATABASE_URL:
        return None
    try:
        # Creamos el motor. echo=False para producción.
        engine = create_engine(DATABASE_URL, echo=False)
        return engine
    except Exception as e:
        print(f"Error creando engine: {e}")
        return None

def init_db():
    """
    Inicializa la tabla de preferencias en PostgreSQL si no existe.
    Llamar a esto al inicio de la app (una sola vez o en caché).
    """
    engine = get_engine()
    if engine is not None:
        try:
            with engine.connect() as conn:
                # Sintaxis PostgreSQL para crear tabla
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id SERIAL PRIMARY KEY,
                        username TEXT NOT NULL,
                        preference_key TEXT NOT NULL,
                        preference_value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                # Crear índice para búsquedas rápidas por usuario
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_user_pref ON user_preferences (username, preference_key);
                """))
                conn.commit()
        except SQLAlchemyError as e:
            print(f"Error inicializando DB: {e}")

def save_user_preference(username, key, value):
    """
    Guarda o actualiza una preferencia.
    Ejemplo: save_user_preference("usuario1", "last_station", "Medellín")
    """
    engine = get_engine()
    if engine is not None:
        try:
            # Serializar si es objeto complejo (dict/list)
            if isinstance(value, (dict, list)):
                val_str = json.dumps(value)
            else:
                val_str = str(value)

            with engine.connect() as conn:
                # Usamos UPSERT (Insert or Update) compatible con Postgres
                # Nota: En Postgres puro es INSERT ... ON CONFLICT
                # Aquí hacemos una lógica simple: Borrar anterior e insertar nuevo (más fácil de mantener)
                
                # 1. Borrar si existe
                conn.execute(text("""
                    DELETE FROM user_preferences 
                    WHERE username = :user AND preference_key = :key
                """), {"user": username, "key": key})
                
                # 2. Insertar nuevo
                conn.execute(text("""
                    INSERT INTO user_preferences (username, preference_key, preference_value)
                    VALUES (:user, :key, :val)
                """), {"user": username, "key": key, "val": val_str})
                
                conn.commit()
            return True
        except SQLAlchemyError as e:
            st.error(f"Error guardando preferencia: {e}")
            return False
    return False

def get_user_preference(username, key, default=None):
    """
    Recupera una preferencia específica.
    """
    engine = get_engine()
    if engine is not None:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT preference_value FROM user_preferences 
                    WHERE username = :user AND preference_key = :key
                    LIMIT 1
                """), {"user": username, "key": key}).fetchone()
                
                if result:
                    val = result[0]
                    # Intentar deserializar JSON si parece serlo
                    try:
                        return json.loads(val)
                    except:
                        return val # Retornar como string si no es JSON
        except SQLAlchemyError as e:
            print(f"Error leyendo DB: {e}")
            
    return default

def get_all_user_preferences(username):
    """Retorna todas las preferencias de un usuario como diccionario."""
    engine = get_engine()
    prefs = {}
    if engine is not None:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT preference_key, preference_value FROM user_preferences 
                    WHERE username = :user
                """), {"user": username}).fetchall()
                
                for row in result:
                    k, v = row
                    try:
                        prefs[k] = json.loads(v)
                    except:
                        prefs[k] = v
        except Exception:
            pass
    return prefs
