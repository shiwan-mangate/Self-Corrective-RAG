# scripts/check_database.py

import os
import sys
from sqlalchemy import text

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import db_manager
from database.models.base import Base

# Import all models to ensure Base.metadata is fully populated
try:
    import database.models.session
    import database.models.conversation
    import database.models.vector
    import database.models.evaluation_run
    import database.models.knowledge_gap
except ImportError as e:
    print(f"❌ Failed to import database models. Check your model files: {e}")
    sys.exit(1)


def main():
    print("\n========================================")
    print(" SELF-HEALING RAG DATABASE CHECK")
    print("========================================\n")
    
    # ---------------------------------------------------------
    # [1/5] Database Connectivity
    # ---------------------------------------------------------
    print("[1/5] Database Connectivity")
    try:
        db_manager.initialize()
        is_connected = db_manager.check_connection()
        if is_connected:
            print(f"      {'PostgreSQL connection':<30} PASS\n")
        else:
            print(f"      {'PostgreSQL connection':<30} FAIL")
            print("\nDATABASE CHECK FAILED")
            print("Reason: Cannot reach Neon PostgreSQL. Check DATABASE_URL and network.")
            sys.exit(1)
    except Exception as e:
        print(f"      {'PostgreSQL connection':<30} FAIL")
        print(f"\nDATABASE CHECK FAILED\nReason: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # [2/5] Server Information
    # ---------------------------------------------------------
    print("[2/5] Server Information")
    try:
        with db_manager.engine.connect() as conn:
            db_name = conn.execute(text("SELECT current_database();")).scalar()
            pg_version = conn.execute(text("SELECT version();")).scalar()
            # Extract just the version number part for a cleaner display
            short_version = pg_version.split(" on ")[0] if pg_version else "Unknown"
            
            print(f"      {'Database':<30} {db_name}")
            print(f"      {'PostgreSQL':<30} {short_version}\n")
    except Exception as e:
        print(f"      {'Server Information':<30} FAIL ({e})\n")
        print("\nDATABASE CHECK FAILED")
        print("Reason: Failed to retrieve server metadata.")
        sys.exit(1)

    # ---------------------------------------------------------
    # [3/5] Vector Infrastructure
    # ---------------------------------------------------------
    print("[3/5] Vector Infrastructure")
    try:
        with db_manager.engine.connect() as conn:
            ext_check = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';")).fetchone()
            if ext_check:
                print(f"      {'pgvector extension':<30} PASS\n")
            else:
                print(f"      {'pgvector extension':<30} MISSING")
                print("\nDATABASE CHECK FAILED")
                print("Reason: 'pgvector' extension is not installed. Run: CREATE EXTENSION vector;")
                sys.exit(1)
    except Exception as e:
        print(f"      {'pgvector extension':<30} FAIL ({e})\n")
        print("\nDATABASE CHECK FAILED")
        print("Reason: Database query failed while inspecting vector extension.")
        sys.exit(1)

    # ---------------------------------------------------------
    # [4/5] Application Schema
    # ---------------------------------------------------------
    print("[4/5] Application Schema")
    expected_tables = set(Base.metadata.tables.keys())
    
    try:
        with db_manager.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)).fetchall()
            
            actual_tables = {row[0] for row in result}
            missing_tables = expected_tables - actual_tables
            
            print(f"      {'Expected tables':<30} {len(expected_tables)}")
            print(f"      {'Existing tables':<30} {len(actual_tables)}")
            print(f"      {'Missing tables':<30} {len(missing_tables)}")
            
            if missing_tables:
                for t in missing_tables:
                    print(f"      {t:<30} MISSING")
                print(f"      {'Schema validation':<30} FAIL\n")
                print("\nDATABASE CHECK FAILED")
                print("Reason: Required database tables are missing.")
                print("Command: alembic upgrade head")
                print("Run and verify Alembic migrations before subsystem testing.")
                sys.exit(1)
            else:
                print(f"      {'Schema validation':<30} PASS\n")
                
    except Exception as e:
        print(f"      {'Schema validation':<30} FAIL ({e})\n")
        print("\nDATABASE CHECK FAILED")
        print("Reason: Failed to query PostgreSQL schema metadata.")
        sys.exit(1)

    # ---------------------------------------------------------
    # [5/5] SQLAlchemy Session Health
    # ---------------------------------------------------------
    print("[5/5] SQLAlchemy Session")
    try:
        if db_manager.SessionLocal is None:
            raise ValueError("SessionLocal factory was not initialized.")
        print(f"      {'SessionLocal':<30} PASS")
        
        session = db_manager.SessionLocal()
        try:
            # Execute a lightweight read-only operation through the session
            session.execute(text("SELECT 1"))
            print(f"      {'Session query':<30} PASS")
        finally:
            session.close()
            print(f"      {'Session cleanup':<30} PASS\n")
            
    except Exception as e:
        print(f"      {'Session validation':<30} FAIL ({e})\n")
        print("\nDATABASE CHECK FAILED")
        print("Reason: Request-scoped SQLAlchemy session failed to initialize or execute.")
        sys.exit(1)

    # ---------------------------------------------------------
    # Final Verdict
    # ---------------------------------------------------------
    print("========================================")
    print(" DATABASE CHECK PASSED 🎉")
    print("========================================\n")


if __name__ == "__main__":
    main()