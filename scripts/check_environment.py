# scripts/check_environment.py

import os
import sys

# Ensure the root project directory is in the Python path
# so 'config' and other modules can be imported correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def mask_secret(value: str) -> str:
    """Safely masks a secret string, revealing only the last 4 characters."""
    val_str = str(value).strip()
    if not val_str:
        return ""
    if len(val_str) <= 6:
        return "***"
    return f"...{val_str[-4:]}"

def main():
    print("\n========================================")
    print(" SELF-HEALING RAG ENVIRONMENT CHECK")
    print("========================================\n")

    # 1. Validate the Configuration Import Path
    try:
        from config.settings import settings
        print(f"{'Settings Import':<25} OK")
    except Exception as e:
        print(f"{'Settings Import':<25} FAILED")
        print(f"\nEnvironment Check FAILED")
        print(f"Configuration import error: {e}")
        sys.exit(1)

    # 2. Define Checks (Value, Is_Secret)
    # Using getattr for models in case they are defined as hardcoded constants
    # instead of strictly required Pydantic settings.
    checks = {
        "Database URL": (getattr(settings, 'NEON_VECTOR_DATABASE_URL', None), True),
        "Groq API Key": (getattr(settings, 'GROQ_API_KEY', None), True),
        "Tavily API Key": (getattr(settings, 'TAVILY_API_KEY', None), True),
        "Groq Model": (getattr(settings, 'GROQ_MODEL', 'llama-3.3-70b-versatile'), False),
        "Embedding Model": (getattr(settings, 'EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5'), False),
    }

    missing_critical = []

    # 3. Inspect Each Setting
    for name, (value, is_secret) in checks.items():
        if not value:
            status = "MISSING"
            missing_critical.append(name)
        else:
            if is_secret:
                status = f"CONFIGURED (ends in {mask_secret(value)})"
            else:
                status = f"CONFIGURED ({value})"
        
        print(f"{name:<25} {status}")

    print("\n========================================")
    
    # 4. Final Aggregation and Fast-Fail
    if missing_critical:
        print(" Environment Check FAILED 💥\n")
        print(" Missing critical configuration:")
        for missing in missing_critical:
            print(f" - {missing}")
        print("\n No API call should happen.")
        sys.exit(1)
    else:
        print(" Environment Check PASSED 🎉")

if __name__ == "__main__":
    main()