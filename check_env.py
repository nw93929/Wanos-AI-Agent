"""
Environment Variable Checker
=============================

Run this before starting the API to verify your .env file is set up correctly.

Usage:
    python check_env.py
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def check_env_var(name, required=True):
    """Check if an environment variable is set"""
    value = os.getenv(name)

    if value:
        # Mask the value for security (show first 8 chars)
        masked = value[:8] + "..." if len(value) > 8 else value[:4] + "..."
        status = "✅"
        message = f"Set ({masked})"
    else:
        status = "❌" if required else "⚠️"
        message = "NOT SET" if required else "Optional (not set)"

    print(f"{status} {name:25} {message}")
    return bool(value) if required else True

print("\n" + "="*60)
print(" ENVIRONMENT VARIABLE CHECK")
print("="*60 + "\n")

all_good = True

# Required variables
print("Required:")
all_good &= check_env_var("OPENAI_API_KEY", required=True)
all_good &= check_env_var("PINECONE_API_KEY", required=True)
all_good &= check_env_var("PINECONE_INDEX_NAME", required=True)

print("\nFor Stock Screening:")
all_good &= check_env_var("FMP_API_KEY", required=True)

print("\nOptional (for enhanced features):")
check_env_var("POSTGRES_URI", required=False)
print("  Note: REDIS_URL is optional for persistent result storage")
check_env_var("REDIS_URL", required=False)
print("       If Redis is not installed, remove REDIS_URL from .env")
check_env_var("TAVILY_API_KEY", required=False)
check_env_var("ALPHA_VANTAGE_KEY", required=False)

print("\n" + "="*60)

if all_good:
    print("✅ All required environment variables are set!")
    print("You can now run: python api.py")
else:
    print("❌ Missing required environment variables!")
    print("\nTo fix:")
    print("1. Copy .env.example to .env")
    print("   cp .env.example .env")
    print("\n2. Edit .env and add your actual API keys")
    print("\n3. Re-run this check:")
    print("   python check_env.py")

print("="*60 + "\n")
