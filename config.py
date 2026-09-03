import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# =========================
# PROJECT PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "health_synthetic_200users_90days.csv"
KB_INDEX_PATH = PROJECT_ROOT / "kb" / "processed" / "kb_index.parquet"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "health_coach.txt"
NO_DATA_PROMPT_PATH = PROJECT_ROOT / "prompts" / "no_data_reply.txt"

# =========================
# MODEL / FEATURE CONSTANTS
# =========================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.2

FEATURE_COLS = [
    "avg_steps_7d",
    "avg_sleep_7d",
    "avg_stress_7d",
    "water_glasses",
    "calories_intake",
    "resting_heart_rate",
    "age",
    "bmi",
]

# =========================
# ENVIRONMENT
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set in your environment. "
        "Create a .env file (see .env.example) or set it before running the app, "
        "e.g. in PowerShell: $env:GROQ_API_KEY = 'your_key_here'"
    )
