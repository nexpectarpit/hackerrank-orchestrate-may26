from __future__ import annotations

"""
Central configuration for the support triage agent.

Loads environment variables from .env and defines all constants,
paths, model settings, and classification taxonomies used across
the project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# code/ directory (where this file lives)
CODE_DIR = Path(__file__).resolve().parent.parent

# Repository root (one level above code/)
REPO_ROOT = CODE_DIR.parent

# Data corpus root
DATA_DIR = REPO_ROOT / "data"

# Support tickets directory
TICKETS_DIR = REPO_ROOT / "support_tickets"

# Input and output CSV paths
INPUT_CSV = TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"
SAMPLE_CSV = TICKETS_DIR / "sample_support_tickets.csv"

# FAISS index cache directory (inside code/ so it is easy to clean)
INDEX_DIR = CODE_DIR / "index_cache"

# Decision traces output directory
TRACES_DIR = CODE_DIR / "traces"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Load .env from the code/ directory
_env_path = CODE_DIR / ".env"
load_dotenv(dotenv_path=_env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))

# Stage toggles -- set to "false" to skip and save tokens
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
ENABLE_CRITIC = os.getenv("ENABLE_CRITIC", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Model / retrieval settings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_TOP_K = 15          # chunks retrieved before re-ranking
RERANK_TOP_K = 5          # chunks kept after re-ranking
QUERY_EXPANSION_COUNT = 2  # extra queries generated per ticket
CHUNK_MAX_TOKENS = 512     # soft limit per chunk (characters, not tokens)

# Rate-limiting: seconds to wait between Groq API calls
API_CALL_DELAY = 2.5

# ---------------------------------------------------------------------------
# Classification taxonomies
# ---------------------------------------------------------------------------

ALLOWED_STATUSES = {"replied", "escalated"}

ALLOWED_REQUEST_TYPES = {
    "product_issue",
    "feature_request",
    "bug",
    "invalid",
}

# Supported companies
SUPPORTED_COMPANIES = {"HackerRank", "Claude", "Visa"}

# ---------------------------------------------------------------------------
# Output CSV schema
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "Issue",
    "Subject",
    "Company",
    "Response",
    "Product Area",
    "Status",
    "Request Type",
    "Justification",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
