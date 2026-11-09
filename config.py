"""Central configuration for MedRAG."""
from pathlib import Path
import os
import streamlit as st

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

PROCESSED_DIR = BASE_DIR / "processed"
PRODUCTS_CSV = PROCESSED_DIR / "product_records.csv"

# Add this new line for hybrid retrieval
FDA_DB_PATH = BASE_DIR / "data" / "processed" / "fda_prepared.db"

CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "medrag_fda_products_v2"

# Keep the LLM configurable: set GEMINI_MODEL in the environment if your
# Google AI Studio account uses a different currently available model.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
TOP_K = int(os.getenv("MEDRAG_TOP_K", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("MEDRAG_SIMILARITY_THRESHOLD", "0.35"))
MAX_CONTEXT_CHARS = int(os.getenv("MEDRAG_MAX_CONTEXT_CHARS", "5000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MEDRAG_MAX_OUTPUT_TOKENS", "1024"))
NO_FDA_DATA_MESSAGE = "No FDA data available for this question."

REQUIRED_FILES = [
    "Applications.txt",
    "Products.txt",
    "MarketingStatus.txt",
    "MarketingStatus_Lookup.txt",
]
