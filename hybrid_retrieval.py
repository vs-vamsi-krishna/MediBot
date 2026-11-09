
"""Efficient SQLite retrieval for MedRAG."""

import sqlite3
import pandas as pd

from config import FDA_DB_PATH


PRODUCT_COLUMNS = [
    "ApplNo",
    "ApplType",
    "SponsorName",
    "ProductNo",
    "DrugName",
    "ActiveIngredient",
    "Strength",
    "Form",
    "MarketingStatusID",
    "MarketingStatusDescription",
]

SUBMISSION_COLUMNS = [
    "ApplNo",
    "SubmissionClassCode",
    "SubmissionClassDescription",
    "SubmissionType",
    "SubmissionNo",
    "SubmissionStatus",
    "SubmissionStatusDate",
    "ReviewPriority",
    "SubmissionsPublicNotes",
]


def search_products(query: str, limit: int = 5) -> pd.DataFrame:
    """Retrieve a small set of relevant FDA product records."""

    query = query.strip()
    if not query:
        return pd.DataFrame(columns=PRODUCT_COLUMNS)

    limit = max(1, min(limit, 10))
    pattern = f"%{query}%"

    sql = """
        SELECT
            ApplNo, ApplType, SponsorName, ProductNo,
            DrugName, ActiveIngredient, Strength, Form,
            MarketingStatusID, MarketingStatusDescription
        FROM product_records
        WHERE TRIM(ApplNo) = ?
           OR DrugName = ? COLLATE NOCASE
           OR ActiveIngredient = ? COLLATE NOCASE
           OR SponsorName = ? COLLATE NOCASE
           OR DrugName LIKE ? COLLATE NOCASE
           OR ActiveIngredient LIKE ? COLLATE NOCASE
        LIMIT ?
    """

    try:
        with sqlite3.connect(str(FDA_DB_PATH), timeout=10) as conn:
            return pd.read_sql_query(
                sql,
                conn,
                params=(
                    query, query, query, query,
                    pattern, pattern, limit,
                ),
            )
    except sqlite3.Error as exc:
        print(f"FDA product search error: {exc}")
        return pd.DataFrame(columns=PRODUCT_COLUMNS)


def search_submissions(
    application_number: str,
    limit: int = 5,
) -> pd.DataFrame:
    """Retrieve a limited submission history for an application."""

    application_number = application_number.strip()
    if not application_number:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)

    limit = max(1, min(limit, 10))

    sql = """
        SELECT
            ApplNo, SubmissionClassCode,
            SubmissionClassDescription, SubmissionType,
            SubmissionNo, SubmissionStatus,
            SubmissionStatusDate, ReviewPriority,
            SubmissionsPublicNotes
        FROM submission_records
        WHERE TRIM(ApplNo) = ?
        ORDER BY SubmissionStatusDate DESC
        LIMIT ?
    """

    try:
        with sqlite3.connect(str(FDA_DB_PATH), timeout=10) as conn:
            return pd.read_sql_query(
                sql,
                conn,
                params=(application_number, limit),
            )
    except sqlite3.Error as exc:
        print(f"FDA submission search error: {exc}")
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)
