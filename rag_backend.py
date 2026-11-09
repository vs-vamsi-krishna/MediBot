
"""Retrieval and Gemini response logic for MedRAG."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from google import genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from websockets import route

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
    MAX_CONTEXT_CHARS,
    MAX_OUTPUT_TOKENS,
    NO_FDA_DATA_MESSAGE,
)
from hybrid_retrieval import search_products, search_submissions


MEDICAL_CUES = (
    "fever", "temperature", "cough", "cold", "headache", "pain",
    "rash", "vomit", "vomiting", "diarrhea", "diarrhoea",
    "symptom", "symptoms", "what is it used for", "used for",
    "uses of", "what are the uses", "side effect", "side effects",
    "dosage for me", "what tablet", "which medicine", "which tablet",
    "can i take", "treatment for", "home remedy", "natural remedy",
    "molecular formula", "chemical formula", "formula of",
    "drug interaction", "contraindication", "precautions for",
    "medicine for",
)

FDA_CUES = (
    "application", "applno", "sponsor", "marketing status",
    "submission", "submissionno", "product number", "productno",
    "fda record", "fda records", "fda data", "strength",
    "dosage form", "application type", "nda", "anda", "bla",
    "discontinued",
)

SAFE_SOURCE_COLUMNS = [
    "RecordType",
    "ApplNo",
    "ProductNo",
    "DrugName",
    "SponsorName",
    "Strength",
    "Form",
    "MarketingStatusDescription",
    "MarketingStatusID",
    "SubmissionClassCode",
    "SubmissionClassDescription",
    "SubmissionType",
    "SubmissionNo",
    "SubmissionStatus",
    "SubmissionStatusDate",
    "ReviewPriority",
]

_VALUE_CACHE: dict[tuple[int, str], list[str]] = {}


def classify_question(query: str, records=None) -> str:
    q = query.casefold().strip()

    medical_cues = (
        "fever", "temperature", "cough", "cold",
        "headache", "pain", "rash", "vomiting",
        "diarrhea", "symptom", "symptoms",
        "side effect", "side effects",
        "what is it used for", "used for", "uses of",
        "what are the uses", "dosage for me",
        "which medicine", "which tablet", "can i take",
        "treatment for", "home remedy",
        "drug interaction", "contraindication",
        "precautions for", "medicine for",
        "diabetes", "asthma", "hypertension",
        "blood pressure", "heart disease"
    )

    medical_phrases = (
        "what is it used for", "used for", "uses of",
        "what are the uses", "side effect",
        "side effects", "molecular formula",
        "chemical formula", "formula of"
    )

    # Explicit FDA questions
    if any(cue in q for cue in FDA_CUES):
        if any(phrase in q for phrase in medical_phrases):
            return "medical"
        return "fda"

    # General medical questions
    if any(cue in q for cue in medical_cues):
        return "medical"

    # Recognize known FDA products, ingredients and sponsors
    if records is not None:
        for column in ("DrugName", "ActiveIngredient", "SponsorName"):
            if _known_value(query, records, column):
                return "fda"

    # Reject everything outside the supported domains
    return "out_of_scope"

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def get_vectorstore(
    embeddings: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Connect to the existing ChromaDB collection."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "ChromaDB folder not found. "
            "Run: python build_vectorstore.py"
        )

    embeddings = embeddings or get_embeddings()

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if store._collection.count() == 0:
        raise FileNotFoundError(
            "ChromaDB collection is empty. "
            "Run: python build_vectorstore.py"
        )

    return store


def get_gemini_client(
    api_key: str | None,
) -> genai.Client | None:
    """Create a Gemini client when an API key is available."""
    if not api_key or not api_key.strip():
        return None

    return genai.Client(api_key=api_key.strip())


def _nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _exact_application_match(
    query: str,
    records: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Find records by an exact application number."""
    if "ApplNo" not in records.columns:
        return pd.DataFrame()

    for appl_no in re.findall(r"\b\d{6,8}\b", query):
        matched = records[
            records["ApplNo"].astype(str).str.strip() == appl_no
        ]

        if not matched.empty:
            return matched.head(k).copy()

    return pd.DataFrame()


def _known_value(
    query: str,
    records: pd.DataFrame,
    column: str,
) -> str | None:
    """Find a known drug, ingredient, or sponsor in the question."""
    if column not in records.columns:
        return None

    cache_key = (id(records), column)

    if cache_key not in _VALUE_CACHE:
        values = _nonempty(records[column]).unique().tolist()

        values = [
            value
            for value in values
            if value
            and value.casefold()
            not in {"not available", "unknown", "nan"}
        ]

        _VALUE_CACHE[cache_key] = sorted(
            values,
            key=len,
            reverse=True,
        )

    q = query.casefold()

    for value in _VALUE_CACHE[cache_key]:
        if len(value) >= 3 and value.casefold() in q:
            return value

    return None


def _known_value_match(
    query: str,
    records: pd.DataFrame,
    column: str,
    k: int,
) -> pd.DataFrame:
    """Match a known value against a structured dataframe column."""
    value = _known_value(query, records, column)

    if value is None:
        return pd.DataFrame()

    mask = _nonempty(records[column]).str.casefold() == value.casefold()
    return records[mask].head(k).copy()


def _apply_structured_filters(
    query: str,
    records: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Apply application type, strength, form, sponsor and status filters."""
    q = query.upper()
    conditions: list[pd.Series] = []

    if "ApplType" in records.columns:
        for app_type in ("ANDA", "NDA", "BLA"):
            if re.search(rf"\b{app_type}\b", q):
                conditions.append(
                    _nonempty(records["ApplType"]).str.upper()
                    == app_type
                )
                break

    if "Strength" in records.columns:
        match = re.search(
            r"\b\d+(?:\.\d+)?\s*(?:%|MG|MCG|G|ML)(?=\W|$)",
            q,
        )

        if match:
            value = re.sub(r"\s+", "", match.group(0)).upper()

            strength_column = (
                _nonempty(records["Strength"])
                .str.upper()
                .str.replace(" ", "", regex=False)
            )

            conditions.append(
                strength_column.str.contains(
                    rf"(?<!\d){re.escape(value)}(?!\d)",
                    regex=True,
                    na=False,
                )
            )

    if "Form" in records.columns:
        for form in (
            "TABLET",
            "CAPSULE",
            "SOLUTION",
            "INJECTION",
            "CREAM",
            "OINTMENT",
            "DROPS",
            "SPRAY",
            "SUSPENSION",
        ):
            if form in q:
                conditions.append(
                    _nonempty(records["Form"])
                    .str.upper()
                    .str.contains(form, na=False)
                )
                break

    if "MarketingStatusDescription" in records.columns:
        status_terms = {
            "DISCONTINUED": "Discontinued",
            "PRESCRIPTION": "Prescription",
            "OVER-THE-COUNTER": "Over-the-counter",
            "OVER THE COUNTER": "Over-the-counter",
            "TENTATIVE APPROVAL": "None (Tentative Approval)",
        }

        for term, description in status_terms.items():
            if term in q:
                conditions.append(
                    _nonempty(
                        records["MarketingStatusDescription"]
                    ).str.casefold()
                    == description.casefold()
                )
                break

    sponsor = _known_value(query, records, "SponsorName")

    if sponsor and "SponsorName" in records.columns:
        conditions.append(
            _nonempty(records["SponsorName"]).str.casefold()
            == sponsor.casefold()
        )

    ingredient = _known_value(query, records, "ActiveIngredient")

    if ingredient and "ActiveIngredient" in records.columns:
        conditions.append(
            _nonempty(records["ActiveIngredient"]).str.casefold()
            == ingredient.casefold()
        )

    if not conditions:
        return pd.DataFrame()

    mask = conditions[0]

    for condition in conditions[1:]:
        mask &= condition

    result = records[mask]

    return result.head(k).copy() if not result.empty else pd.DataFrame()


def retrieve_fda_records(
    query: str,
    records: pd.DataFrame,
    vectorstore: Chroma | None,
    k: int = TOP_K,
) -> pd.DataFrame:
    """Search SQLite and structured records before semantic retrieval."""
    query_lower = query.casefold()
    application_numbers = re.findall(r"\b\d{6,8}\b", query)

    submission_intent = any(
        term in query_lower
        for term in (
            "submission",
            "submission history",
            "submission status",
        )
    )

    if application_numbers and submission_intent:
        result = search_submissions(
            application_numbers[0],
            limit=k,
        )

        if not result.empty:
            result["RecordType"] = "submission"
            return result

    if application_numbers:
        result = search_products(
            application_numbers[0],
            limit=k,
        )

        if not result.empty:
            result["RecordType"] = "product"
            return result

    for column in ("DrugName", "ActiveIngredient", "SponsorName"):
        value = _known_value(query, records, column)

        if value:
            result = search_products(value, limit=k)

            if not result.empty:
                result["RecordType"] = "product"
                return result

    result = _exact_application_match(query, records, k)

    if not result.empty:
        return result

    result = _known_value_match(
        query,
        records,
        "DrugName",
        k,
    )

    if not result.empty:
        return result

    result = _apply_structured_filters(query, records, k)

    if not result.empty:
        return result

    fda_intent = any(
        term in query_lower
        for term in (
            "fda",
            "drug",
            "product",
            "application",
            "sponsor",
            "submission",
            "marketing status",
            "strength",
            "dosage form",
            "ingredient",
            "discontinued",
            "nda",
            "anda",
            "bla",
            "record",
            "lookup",
            "list",
            "find",
            "show",
        )
    )

    if not fda_intent or vectorstore is None:
        return pd.DataFrame()

    try:
        hits = vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
        )
    except Exception as exc:
        print(
            f"Chroma retrieval error: {type(exc).__name__}: {exc}"
        )
        return pd.DataFrame()

    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()

    for doc, score in hits:
        if score < SIMILARITY_THRESHOLD:
            continue

        record_id = str(doc.metadata.get("RecordID", ""))

        if record_id and record_id in seen:
            continue

        if record_id:
            seen.add(record_id)

        accepted.append({
            "RecordType": "product",
            "RecordID": record_id,
            "ApplNo": doc.metadata.get("ApplNo", ""),
            "ApplType": doc.metadata.get("ApplType", ""),
            "SponsorName": doc.metadata.get("SponsorName", ""),
            "ProductNo": doc.metadata.get("ProductNo", ""),
            "DrugName": doc.metadata.get("DrugName", ""),
            "ActiveIngredient": "",
            "Strength": doc.metadata.get("Strength", ""),
            "Form": doc.metadata.get("Form", ""),
            "MarketingStatusID": doc.metadata.get(
                "MarketingStatusID", ""
            ),
            "MarketingStatusDescription": doc.metadata.get(
                "MarketingStatusDescription", ""
            ),
            "document_text": doc.page_content,
            "_score": score,
        })

        if len(accepted) >= k:
            break

    return pd.DataFrame(accepted)


def _format_context(records: pd.DataFrame) -> str:
    """Format only useful FDA fields into compact text for Gemini."""
    fields = [
        ("ApplNo", "Application number"),
        ("ApplType", "Application type"),
        ("SponsorName", "Sponsor"),
        ("ProductNo", "Product number"),
        ("DrugName", "Drug name"),
        ("ActiveIngredient", "Active ingredient"),
        ("Strength", "Strength"),
        ("Form", "Dosage form"),
        ("MarketingStatusID", "Marketing status ID"),
        ("MarketingStatusDescription", "Marketing status"),
        ("SubmissionClassCode", "Submission class code"),
        ("SubmissionClassDescription", "Submission class"),
        ("SubmissionType", "Submission type"),
        ("SubmissionNo", "Submission number"),
        ("SubmissionStatus", "Submission status"),
        ("SubmissionStatusDate", "Submission status date"),
        ("ReviewPriority", "Review priority"),
        ("SubmissionsPublicNotes", "Public notes"),
    ]

    chunks: list[str] = []
    used = 0

    for _, row in records.head(TOP_K).iterrows():
        lines = []

        for column, label in fields:
            value = row.get(column, "")

            if pd.notna(value) and str(value).strip():
                lines.append(f"{label}: {str(value).strip()}")

        if not lines:
            continue

        chunk = "\n".join(lines)
        remaining = MAX_CONTEXT_CHARS - used

        if remaining <= 0:
            break

        chunk = chunk[:remaining]
        chunks.append(chunk)
        used += len(chunk)

        if used >= MAX_CONTEXT_CHARS:
            break

    return "\n\n---\n\n".join(chunks)



def _call_gemini(
    client: genai.Client,
    prompt: str,
    medical: bool = False,
) -> str:
    """Generate text and handle token limits and empty responses."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )

    answer = getattr(response, "text", None)

    # Return the normal response when available.
    if answer and answer.strip():
        return answer.strip()

    # Check whether candidate text parts are available.
    candidates = getattr(response, "candidates", None) or []

    if candidates:
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []

        text_parts = [
            part.text.strip()
            for part in parts
            if getattr(part, "text", None)
            and part.text.strip()
        ]

        if text_parts:
            return "\n".join(text_parts)

        finish_reason = getattr(
            candidate, "finish_reason", "unknown"
        )
    else:
        finish_reason = "no candidates returned"

    print(f"Gemini returned no text. Finish reason: {finish_reason}")

    raise RuntimeError(
        f"Gemini returned no text; finish reason: {finish_reason}"
    )


def _safe_sources(retrieved: pd.DataFrame) -> pd.DataFrame:
    """Hide internal text and scores from the source table."""
    columns = [
        column
        for column in SAFE_SOURCE_COLUMNS
        if column in retrieved.columns
    ]

    return retrieved[columns].head(TOP_K).copy()


def friendly_gemini_error(exc: Exception) -> str:
    """Convert Gemini API failures into useful messages for chatbot users."""
    error = str(exc).lower()

    # Quota, rate limits, and temporary overload
    if (
        "429" in error
        or "resource_exhausted" in error
        or "quota" in error
        or "rate limit" in error
        or "too many requests" in error
    ):
        return (
            "**Gemini API limit reached.** The service has temporarily "
            "limited requests because of quota or high demand. Please try "
            "again later. If your daily free quota is exhausted, wait until "
            "the quota resets or use an API key with available quota."
        )

    # Invalid or revoked API key
    if (
        "api_key_invalid" in error
        or "invalid api key" in error
        or "unauthorized" in error
        or "permission_denied" in error
        or "403" in error
    ):
        return (
            "**Gemini authentication failed.** The configured API key may "
            "be invalid or lack permission for this model. Please try again "
            "later or contact the application administrator."
        )

    # Temporary service problems
    if (
        "503" in error
        or "unavailable" in error
        or "overloaded" in error
        or "500" in error
        or "internal server error" in error
    ):
        return (
            "**Gemini is temporarily unavailable or experiencing high "
            "demand.** Please wait a little and try again."
        )

    # Network and timeout errors
    if (
        "timeout" in error
        or "timed out" in error
        or "connection" in error
    ):
        return (
            "**Unable to connect to Gemini right now.** Please check your "
            "internet connection or try again in a few minutes."
        )

    return (
        "**MedRAG couldn't generate an answer right now.** "
        "Please try again later. If the problem continues, contact the "
        "application administrator."
    )



def answer_question(
    query: str,
    records: pd.DataFrame,
    vectorstore: Chroma | None,
    client: genai.Client | None,
) -> dict[str, Any]:

    route = classify_question(query, records)
    if route == "out_of_scope":
     return {
        "answer": NO_FDA_DATA_MESSAGE,
        "sources": pd.DataFrame(),
        "route": route,
    }

    if client is None:
        return {
            "answer": (
                "Gemini API key is missing. Add GEMINI_API_KEY "
                "to .streamlit/secrets.toml."
            ),
            "sources": pd.DataFrame(),
            "route": route,
        }

    # General medical questions use Gemini directly.
    if route == "medical":
        prompt = f"""You are MedRAG, a helpful medical information assistant.

Answer the user's general medical question clearly and safely.
Use normal, complete sentences in plain text.
Do not use tables, JSON, or unnecessary bullet lists.
Do not claim that information comes from FDA records when no
FDA records were retrieved.
For medicine questions, provide general information and relevant
safety precautions. Do not give a personalized diagnosis.

User question: {query}

Write a concise answer in 2-5 natural sentences.
Answer:"""

        try:
            answer = _call_gemini(
                client,
                prompt,
                medical=True,
            )

            return {
                "answer": answer,
                "sources": pd.DataFrame(),
                "route": route,
            }

        except Exception as e:
            print(f"Medical Gemini error: {type(e).__name__}: {e}")
            return {
                "answer": (
                    "Sorry, I couldn't generate a medical answer right now. "
                    "Please try again later."
                ),
                "sources": pd.DataFrame(),
                "route": route,
            }

    # General questions use Gemini directly without FDA retrieval.
    if route == "general":
        prompt = f"""You are MedRAG, a helpful AI assistant.

Answer the user's general question clearly and accurately.
Use normal, complete sentences in plain text.
Do not claim that information comes from FDA records.
Do not invent real-time information such as weather, news,
or current events if you do not have access to live data.
If the question requires real-time information, explain
that you cannot access live information.

User question: {query}

Write a concise and relevant answer.

Answer:"""

        try:
            answer = _call_gemini(client, prompt)

            return {
                "answer": answer,
                "sources": pd.DataFrame(),
                "route": route,
            }

        except Exception as exc:
            print(
                f"General Gemini error: {type(exc).__name__}: {exc}"
            )

            return {
                "answer": (
                    "Sorry, I couldn't generate a response right now. "
                    "Please try again later."
                ),
                "sources": pd.DataFrame(),
                "route": route,
            }

    # FDA questions use the retrieval pipeline.
    if vectorstore is None:
        return {
            "answer": (
                "FDA search is not ready. Build the vector index "
                "with python build_vectorstore.py."
            ),
            "sources": pd.DataFrame(),
            "route": route,
        }

    try:
        retrieved = retrieve_fda_records(
            query,
            records,
            vectorstore,
            k=TOP_K,
        )

    except Exception as exc:
        print(
            f"FDA retrieval error: {type(exc).__name__}: {exc}"
        )

        return {
            "answer": (
                "FDA record retrieval failed. "
                "Please check the VS Code terminal."
            ),
            "sources": pd.DataFrame(),
            "route": route,
        }

    if retrieved is None or retrieved.empty:
        return {
            "answer": NO_FDA_DATA_MESSAGE,
            "sources": pd.DataFrame(),
            "route": route,
        }

    sources = _safe_sources(retrieved)

    try:
        context = _format_context(retrieved)

    except Exception as exc:
        print(
            f"FDA context formatting error: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "answer": "I couldn't prepare the retrieved FDA records.",
            "sources": sources,
            "route": route,
        }

    if not context or not context.strip():
        return {
            "answer": NO_FDA_DATA_MESSAGE,
            "sources": sources,
            "route": route,
        }

    # FDA prompt stays separate from the medical prompt.

    prompt = f"""You are MedRAG, an assistant for FDA drug information.

Answer the user's question in complete, natural sentences using plain text.
Never generate a table, Markdown table, JSON, or bullet list.
Start directly with the answer.
Explain each requested field supported by the FDA records.
If a requested field is missing, state that it is unavailable in the retrieved record.
Never invent facts or infer why a product was discontinued.
Do not infer current availability from marketing status.
Keep the answer concise but complete, around 2-5 sentences.

FDA records:
{context}

Question: {query}

Answer:"""

    try:
        answer = _call_gemini(client, prompt)

        return {
            "answer": answer,
            "sources": sources,
            "route": route,
        }

    except Exception as exc:
        print(
            f"FDA Gemini error: {type(exc).__name__}: {exc}"
        )

        return {
            "answer": (
                "I couldn't generate an answer right now. "
                "Please check the VS Code terminal for details."
            ),
            "sources": sources,
            "route": route,
        }
