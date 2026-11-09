"""Read, clean and join FDA tables into compact searchable product records."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import pandas as pd

from config import RAW_DATA_DIR, PROCESSED_DIR, PRODUCTS_CSV, REQUIRED_FILES

ALL_TABLE_FILES = [
    "ActionTypes_Lookup.txt",
    "ApplicationDocs.txt",
    "Applications.txt",
    "ApplicationsDocsType_Lookup.txt",
    "Join_Submission_ActionTypes_Lookup.txt",
    "MarketingStatus.txt",
    "MarketingStatus_Lookup.txt",
    "Products.txt",
    "SubmissionClass_Lookup.txt",
    "SubmissionPropertyType.txt",
    "Submissions.txt",
    "TE.txt",
]


def _read_application_docs(path: Path) -> pd.DataFrame:
    """Read ApplicationDocs and repair the known extra-empty-field row shape."""
    with path.open("r", encoding="latin1", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows = []
        for row in reader:
            # In this export, a malformed row may contain one extra empty field
            # immediately before the date. Remove only that known empty field.
            if len(row) == len(header) + 1 and len(row) >= 2 and row[-2] == "":
                row.pop(-2)
            if len(row) < len(header):
                row += [""] * (len(header) - len(row))
            elif len(row) > len(header):
                # Preserve alignment by rejecting unexpected malformed records.
                continue
            rows.append(row)
    return pd.DataFrame(rows, columns=header, dtype=str)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and normalize empty/null-like cells without changing IDs."""
    frame = frame.copy()
    for col in frame.columns:
        frame[col] = frame[col].astype("string").str.strip()
        frame[col] = frame[col].replace({"": pd.NA, "NULL": pd.NA, "null": pd.NA})
    return frame


def load_all_tables(data_dir: Path = RAW_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load available FDA text tables. IDs remain strings to preserve leading zeros."""
    data_dir = Path(data_dir)
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required FDA file(s): " + ", ".join(missing)
            + f". Extract the ZIP into: {data_dir}"
        )

    tables: Dict[str, pd.DataFrame] = {}
    for filename in ALL_TABLE_FILES:
        path = data_dir / filename
        if not path.exists():
            continue
        if filename == "ApplicationDocs.txt":
            frame = _read_application_docs(path)
        else:
            frame = pd.read_csv(
                path, sep="\t", dtype=str, encoding="latin1", engine="python"
            )
        tables[filename.removesuffix(".txt")] = clean_frame(frame)
    return tables


def _require_columns(frame: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def build_product_records(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join application/product/status records while limiting columns to useful fields."""
    applications = tables["Applications"]
    products = tables["Products"]
    statuses = tables["MarketingStatus"]
    status_lookup = tables["MarketingStatus_Lookup"]

    _require_columns(applications, ["ApplNo", "ApplType", "SponsorName"], "Applications")
    _require_columns(products, ["ApplNo", "ProductNo", "DrugName", "ActiveIngredient", "Strength", "Form"], "Products")
    _require_columns(statuses, ["ApplNo", "ProductNo", "MarketingStatusID"], "MarketingStatus")
    _require_columns(status_lookup, ["MarketingStatusID", "MarketingStatusDescription"], "MarketingStatus_Lookup")

    # De-duplicate lookup keys only; do not arbitrarily remove legitimate products.
    apps = applications.drop_duplicates(subset=["ApplNo"], keep="first")
    lookup = status_lookup.drop_duplicates(subset=["MarketingStatusID"], keep="first")
    status_rows = statuses.drop_duplicates(
        subset=["ApplNo", "ProductNo", "MarketingStatusID"], keep="first"
    )

    merged = products.merge(apps, on="ApplNo", how="left", validate="many_to_one")
    merged = merged.merge(status_rows, on=["ApplNo", "ProductNo"], how="left")
    merged = merged.merge(lookup, on="MarketingStatusID", how="left", validate="many_to_one")

    wanted = [
        "ApplNo", "ApplType", "SponsorName", "ProductNo", "Form", "Strength",
        "DrugName", "ActiveIngredient", "MarketingStatusID", "MarketingStatusDescription",
    ]
    for col in wanted:
        if col not in merged.columns:
            merged[col] = pd.NA
    records = merged[wanted].copy()
    records = records.fillna("")

    # Remove rows with no useful product/application identifier and exact duplicates.
    records = records[(records["ApplNo"].str.strip() != "") & (records["ProductNo"].str.strip() != "")]
    records = records.drop_duplicates().reset_index(drop=True)
    records.insert(0, "RecordID", records.apply(
        lambda r: "|".join([r["ApplNo"], r["ProductNo"], r["MarketingStatusID"]]), axis=1
    ))
    records["document_text"] = records.apply(_record_to_text, axis=1)
    return records


def _record_to_text(row: pd.Series) -> str:
    fields = [
        ("FDA application number", "ApplNo"),
        ("Application type", "ApplType"),
        ("Sponsor", "SponsorName"),
        ("Product number", "ProductNo"),
        ("Drug name", "DrugName"),
        ("Active ingredient", "ActiveIngredient"),
        ("Strength", "Strength"),
        ("Dosage form", "Form"),
        ("Marketing status ID", "MarketingStatusID"),
        ("Marketing status", "MarketingStatusDescription"),
    ]
    lines = [f"{label}: {str(row[col]).strip()}" for label, col in fields if str(row[col]).strip()]
    return "\n".join(lines)


def prepare_dataset(data_dir: Path = RAW_DATA_DIR, output_path: Path = PRODUCTS_CSV) -> pd.DataFrame:
    tables = load_all_tables(data_dir)
    records = build_product_records(tables)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_path, index=False, encoding="utf-8")
    return records


def load_or_prepare_records() -> pd.DataFrame:
    if PRODUCTS_CSV.exists():
        frame = pd.read_csv(PRODUCTS_CSV, dtype=str, keep_default_na=False)
        if "document_text" in frame.columns and not frame.empty:
            return frame
    return prepare_dataset()


if __name__ == "__main__":
    prepared = prepare_dataset()
    print(f"Saved {len(prepared):,} product/application records to {PRODUCTS_CSV}")
