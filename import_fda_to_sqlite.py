
import csv
import sqlite3
from pathlib import Path

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TEMP_DB = PROCESSED_DIR / "fda_import_temp.db"
FINAL_DB = PROCESSED_DIR / "fda.db"
REJECTED_FILE = PROCESSED_DIR / "ApplicationDocs_rejected_rows.tsv"

BATCH_SIZE = 5000

# --------------------------------------------------
# 2. Expected FDA files
# --------------------------------------------------

FDA_FILES = [
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


def quote_identifier(name):
    """Safely quote a SQLite table or column name."""
    return '"' + name.replace('"', '""') + '"'


# --------------------------------------------------
# 3. Import one FDA text file
# --------------------------------------------------

def import_file(connection, file_path, rejected_writer):
    table_name = file_path.stem
    imported_rows = 0
    rejected_rows = 0

    with open(
        file_path,
        "r",
        encoding="latin1",
        newline=""
    ) as file:

        reader = csv.reader(file, delimiter="\t")

        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError(f"{file_path.name} is empty.")

        headers = [
            header.strip().lstrip("\ufeff")
            for header in headers
        ]

        if not headers or any(not h for h in headers):
            raise ValueError(
                f"Invalid or empty column name in {file_path.name}"
            )

        if len(set(headers)) != len(headers):
            raise ValueError(
                f"Duplicate column names in {file_path.name}"
            )

        column_count = len(headers)

        # Store all values as TEXT to preserve FDA identifiers.
        columns_sql = ", ".join(
            f"{quote_identifier(header)} TEXT"
            for header in headers
        )

        connection.execute(
            f"DROP TABLE IF EXISTS {quote_identifier(table_name)}"
        )

        connection.execute(
            f"CREATE TABLE {quote_identifier(table_name)} "
            f"({columns_sql})"
        )

        placeholders = ", ".join(["?"] * column_count)

        insert_sql = (
            f"INSERT INTO {quote_identifier(table_name)} "
            f"VALUES ({placeholders})"
        )

        batch = []

        for line_number, row in enumerate(reader, start=2):

            # Repair known ApplicationDocs formatting issue.
            if (
                table_name == "ApplicationDocs"
                and len(row) == column_count + 1
                and row[-2] == ""
            ):
                row.pop(-2)

            # Remove an extra trailing empty field.
            elif (
                len(row) == column_count + 1
                and row[-1] == ""
            ):
                row.pop()

            # Pad rows that contain fewer fields.
            if len(row) < column_count:
                row.extend([""] * (column_count - len(row)))

            # Reject rows with unexpected extra fields.
            if len(row) != column_count:
                rejected_rows += 1

                rejected_writer.writerow([
                    file_path.name,
                    line_number,
                    repr(row)
                ])
                continue

            batch.append(row)

            if len(batch) >= BATCH_SIZE:
                connection.executemany(insert_sql, batch)
                imported_rows += len(batch)
                batch.clear()

        # Insert remaining rows.
        if batch:
            connection.executemany(insert_sql, batch)
            imported_rows += len(batch)

    return imported_rows, rejected_rows


# --------------------------------------------------
# 4. Import all 12 files into a temporary database
# --------------------------------------------------

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    missing_files = [
        name for name in FDA_FILES
        if not (RAW_DIR / name).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing FDA files in data/raw:\n"
            + "\n".join(missing_files)
        )

    if TEMP_DB.exists():
        TEMP_DB.unlink()

    if TEMP_DB.with_name(TEMP_DB.name + "-wal").exists():
        TEMP_DB.with_name(TEMP_DB.name + "-wal").unlink()

    if TEMP_DB.with_name(TEMP_DB.name + "-shm").exists():
        TEMP_DB.with_name(TEMP_DB.name + "-shm").unlink()

    total_rows = 0
    total_rejected = 0
    results = []

    try:
        with open(
            REJECTED_FILE,
            "w",
            encoding="utf-8",
            newline=""
        ) as rejected_file:

            rejected_writer = csv.writer(
                rejected_file,
                delimiter="\t"
            )

            rejected_writer.writerow([
                "source_file",
                "line_number",
                "row_data"
            ])

            with sqlite3.connect(TEMP_DB, timeout=60) as connection:
                connection.execute("PRAGMA journal_mode=DELETE")

                for filename in FDA_FILES:
                    file_path = RAW_DIR / filename

                    print(f"\nImporting {filename} ...")

                    rows, rejected = import_file(
                        connection,
                        file_path,
                        rejected_writer
                    )

                    total_rows += rows
                    total_rejected += rejected

                    results.append(
                        (filename, rows, rejected)
                    )

                    print(f"  Imported rows: {rows:,}")
                    print(f"  Rejected rows: {rejected:,}")

                connection.commit()

                # Verify that all 12 tables exist.
                table_count = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name NOT LIKE 'sqlite_%'
                    """
                ).fetchone()[0]

                if table_count != len(FDA_FILES):
                    raise RuntimeError(
                        f"Expected {len(FDA_FILES)} tables, "
                        f"but found {table_count}."
                    )

                # Verify actual row counts in each table.
                for filename, expected_rows, _ in results:
                    table_name = Path(filename).stem

                    actual_rows = connection.execute(
                        f"SELECT COUNT(*) FROM "
                        f"{quote_identifier(table_name)}"
                    ).fetchone()[0]

                    if actual_rows != expected_rows:
                        raise RuntimeError(
                            f"Row count mismatch in {table_name}: "
                            f"{actual_rows} != {expected_rows}"
                        )

        # Close SQLite connections before replacing the final DB.
       # Close the database connection before promoting the database.
        with sqlite3.connect(TEMP_DB, timeout=60) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.commit()

# The connection is now closed.
# SQLite may leave WAL/SHM files behind, so checkpointing alone
# is not enough to guarantee the database is ready to move.
         
        # Promote the completed database.
        # Do not manually delete WAL/SHM files while SQLite may still use them.
        TEMP_DB.replace(FINAL_DB)

    except Exception:
        print(
            "\nImport failed. The existing final database, "
            "if any, has not intentionally been replaced."
        )
        raise

    print("\n" + "=" * 50)
    print("IMPORT COMPLETE")
    print("=" * 50)
    print(f"Imported tables: {len(FDA_FILES)}")
    print(f"Total imported rows: {total_rows:,}")
    print(f"Total rejected rows: {total_rejected:,}")
    print(f"Database location: {FINAL_DB}")
    print(f"Rejected-row report: {REJECTED_FILE}")

    if total_rejected:
        print(
            "\nPlease inspect the rejected-row report "
            "before continuing."
        )


if __name__ == "__main__":
    main()
