
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "processed" / "fda.db"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "fda_prepared.db"


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    con = sqlite3.connect(OUTPUT_PATH, timeout=60)

    try:
        con.execute("ATTACH DATABASE ? AS src", (str(DB_PATH),))
        con.execute("PRAGMA temp_store = MEMORY")

        print("1/5 Normalizing application keys...", flush=True)

        con.execute("""
            CREATE TABLE apps AS
            SELECT TRIM(ApplNo) ApplNo,
                   TRIM(ApplType) ApplType,
                   TRIM(SponsorName) SponsorName,
                   TRIM(ApplPublicNotes) ApplPublicNotes
            FROM src.Applications
        """)
        con.execute("CREATE INDEX idx_apps_no ON apps(ApplNo)")

        print("2/5 Normalizing product keys...", flush=True)

        con.execute("""
            CREATE TABLE products AS
            SELECT TRIM(ApplNo) ApplNo,
                   TRIM(ProductNo) ProductNo,
                   TRIM(Form) Form,
                   TRIM(Strength) Strength,
                   TRIM(ReferenceDrug) ReferenceDrug,
                   TRIM(DrugName) DrugName,
                   TRIM(ActiveIngredient) ActiveIngredient,
                   TRIM(ReferenceStandard) ReferenceStandard
            FROM src.Products
        """)
        con.execute(
            "CREATE INDEX idx_products_key ON products(ApplNo, ProductNo)"
        )

        print("3/5 Normalizing marketing status...", flush=True)

        con.execute("""
            CREATE TABLE statuses AS
            SELECT TRIM(MarketingStatusID) MarketingStatusID,
                   TRIM(ApplNo) ApplNo,
                   TRIM(ProductNo) ProductNo
            FROM src.MarketingStatus
        """)
        con.execute(
            "CREATE INDEX idx_status_key ON statuses(ApplNo, ProductNo)"
        )

        con.execute("""
            CREATE TABLE status_lookup AS
            SELECT TRIM(MarketingStatusID) MarketingStatusID,
                   TRIM(MarketingStatusDescription)
                       MarketingStatusDescription
            FROM src.MarketingStatus_Lookup
        """)
        con.execute(
            "CREATE INDEX idx_lookup_id ON status_lookup(MarketingStatusID)"
        )

        print("4/5 Joining product records...", flush=True)

        con.execute("""
            CREATE TABLE product_records AS
            SELECT
                p.ApplNo,
                a.ApplType,
                a.SponsorName,
                p.ProductNo,
                p.DrugName,
                p.ActiveIngredient,
                p.Strength,
                p.Form,
                p.ReferenceDrug,
                p.ReferenceStandard,
                s.MarketingStatusID,
                l.MarketingStatusDescription
            FROM products p
            LEFT JOIN apps a ON p.ApplNo = a.ApplNo
            LEFT JOIN statuses s
                ON p.ApplNo = s.ApplNo
               AND p.ProductNo = s.ProductNo
            LEFT JOIN status_lookup l
                ON s.MarketingStatusID = l.MarketingStatusID
        """)

        con.execute(
            "CREATE INDEX idx_records_applno ON product_records(ApplNo)"
        )
        con.execute(
            "CREATE INDEX idx_records_drugname ON product_records(DrugName)"
        )
        con.execute(
            "CREATE INDEX idx_records_ingredient "
            "ON product_records(ActiveIngredient)"
        )
        con.execute(
            "CREATE INDEX idx_records_sponsor ON product_records(SponsorName)"
        )

        print("5/5 Preparing submission records...", flush=True)

        con.execute("""
            CREATE TABLE submission_records AS
            SELECT
                TRIM(s.ApplNo) ApplNo,
                TRIM(s.SubmissionClassCodeID) SubmissionClassCodeID,
                TRIM(s.SubmissionType) SubmissionType,
                TRIM(s.SubmissionNo) SubmissionNo,
                TRIM(s.SubmissionStatus) SubmissionStatus,
                TRIM(s.SubmissionStatusDate) SubmissionStatusDate,
                TRIM(s.SubmissionsPublicNotes) SubmissionsPublicNotes,
                TRIM(s.ReviewPriority) ReviewPriority,
                TRIM(l.SubmissionClassCode) SubmissionClassCode,
                TRIM(l.SubmissionClassCodeDescription)
                    SubmissionClassDescription
            FROM src.Submissions s
            LEFT JOIN src.SubmissionClass_Lookup l
                ON TRIM(s.SubmissionClassCodeID) =
                   TRIM(l.SubmissionClassCodeID)
        """)

        con.execute("""
            CREATE INDEX idx_submission_records_applno
            ON submission_records(ApplNo)
        """)

        con.commit()

        products_count = con.execute(
            "SELECT COUNT(*) FROM product_records"
        ).fetchone()[0]

        submissions_count = con.execute(
            "SELECT COUNT(*) FROM submission_records"
        ).fetchone()[0]

        print("\nDATA PREPARATION COMPLETE")
        print(f"Product records: {products_count:,}")
        print(f"Submission records: {submissions_count:,}")
        print(f"Output: {OUTPUT_PATH}")

    except Exception:
        con.rollback()
        raise

    finally:
        con.close()


if __name__ == "__main__":
    main()
