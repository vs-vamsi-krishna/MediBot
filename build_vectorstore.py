"""Build the local ChromaDB index. Run once after extracting and cleaning the FDA data."""
from __future__ import annotations

import sys

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from data_processing import load_or_prepare_records


def main() -> None:
    records = load_or_prepare_records()
    print(f"Prepared product records: {len(records):,}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=60)
    docs = []
    ids = []
    for _, row in records.iterrows():
        base = Document(
            page_content=str(row["document_text"]),
            metadata={
                "RecordID": str(row.get("RecordID", "")),
                "ApplNo": str(row.get("ApplNo", "")),
                "ProductNo": str(row.get("ProductNo", "")),
                "DrugName": str(row.get("DrugName", "")),
                "ApplType": str(row.get("ApplType", "")),
                "SponsorName": str(row.get("SponsorName", "")),
                "Strength": str(row.get("Strength", "")),
                "Form": str(row.get("Form", "")),
                "MarketingStatusID": str(row.get("MarketingStatusID", "")),
                "MarketingStatusDescription": str(row.get("MarketingStatusDescription", "")),
                "source_table": "FDA product/application/status join",
            },
        )
        chunks = splitter.split_documents([base])
        for chunk_index, chunk in enumerate(chunks):
            docs.append(chunk)
            ids.append(f"{base.metadata['RecordID']}::chunk{chunk_index}")

    if not docs:
        raise RuntimeError("No documents were created. Check the FDA TXT files.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    batch_size = 400
    for start in range(0, len(docs), batch_size):
        vectorstore.add_documents(
            documents=docs[start:start + batch_size],
            ids=ids[start:start + batch_size],
        )
        done = min(start + batch_size, len(docs))
        print(f"Indexed {done:,}/{len(docs):,} chunks", end="\r", flush=True)
    print(f"\nChromaDB index ready: {len(docs):,} chunks in {CHROMA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Index build failed: {exc}", file=sys.stderr)
        raise
