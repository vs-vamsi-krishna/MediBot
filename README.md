# MedRAG — refactored FDA RAG chatbot

This version separates the UI, styling, data preparation, vector indexing and RAG backend.

## Files

- `app.py`: Streamlit UI and chat state
- `styles.css`: UI styling only
- `data_processing.py`: loads, cleans and joins FDA tables into compact product/application/status records
- `build_vectorstore.py`: chunks records and builds a persistent ChromaDB index
- `rag_backend.py`: routes questions, retrieves FDA records, and calls Gemini with a short context
- `config.py`: paths, model settings and retrieval limits

## 1. Put the FDA files in place

Extract the ZIP so the 12 `.txt` files are directly inside `data/raw/`, for example:

```text
data/raw/Applications.txt
data/raw/Products.txt
data/raw/MarketingStatus.txt
...
```

The project currently creates a joined product/application/marketing-status search collection. It uses only useful columns in prompts. The other FDA files are loaded and cleaned when present, but this first retrieval index is focused on product/application/status questions.

## 2. Create a virtual environment and install packages

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure Gemini

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, then replace the placeholder with your own API key. Do not commit this file to GitHub.

You can optionally set `GEMINI_MODEL` in your environment if the model name configured in `config.py` is not available to your account.

## 4. Build cleaned records and ChromaDB

```powershell
python -c "from data_processing import prepare_dataset; df=prepare_dataset(); print(df.shape)"
python build_vectorstore.py
```

The vector index is built once and reused by the app. Re-running `build_vectorstore.py` replaces this project's Chroma collection to avoid duplicate embeddings.

## 5. Run the app

```powershell
streamlit run app.py
```

## Retrieval behavior

- FDA questions: exact application/product/drug/sponsor matches and structured filters are attempted before semantic search.
- Semantic search has a relevance threshold to reduce unrelated matches.
- If no FDA records are found, the exact response is `No FDA data available for this question.`
- General medical-information questions are routed to Gemini separately and do not pretend the FDA product table contains clinical advice.
- Only a small number of relevant records and selected fields are sent to Gemini. Raw embeddings and full dataset rows are never sent.

## Important scope note

The supplied FDA export contains regulatory tables, not a complete clinical reference. This initial version can answer product/application/marketing-status questions represented by the joined records. For clinically detailed answers such as contraindications, side effects, dosage instructions and drug interactions, add a vetted drug-label or medical-reference corpus before treating the app as a clinical assistant.


## Notebook included

`notebook.ipynb` is included as a learning/testing notebook. Run its earlier data loading and vector-store setup cells first, then run the final section titled **Updated MedRAG retrieval and answer logic (final version)**. The Streamlit app uses `app.py` + `rag_backend.py`; editing the notebook alone does not change the running app. Both are configured to use `gemini-3.5-flash` by default.
