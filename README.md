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

## ⚙️ Installation & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vs-vamsi-krishna/MediBot.git
cd MediBot
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create `.streamlit/secrets.toml` and add:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
HF_TOKEN = "your_huggingface_token"
```

Never commit this file to GitHub.

### 5. Prepare Required Data

Ensure the prepared FDA product records are available at:

```text
processed/product_records.csv
```

Ensure the FDA SQLite database is available at:

```text
data/processed/fda_prepared.db
```

The ChromaDB vector store can be downloaded from the Hugging Face dataset repository when the application starts.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Cloud Deployment

MedRAG is deployed using **Streamlit Community Cloud**, with GitHub for source-code management and Hugging Face for ChromaDB storage.

### Deployment Workflow

1. Push the project source code to a GitHub repository.
2. Upload the ChromaDB vector store to a Hugging Face Dataset repository.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/).
4. Connect your GitHub account.
5. Select the MedRAG repository and `main` branch.
6. Set the main application file to `app.py`.
7. Configure the following secrets in Streamlit Cloud:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
HF_TOKEN = "your_huggingface_token"
```

8. Select Python 3.12 and deploy the application.

### Deployment Architecture

- **GitHub:** Stores application source code and prepared FDA data.
- **Hugging Face:** Hosts the ChromaDB vector store.
- **Streamlit Community Cloud:** Hosts the web application.
- **Google Gemini:** Generates responses using retrieved context.

### Live Application

[https://medibot-qs6s97g9styalqhapkpejx.streamlit.app/]
