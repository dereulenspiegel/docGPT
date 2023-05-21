from chromadb.config import Settings

VECTOR_STORE_PATH="./db"
EMBEDDINGS_MODEL_NAME="all-MiniLM-L6-v2"
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=VECTOR_STORE_PATH,
        anonymized_telemetry=False
    )