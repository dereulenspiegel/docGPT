import os
import argparse
import glob
from typing import List
from config import VECTOR_STORE_PATH, EMBEDDINGS_MODEL_NAME, CHROMA_SETTINGS

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    SitemapLoader,
    SeleniumURLLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from langchain.vectorstores import Chroma

import nest_asyncio

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]

def load_url(source_url: str, is_sitemap: bool) -> List[Document]:
    nest_asyncio.apply()
    if is_sitemap:
        loader = SitemapLoader(web_path=source_url)
    else:
        loader = SeleniumURLLoader(urls=[source_url])
    return loader.load()

def ingest():    
    chunk_size = 500
    chunk_overlap = 50

    parser = argparse.ArgumentParser("ingest")
    parser.add_argument('target')
    parser.add_argument('-s', '--sitemap', action='store_true')
    args = parser.parse_args()
    if args.target.startswith('http'):
        print("Loading from url")
        documents = load_url(args.target, args.sitemap)
    else:
        print(f"Loading documents from {args.target}")
        documents = load_documents(args.target)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {args.target}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, embeddings, persist_directory=VECTOR_STORE_PATH, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

if __name__ == "__main__":
    ingest()
