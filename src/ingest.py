import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
import logging

load_dotenv()

logging.basicConfig(
    filename="ingest_audit.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

COLLECTION = "documento_colecao"
PG_CONN_STR = os.getenv("PG_CONN_STR")  # connection string SQLAlchemy FORMAT


def normalize(text):
    return (
        text.replace("\t", " ")
            .replace("  ", " ")
            .replace("\n\n", "\n")
            .strip()
    )


def extract_pdf(path="document.pdf"):
    reader = PdfReader(path)
    pages = []

    for page in reader.pages:
        raw = page.extract_text()
        if raw:
            pages.append(normalize(raw))

    return pages


def split_chunks(raw_pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", ";"]
    )

    docs = []

    for page in raw_pages:
        parts = splitter.split_text(page)
        for p in parts:
            docs.append(Document(page_content=p))

    return docs


def ingest():
    logging.info("=== INÍCIO DA INGESTÃO ===")

    raw = extract_pdf()
    logging.info(f"PDF carregado. {len(raw)} páginas processadas.")

    docs = split_chunks(raw)
    logging.info(f"Total de {len(docs)} chunks gerados.")

    embedding = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL")
    )

    # 🚀 CORRETO: connection=PG_CONN_STR
    PGVector.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=COLLECTION,
        connection=PG_CONN_STR,
        pre_delete_collection=True
    )

    logging.info("Ingestão concluída.")
    logging.info("==============================")


if __name__ == "__main__":
    ingest()
