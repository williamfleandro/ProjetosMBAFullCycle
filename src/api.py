from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import ensure_pgvector_extension, get_vector_store, build_rag_chain


class QueryRequest(BaseModel):
    question: str
    k: int = 10
    collection: str = "documento_colecao"


class QueryResponse(BaseModel):
    answer: str


app = FastAPI(title="Ingestão e Busca Semântica com LangChain e Postgres", version="1.0.0")


@app.on_event("startup")
def on_startup():
    # Garante que a extensão pgvector exista
    ensure_pgvector_extension()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Endpoint principal de busca via RAG (PDF + pgvector)."""
    vs = get_vector_store(req.collection)
    rag = build_rag_chain(vs, k=req.k)
    answer = rag.invoke(req.question)
    return QueryResponse(answer=answer)


# ---- Static / Frontend ---------------------------------------------------

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
