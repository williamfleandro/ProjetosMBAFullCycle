import os
import sys
import argparse
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

# ---- Embeddings e LLM (OpenAI ou Gemini) -------------------------------
def get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=model)
    else:
        # padrão = OpenAI
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)

def get_chat_model():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    else:
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        return ChatOpenAI(model=model, temperature=temperature)

# ---- Vector Store (Postgres/pgvector) ----------------------------------
def import_pgvector():
    """
    Tenta usar o pacote novo `langchain_postgres` (recomendado).
    Se indisponível, recua para `langchain_community.vectorstores.PGVector`.
    """
    try:
        from langchain_postgres import PGVector
        return PGVector
    except Exception:
        from langchain_community.vectorstores.pgvector import PGVector
        return PGVector

def ensure_pgvector_extension():
    """
    Cria a extensão pgvector se possível (não falha se já existir).
    Requer `psycopg2-binary`.
    """
    import psycopg2
    conn_str_env = os.getenv("PG_CONN_STR", "")
    if not conn_str_env:
        print("ATENÇÃO: defina PG_CONN_STR. Ex.: postgresql+psycopg2://user:pass@host:5432/db")
        return
    # psycopg2 não aceita o prefixo '+psycopg2', então normalizamos se vier no estilo sqlalchemy
    conn_str = conn_str_env.replace("+psycopg2", "")
    try:
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            print("Extensão pgvector verificada/criada.")
    except Exception as e:
        print(f"Não foi possível criar/verificar a extensão pgvector automaticamente: {e}")

def get_vector_store(collection_name: str):
    import inspect, os
    conn_str = os.getenv("PG_CONN_STR")
    if not conn_str:
        raise RuntimeError(
            "Defina PG_CONN_STR (ex.: postgresql+psycopg2://user:pass@localhost:5432/vectordb)"
        )

    embeddings = get_embeddings()
    PGVector = import_pgvector()
    params = inspect.signature(PGVector).parameters

    if "connection" in params:
        # API nova: langchain_postgres.PGVector
        kwargs = {
            "collection_name": collection_name,
            "connection": conn_str,
            "embeddings": embeddings,
            # NÃO passe 'use_jsonb' aqui
        }
        if "create_schema" in params:
            kwargs["create_schema"] = True
        return PGVector(**kwargs)
    else:
        # API antiga: langchain_community.vectorstores.pgvector.PGVector
        return PGVector(
            collection_name=collection_name,
            connection_string=conn_str,
            embedding_function=embeddings,
        )

# ---- Leitura de PDF e chunking -----------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def load_pdfs_from_path(path: Path) -> List[Document]:
    docs: List[Document] = []
    pdf_files = []
    if path.is_file() and path.suffix.lower() == ".pdf":
        pdf_files = [path]
    else:
        pdf_files = sorted([p for p in path.rglob("*.pdf")])

    if not pdf_files:
        print(f"Nenhum PDF encontrado em: {path}")
        return docs

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        # Acrescenta metadados úteis
        for d in pages:
            d.metadata = {
                **(d.metadata or {}),
                "source_file": str(pdf.resolve()),
                "page": d.metadata.get("page", None),
            }
        docs.extend(pages)
    return docs

def split_docs(docs: List[Document], chunk_size=1000, chunk_overlap=150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

# ---- Prompt (rigoroso ao enunciado) ------------------------------------
from langchain.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """Você é um assistente especializado em análise de contratos, notas fiscais e documentos correlatos.
Responda SEMPRE no MESMO idioma da pergunta (se português, use pt-BR), de forma direta e concisa (máx. 2-3 frases).

Regras:
1) Use EXCLUSIVAMENTE o CONTEXTO fornecido abaixo. Não invente fatos.
2) Se a informação não estiver explicitamente no CONTEXTO, responda: "Não tenho informações necessárias para responder."
3) Se houver múltiplos documentos, considere-os em conjunto. Se ainda assim faltar dado, responda: "Não tenho informações necessárias para responder."
4) Para valores monetários, mantenha o formato simples e preserve a moeda original quando não for BRL (ex.: USD 2.3M). Em BRL, use “X milhões de reais”.
5) Se a pergunta exigir nomes, datas, CNPJs ou itens específicos que não estejam no CONTEXTO, responda: "Não tenho informações necessárias para responder."

CONTEXTO:
{context}
"""

USER_PROMPT = """Pergunta do usuário:
{question}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ]
)

# ---- Cadeia RAG (Retriever -> Prompt -> LLM) ---------------------------
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

def format_docs(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        src = d.metadata.get("source_file", "")
        pg = d.metadata.get("page", "")
        head = f"[Fonte: {src} | página: {pg}]"
        blocks.append(head + "\n" + (d.page_content or ""))
    return "\n\n---\n\n".join(blocks)

def build_rag_chain(vs, k=10):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    llm = get_chat_model()
    chain = {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    } | PROMPT | llm | StrOutputParser()
    return chain

# ---- Comandos CLI ------------------------------------------------------
def cmd_ingest(args):
    ensure_pgvector_extension()
    path = Path(args.path)
    collection = args.collection
    print(f"Ingestão: path={path} | coleção={collection}")

    raw_docs = load_pdfs_from_path(path)
    if not raw_docs:
        print("Nada a ingerir.")
        return

    chunks = split_docs(raw_docs, chunk_size=1000, chunk_overlap=150)
    print(f"Total de páginas: {len(raw_docs)} | Total de chunks: {len(chunks)}")

    vs = get_vector_store(collection)
    # add_documents é incremental
    vs.add_documents(chunks)
    print("Ingestão concluída.")

def cmd_query(args):
    collection = args.collection
    question = args.question.strip()
    print(f"PERGUNTA: {question}")

    vs = get_vector_store(collection)
    rag = build_rag_chain(vs, k=args.k)
    answer = rag.invoke(question)

    print("\nRESPOSTA:", answer)

def cmd_warmup(args):
    """Diagnóstico rápido: verifica conexão com DB e provedor de embeddings/LLM."""
    ensure_pgvector_extension()
    # Teste de embeddings
    emb = get_embeddings()
    vec = emb.embed_query("teste de embeddings")
    print(f"Embeddings OK (dim={len(vec)}).")

    # Teste de LLM
    llm = get_chat_model()
    from langchain.prompts import PromptTemplate
    pt = PromptTemplate.from_template("Responda 'ok' se estiver funcionando.")
    txt = pt.format()
    out = llm.invoke(txt)
    print("LLM OK:", out.content.strip() if hasattr(out, "content") else str(out))

# ---- Main --------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Ingestão e Busca Semântica com LangChain + Postgres/pgvector (PDF → RAG via CLI)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ingest
    pi = sub.add_parser("ingest", help="Ingerir PDFs em uma coleção")
    pi.add_argument("--path", required=True, help="Caminho do PDF ou diretório com PDFs")
    pi.add_argument("--collection", required=True, help="Nome da coleção no Postgres")
    pi.set_defaults(func=cmd_ingest)

    # query
    pq = sub.add_parser("ask", help="Perguntar via RAG")
    pq.add_argument("--collection", required=True, help="Nome da coleção no Postgres")
    pq.add_argument("--k", type=int, default=5, help="Quantidade de chunks recuperados (k)")
    pq.add_argument("question", help="Pergunta em linguagem natural")
    pq.set_defaults(func=cmd_query)

    # warmup
    pw = sub.add_parser("warmup", help="Checagens rápidas de ambiente")
    pw.set_defaults(func=cmd_warmup)

    return p

if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)
