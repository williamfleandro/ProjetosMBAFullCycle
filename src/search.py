import os
import re
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

PG_CONN_STR = os.getenv("PG_CONN_STR")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION = "documento_colecao"


def extract_exact_company_block(text, question):
    """Extrai apenas o bloco referente à empresa perguntada."""
    empresa = re.findall(r"empresa\s+([A-Za-z0-9À-ú ]+)", question, re.IGNORECASE)
    empresa = empresa[0].strip() if empresa else None

    if not empresa:
        return ""

    padrao_linha = r"[A-Za-zÀ-ú0-9\. ]+\s+R\$[\s0-9\.\,]+\s+\d{4}"
    linhas = re.findall(padrao_linha, text)

    for linha in linhas:
        if empresa.lower() in linha.lower():
            return linha.strip()

    return ""


def format_faturamento(linha: str) -> str:
    """
    Converte a linha bruta do PDF para uma frase natural.
    Exemplo de entrada:
        'SuperTechIABrazil R$ 10.000.000,00 2025'
    Exemplo de saída:
        'O faturamento foi de 10 milhões de reais.'
    """

    # captura valor monetário
    match = re.search(r"R\$\s*([\d\.\,]+)", linha)
    if not match:
        return "Informação encontrada, mas não consegui interpretar o faturamento."

    valor_str = match.group(1)

    # converte para número float
    valor_float = float(valor_str.replace(".", "").replace(",", "."))

    # formata para milhões/bilhões
    if valor_float >= 1_000_000_000:
        valor_formatado = f"{valor_float/1_000_000_000:.1f} bilhões"
    elif valor_float >= 1_000_000:
        valor_formatado = f"{valor_float/1_000_000:.1f} milhões"
    else:
        valor_formatado = f"{valor_float:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    return f"O faturamento foi de {valor_formatado} de reais."


def search_prompt(question: str):
    embed_model = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vs = PGVector(
        embeddings=embed_model,
        connection=PG_CONN_STR,
        collection_name=COLLECTION
    )

    docs = vs.similarity_search_with_score(question, k=5)

    docs_ordenados = sorted(docs, key=lambda x: x[1])
    linha_relevante = ""

    for doc, score in docs_ordenados:
        if score > 0.40:
            continue

        linha = extract_exact_company_block(doc.page_content, question)
        if linha:
            linha_relevante = linha
            break

    if not linha_relevante:
        return "Nenhum dado encontrado."

    # converte para frase natural
    return format_faturamento(linha_relevante)
