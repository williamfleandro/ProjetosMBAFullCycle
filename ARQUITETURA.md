# Arquitetura do Projeto RAG PDF + pgvector

Visão em alto nível da solução:

1. O usuário interage via:
   - Frontend web minimalista (navegador), ou
   - CLI (`src/chat.py`).
2. Em ambos os casos, as perguntas são enviadas para a camada de aplicação:
   - API FastAPI (`src/api.py`) no caso do frontend;
   - Função `search_prompt` (`src/search.py`) no caso da CLI.
3. A aplicação:
   - Converte a pergunta em embedding;
   - Consulta o banco vetorial (Postgres + pgvector) para recuperar os trechos mais relevantes;
   - Monta o contexto e o prompt;
   - Chama o LLM (OpenAI ou Gemini) via LangChain;
   - Retorna a resposta final ao usuário.

## Diagrama (Mermaid)

```mermaid
flowchart LR
    subgraph Cliente
        A[Navegador (Frontend)] -->|HTTP/JSON| B(API FastAPI)
        C[CLI (chat.py)] -->|Função search_prompt| D[Camada RAG (app.py)]
    end

    subgraph Servidor
        B --> D
        D -->|embeddings / similarity search| E[(Postgres + pgvector)]
        D -->|chamada LLM| F[Provedor de IA (OpenAI/Gemini)]
    end

    E:::db
    classDef db fill:#1f2933,stroke:#4b5563,color:#f9fafb;
```

## Componentes principais

- **Frontend (`static/index.html`)**  
  Interface simples em HTML/JS que envia requisições `POST /api/query` para o FastAPI.

- **API FastAPI (`src/api.py`)**  
  Expõe:
  - `POST /api/query` para perguntas RAG;
  - `GET /health` para monitoramento simples.

- **Camada RAG (`app.py`)**  
  Responsável por:
  - Embeddings e LLM (OpenAI/Gemini via LangChain);
  - Conexão com Postgres + pgvector;
  - Construção da cadeia RAG (retriever + prompt + LLM).

- **Banco vetorial (Postgres + pgvector)**  
  Armazena os embeddings dos chunks de `document.pdf` na coleção `documento_colecao`.

- **CLI (`src/chat.py`)**  
  Interface de terminal que reutiliza a mesma lógica de busca (`search_prompt`) e RAG.
