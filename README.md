# Projeto RAG com pgvector, FastAPI e Frontend Minimalista

Este projeto realiza:

- Ingestão de um PDF (`document.pdf`);
- Geração de embeddings e armazenamento no PostgreSQL + pgvector;
- Busca por similaridade (k=10) sobre o banco vetorial;
- Geração de respostas via LLM (RAG);
- Interface de chat via CLI;
- API HTTP com FastAPI;
- Frontend web minimalista consumindo a API;
- Dockerfile para empacotar o app.

A coleção usada no banco vetorial é:

```text
documento_colecao
```

---

## 1. Preparação do ambiente Python (opcional, fora de Docker)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

Copie o arquivo `.env.example` para `.env` e preencha suas chaves:

```bash
cp .env.example .env
```

---

## 2. Subir o banco de dados (Postgres + pgvector)

```bash
docker compose up -d
```

O serviço do banco ficará acessível como `pgvector-db` na porta definida no `docker-compose.yml`
(por padrão, 5432 mapeado para a máquina host).

---

## 3. Ingestão do PDF

Certifique-se de que o arquivo `document.pdf` está na raiz do projeto.

Execute:

```bash
python src/ingest.py
```

Isso irá:

1. Carregar o PDF (`document.pdf`);
2. Dividir em *chunks* de 1000 caracteres com `overlap` 150;
3. Gerar embeddings;
4. Gravar no Postgres + pgvector na coleção `documento_colecao`.

---

## 4. Chat via CLI

Após a ingestão, você pode fazer perguntas no terminal:

```bash
python src/chat.py
```

O programa iniciará um loop:

- Digite a pergunta e pressione Enter;
- Digite `sair` para encerrar.

---

## 5. API FastAPI

Para subir a API localmente (sem Docker):

```bash
uvicorn src.api:app --reload --port 8000
```

Endpoints principais:

- `GET /health` – verificação simples (retorna `{"status": "ok"}`);
- `POST /api/query` – recebe JSON no formato:

  ```json
  {
    "question": "Qual é a empresa com maior faturamento?",
    "k": 10,
    "collection": "documento_colecao"
  }
  ```

  e retorna:

  ```json
  {
    "answer": "..."
  }
  ```

---

## 6. Frontend minimalista

Com a API em execução (`uvicorn src.api:app --port 8000`), acesse:

```text
http://localhost:8000/
```

A página `static/index.html` será servida automaticamente pelo FastAPI e você poderá:

- Digitar perguntas sobre o conteúdo do PDF;
- Enviar com o botão **Perguntar**;
- Ver a resposta gerada pelo modelo no painel inferior.

---

## 7. Execução via Docker (apenas app)

O `Dockerfile` empacota o app (API + frontend + scripts).

### 7.1 Build da imagem

```bash
docker build -t rag-pgvector-app .
```

### 7.2 Execução da imagem

Certifique-se de que o Postgres do `docker-compose` está rodando (`pgvector-db`).

Execute o container do app:

```bash
docker run --rm \
  --env-file .env \
  -p 8000:8000 \
  --network host \
  rag-pgvector-app
```

> Dependendo do seu ambiente, você pode preferir adicionar o serviço do app ao próprio
> `docker-compose.yml` usando esta mesma imagem, reutilizando a rede default do compose.

Depois, acesse:

```text
http://localhost:8000/
```

---

## 8. Estrutura do projeto

```text
.
├── app.py                 # Lógica principal de RAG (embeddings, vector store, prompt, CLI base)
├── docker-compose.yml     # Serviço Postgres + pgvector
├── Dockerfile             # Container do app (FastAPI + frontend)
├── requirements.txt       # Dependências
├── .env.example           # Template de variáveis de ambiente
├── document.pdf           # PDF a ser ingerido
├── static/
│   └── index.html         # Frontend minimalista
└── src/
    ├── __init__.py
    ├── api.py             # API FastAPI (RAG)
    ├── ingest.py          # Script de ingestão do PDF
    ├── search.py          # Função de busca por similaridade
    └── chat.py            # CLI de chat
```

---

## 9. Observações

- A coleção padrão no pgvector é `documento_colecao`;
- Ajuste `PG_CONN_STR` no `.env` se o host/porta/credenciais do banco mudarem;
- Para trocar de provedor (OpenAI ↔ Gemini), ajuste as variáveis `LLM_PROVIDER` e `EMBEDDING_PROVIDER`.
