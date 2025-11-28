from src.search import search_prompt

def main():
    print("Chat RAG iniciado. Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Pergunta: ").strip()
        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chat.")
            break

        resposta = search_prompt(pergunta)
        print("\nResposta:\n", resposta, "\n")

if __name__ == "__main__":
    main()
