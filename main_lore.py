import os
import singlestoredb as s2db
import requests
from datetime import datetime
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_singlestore import SingleStoreVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def log_step(label):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] {label}")

def ensure_database_exists(host, user, password, db_name):
    """Create the DB if it doesn't exist yet."""
    try:
        with s2db.connect(host=host, user=user, password=password) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        return True
    except Exception as e:
        print("❌ Could not create database:", e)
        return False


def check_database_connection(host, user, password, db_name):
    try:
        with s2db.connect(host=host, user=user, password=password, database=db_name) as conn:
            return True
    except Exception:
        return False

def check_ollama_model_ready(model_name):
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        if model_name not in model_names and f"{model_name}:latest" not in model_names:
            print(f"❌ Model '{model_name}' is not downloaded.")
            print(f"➡️  Run this to download it:\n    ollama pull {model_name}")
            return False
        return True
    except Exception as e:
        print(f"❌ Cannot reach Ollama at http://localhost:11434 : {e}")
        print("➡️  Make sure Ollama is running (use `ollama run llama3` or `ollama serve`)")
        return False


def load_documents(folder="data"):
    documents = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".md") or filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path}
                    )
                    documents.append(doc)
    print(f"Loaded {len(documents)} documents from '{folder}'")
    return documents


def main():
    host = "localhost"
    user = "root"
    password = "secret"
    database = "campagne_ombre"
    table_name = "lore_documents"
    model_name = "llama3:latest"

    print("Ensuring database exists...")
    if not ensure_database_exists(host, user, password, database):
        return

    print("Checking connection to SingleStoreDB...")
    if not check_database_connection(host, user, password, database):
        print("❌ Cannot connect to SingleStoreDB at", host)
        print("➡️  Make sure the server is running with: docker compose up -d")
        return

    print("Connected to SingleStoreDB.")

    print("Loading and embedding new lore documents...")
    documents = load_documents("data")
    embedding = OllamaEmbeddings(model="mxbai-embed-large")

    vector_store = SingleStoreVectorStore(
        embedding=embedding,
        host=host,
        user=user,
        password=password,
        database=database,
        table_name=table_name
    )

    existing_sources = set(
        doc.metadata["source"]
        for doc in vector_store.similarity_search("ignore", k=1000)
    )
    new_docs = [doc for doc in documents if doc.metadata["source"] not in existing_sources]

    if new_docs:
        print(f"Adding {len(new_docs)} new document(s)...")
        vector_store.add_documents(new_docs)
    else:
        print("No new documents to embed.")

    print("Checking Ollama model availability...")
    if not check_ollama_model_ready(model_name):
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    model = OllamaLLM(model="llama3")

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for a tabletop RPG campaign.
Use the following lore documents to answer the player's question.

Lore:
{lore_documents}

Question:
{question}
""")

    chain = prompt | model

    print("\n--- Lore Q&A System ---")
    print("Ask questions about the campaign lore.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Your question: ")
        if user_input.lower() == "exit":
            break
        lore_documents = retriever.invoke(user_input)
        result = chain.invoke({
            "lore_documents": lore_documents,
            "question": user_input
        })
        print("\n--- Answer ---")
        print(result)
        print()


if __name__ == "__main__":
    main()
