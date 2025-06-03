import os
import time
import requests
from datetime import datetime
import singlestoredb as s2db
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_singlestore import SingleStoreVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from embed_cache import load_cache, save_cache, file_needs_embedding, update_cache


def log_step(label):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] {label}")


def ensure_database_exists(host, user, password, db_name):
    try:
        with s2db.connect(host=host, user=user, password=password) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        log_step(f"✅ Database '{db_name}' is ready.")
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
        log_step(f"✅ Model '{model_name}' is available.")
        return True
    except Exception as e:
        print(f"❌ Cannot reach Ollama at http://localhost:11434 : {e}")
        print("➡️  Make sure Ollama is running (use `ollama run llama3` or `ollama serve`)")
        return False


def load_documents(folder="data"):
    documents = []
    cache = load_cache()

    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".md") or filename.endswith(".txt"):
                file_path = os.path.join(root, filename)

                if file_needs_embedding(file_path, cache):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        doc = Document(
                            page_content=content,
                            metadata={"source": file_path}
                        )
                        documents.append(doc)
                        update_cache(file_path, cache)

    save_cache(cache)

    log_step(f"📂 Loaded {len(documents)} documents from '{folder}'")
    return documents


def main():
    host = "localhost"
    user = "root"
    password = "secret"
    database = "campagne_ombre"
    table_name = "lore_documents"
    model_name = "phi3:mini" #tested with llama3

    log_step("🔌 Ensuring database exists...")
    if not ensure_database_exists(host, user, password, database):
        return

    log_step("🔁 Checking database connection...")
    if not check_database_connection(host, user, password, database):
        print("❌ Cannot connect to SingleStoreDB at", host)
        print("➡️  Make sure the server is running with: docker compose up -d")
        return
    log_step("✅ Connected to SingleStoreDB.")

    log_step("📡 Checking Ollama model availability...")
    if not check_ollama_model_ready(model_name):
        return

    log_step("📂 Loading documents from 'data/'...")
    documents = load_documents("data")

    log_step("⚙️ Initializing embedding model...")
    embedding = OllamaEmbeddings(model="mxbai-embed-large")

    log_step("🧱 Initializing vector store...")
    vector_store = SingleStoreVectorStore(
        embedding=embedding,
        host=host,
        user=user,
        password=password,
        database=database,
        table_name=table_name
    )

    log_step("🔍 Checking existing documents in vector store...")
    existing_sources = set(
        doc.metadata["source"]
        for doc in vector_store.similarity_search("ignore", k=1000)
    )
    new_docs = [doc for doc in documents if doc.metadata["source"] not in existing_sources]
    log_step(f"📊 Found {len(new_docs)} new documents to embed.")

    if new_docs:
        start_embed = time.time()
        vector_store.add_documents(new_docs)
        log_step(f"✅ Added {len(new_docs)} documents in {round(time.time() - start_embed, 2)}s.")
    else:
        log_step("📭 No new documents to embed.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    model = OllamaLLM(model=model_name)

    prompt = ChatPromptTemplate.from_template("""
You are a concise, helpful assistant for a tabletop RPG campaign. 
Answer the player's question based only on the information provided below. 
Do not invent questions, do not continue the conversation beyond the answer.

Lore documents:
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

        log_step("🔎 Retrieving documents...")
        t0 = time.time()
        lore_documents = retriever.invoke(user_input)
        log_step(f"📚 Retrieved {len(lore_documents)} docs in {round(time.time() - t0, 2)}s.")

        log_step("✍️ Generating answer with LLM (streaming)...")
        t1 = time.time()

        print("\n--- Answer ---")
        for chunk in chain.stream({
            "lore_documents": lore_documents,
            "question": user_input
        }, config=RunnableConfig()):
            print(chunk, end="", flush=True)

        print()
        log_step(f"\n✅ Answer streamed in {round(time.time() - t1, 2)}s.")



if __name__ == "__main__":
    main()
