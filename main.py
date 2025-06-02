import os
import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_singlestore import SingleStoreVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import singlestoredb as s2db


def check_database_connection():
    """Check if the SingleStoreDB is running and accessible."""
    try:
        conn = s2db.connect()
        conn.close()
        return True
    except Exception as e:
        print("❌ Could not connect to SingleStoreDB.")
        print("➡️  Make sure it's running (e.g., via Docker) before launching this script.")
        exit(1)


def setup_database():
    """Create the database if it doesn't exist."""
    with s2db.connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute("CREATE DATABASE IF NOT EXISTS testdb")


def load_documents(folder="data"):
    """Load lore documents from folder."""
    documents = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".md") or filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    doc = Document(page_content=content, metadata={"source": file_path})
                    documents.append(doc)
    print(f"✔ Loaded {len(documents)} documents from '{folder}'")
    return documents


def main():
    print("🔍 Checking database connection...")
    check_database_connection()
    setup_database()

    print("📚 Loading lore documents...")
    documents = load_documents("data")
    embedding = OllamaEmbeddings(model="mxbai-embed-large")

    vector_store = SingleStoreVectorStore(
        embedding=embedding,
        host="localhost",  # Or your custom host/port
        database="testdb",
        table_name="campaign_lore",
    )

    print("📌 Adding documents to vector store (only new ones)...")
    try:
        vector_store.add_documents(documents)
    except Exception as e:
        for d in documents:
            print(f"Error on: {d.metadata['source']} ({len(d.page_content)} chars)")
        raise e

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    model = OllamaLLM(model="llama3")

    prompt = ChatPromptTemplate.from_template("""
You are a writer assistant helping with a TTRPG campaign's lore.

Relevant documents: {lore_documents}

Question: {question}
""")
    chain = prompt | model

    print("\n🧠 Lore Q&A System Ready")
    print("Ask questions about the TTRPG world. Type 'exit' to quit.\n")

    while True:
        user_input = input("❓ Your question: ")
        if user_input.lower() == "exit":
            break

        print("🔎 Retrieving relevant documents...")
        lore_documents = retriever.invoke(user_input)
        result = chain.invoke({"lore_documents": lore_documents, "question": user_input})

        print("\n📜 Answer:")
        print(result)


if __name__ == "__main__":
    main()
