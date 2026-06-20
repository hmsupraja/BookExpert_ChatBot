import streamlit as st
import chromadb
from pypdf import PdfReader

st.title("📄 Document Q&A Bot")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./db")

# Create collection if it doesn't exist
collection = client.get_or_create_collection(name="documents")

# If collection is empty, load PDF and store chunks
if collection.count() == 0:
    reader = PdfReader("data/sample.pdf")

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    chunk_size = 500
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    for idx, chunk in enumerate(chunks):
        collection.add(
            ids=[str(idx)],
            documents=[chunk]
        )

question = st.text_input("Ask a Question")

if st.button("Submit"):

    results = collection.query(
        query_texts=[question],
        n_results=1
    )

    answer = results["documents"][0][0]

    st.subheader("Answer")
    st.write(answer)