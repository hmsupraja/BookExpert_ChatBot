import streamlit as st
import chromadb

st.title("📄 Document Q&A Bot")

question = st.text_input("Ask a Question")

if st.button("Submit"):

    client = chromadb.PersistentClient(path="./db")

    collection = client.get_collection(name="documents")

    results = collection.query(
        query_texts=[question],
        n_results=1
    )

    answer = results["documents"][0][0]

    st.subheader("Answer")
    st.write(answer)