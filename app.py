import streamlit as st
import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API Key not found in .env file")
    st.stop()

genai.configure(api_key=api_key)

st.title("📄 Document Q&A Bot")

# Connect ChromaDB
client = chromadb.PersistentClient(path="./db")

collection = client.get_collection(
    name="documents"
)

question = st.text_input("Ask a Question")

if st.button("Submit") and question:

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
You are a helpful document assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, reply:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = model.generate_content(
            prompt
        )

        st.subheader("Answer")
        st.write(response.text)

    except Exception as e:
        st.error(str(e))
