import os
import streamlit as st
import chromadb
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📄"
)

st.title("📄 Document Q&A Bot")

# -----------------------
# Load Environment Variables
# -----------------------
load_dotenv()

# Streamlit Cloud Secrets support
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found.")
    st.stop()

genai.configure(api_key=api_key)

# -----------------------
# Initialize ChromaDB
# -----------------------
client = chromadb.PersistentClient(path="./db")

collection = client.get_or_create_collection(
    name="documents"
)

# -----------------------
# Load PDF and Index
# -----------------------
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

# -----------------------
# User Input
# -----------------------
question = st.text_input(
    "Ask a Question"
)

if st.button("Submit"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching..."):

            # Retrieve top chunks
            results = collection.query(
                query_texts=[question],
                n_results=3
            )

            context = "\n\n".join(
                results["documents"][0]
            )

            prompt = f"""
You are a helpful document assistant.

Answer ONLY using the provided context.

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
                st.error(f"Gemini Error: {e}")
