import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key is loaded
api_key = os.getenv("GEMINI_API_KEY")

print("API KEY LOADED:", "YES" if api_key else "NO")

if not api_key:
    print("\nERROR: Gemini API key not found!")
    print("Check your .env file.")
    exit()

# Configure Gemini
genai.configure(api_key=api_key)

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./db")

collection = client.get_collection(
    name="documents"
)

# Get question from user
question = input("\nAsk a Question: ")

# Search relevant chunks
results = collection.query(
    query_texts=[question],
    n_results=3
)

# Build context
context = "\n\n".join(results["documents"][0])

# Create prompt
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
    # Gemini model
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(response.text)

except Exception as e:
    print("\nGemini Error:")
    print(e)
    
