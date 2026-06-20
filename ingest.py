from pypdf import PdfReader
import chromadb

# Read PDF
reader = PdfReader("data/sample.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("PDF Loaded Successfully!")
print(f"Total Characters: {len(text)}")

# Split into chunks
chunk_size = 500
chunks = []

for i in range(0, len(text), chunk_size):
    chunks.append(text[i:i + chunk_size])

print(f"Created {len(chunks)} chunks")

# Create ChromaDB
client = chromadb.PersistentClient(path="./db")

try:
    client.delete_collection("documents")
except:
    pass

collection = client.get_or_create_collection(
    name="documents"
)
collection = client.get_or_create_collection(
    name="documents"
)

# Store chunks
for idx, chunk in enumerate(chunks):
    collection.add(
        ids=[str(idx)],
        documents=[chunk]
    )

print("Data stored successfully in ChromaDB!")