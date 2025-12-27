import os
import json
import time
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
data_path = "data/all_notes_structure.jsonl"
chroma_path = "vectorDB/chroma_schema_improved/"

print("Creating NEW vector database:", chroma_path)
print("Source data:", data_path)
start = time.time()

# Remove existing if present
if os.path.exists(chroma_path):
    print("Removing existing database...")
    shutil.rmtree(chroma_path, ignore_errors=True)

# Initialize embeddings
print("Loading BioBERT embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

# Create new database
db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

# Read JSONL file
print("Reading", data_path)
data = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print("Found", len(data), "notes in JSONL file")

# Process and index
docs, metadatas, ids = [], [], []

for item in data:
    if not isinstance(item, dict):
        continue
        
    # Extract metadata
    meta = item.get("metadata", {})
    note_id = meta.get("note_id", "unknown")
    diseases_list = meta.get("diseases", [])
    
    # Ensure diseases is a list of strings
    if isinstance(diseases_list, list):
        diseases_str = ", ".join([str(d) for d in diseases_list])
    else:
        diseases_str = str(diseases_list)

    # Use the entire JSON object as the document content (1 note = 1 chunk)
    content = json.dumps(item, ensure_ascii=False)

    docs.append(content)
    metadatas.append({
        "note_id": note_id,
        "diseases": diseases_str
    })
    ids.append(f"note-{note_id}")

print("Indexing", len(docs), "notes as individual chunks...")
db.add_texts(docs, metadatas=metadatas, ids=ids)

print("\n" + "="*80)
print("SUCCESS! Vector database created at:", chroma_path)
print("Total chunks indexed:", len(docs))
print("Total time:", round(time.time() - start, 2), "seconds")
print("="*80)

# Verify the indexing
print("\nVerifying database...")
results = db.get()
print("Verified:", len(results['documents']), "documents in database")

# Show sample
print("\nSample metadata (first 3 notes):")
for i in range(min(3, len(results['metadatas']))):
    print(f"\n  Note {i+1}:")
    print(f"    ID: {results['ids'][i]}")
    print(f"    Note ID: {results['metadatas'][i]['note_id']}")
    diseases_preview = results['metadatas'][i]['diseases'][:100]
    print(f"    Diseases: {diseases_preview}...")
    print(f"    Content size: {len(results['documents'][i])} chars")

print("\n" + "="*80)
print("CHUNKING VERIFICATION:")
print(f"  Total notes in JSON: {len(data)}")
print(f"  Total chunks in DB: {len(results['documents'])}")
print(f"  1 note = 1 chunk? {'YES' if len(data) == len(results['documents']) else 'NO'}")
print("="*80)
