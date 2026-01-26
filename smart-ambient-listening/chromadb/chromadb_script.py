import chromadb
import json

print("Loading clinical notes...")
with open('patient_notes_structure.json', 'r') as f:
    notes = json.load(f)

print(f"Loaded {len(notes)} notes")

# Use PersistentClient to create database on disk
print("Creating local ChromaDB database...")
client = chromadb.PersistentClient(path="./chroma_data")

# Create collection
print("Creating collection 'medical-schemas'...")
collection = client.get_or_create_collection("medical-schemas")

# Prepare data
documents = []
metadatas = []
ids = []

print("Preparing documents...")
for note in notes:
    note_text = json.dumps(note, indent=2)

    documents.append(note_text)
    metadatas.append({
        'note_id': note['metadata']['note_id'],
        'diseases': ', '.join(note['metadata']['diseases'])
    })
    ids.append(note['metadata']['note_id'])

# Upload to ChromaDB
print(f"Uploading {len(documents)} notes...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print(f"✅ Uploaded {len(documents)} notes!")
print(f"Database created at: ./chroma_data")