import argparse
import os
import shutil
import pandas as pd
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data/"  # Folder containing PDFs and CSVs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database before adding new data.")
    args = parser.parse_args()

    if args.reset:
        clear_database()

    documents = load_documents()
    if not documents:
        print("⚠️ No documents found! Ensure the 'data/' folder has files.")
        return

    print(f"📄 Loaded {len(documents)} documents. Processing...")

    chunks = split_documents(documents)
    store_in_chroma(chunks)
    print("✅ Data successfully stored in ChromaDB.")


def load_documents():
    """Load PDFs and CSVs, converting them into AI-readable format."""
    documents = []

    # Load PDFs
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    try:
        pdf_documents = pdf_loader.load()
        if pdf_documents:
            print(f"📄 Loaded {len(pdf_documents)} PDFs.")
            for doc in pdf_documents:
                doc.metadata["source"] = doc.metadata.get("source", "unknown_pdf")
                doc.metadata["type"] = "pdf"
            documents.extend(pdf_documents)
        else:
            print("⚠️ No PDFs found in 'data/' folder.")
    except Exception as e:
        print(f"❌ Error loading PDFs: {e}")

    # Load CSVs
    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
    if not csv_files:
        print("⚠️ No CSV files found in 'data/' folder.")

    for file in csv_files:
        csv_path = os.path.join(DATA_PATH, file)
        csv_documents = load_csv(csv_path)
        if csv_documents:
            print(f"📊 Processed {len(csv_documents)} rows from {file}.")
            documents.extend(csv_documents)

    return documents


def load_csv(csv_path):
    """Reads a CSV and converts each row into a Document object."""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", dtype=str).fillna("")  # Ensure all values are strings
    except Exception as e:
        print(f"❌ Error reading {csv_path}: {e}")
        return []

    documents = [
        Document(
            page_content="\n".join([f"{col}: {row[col]}" for col in df.columns]),
            metadata={"source": csv_path, "row_index": index, "type": "csv"}
        )
        for index, row in df.iterrows()
    ]

    return documents

def split_documents(documents):
    """Splits long documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False
    )
    return text_splitter.split_documents(documents)


def store_in_chroma(chunks):
    """Stores document chunks in ChromaDB with unique IDs."""
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Assign unique chunk IDs
    chunks_with_ids = assign_chunk_ids(chunks)

    # Retrieve existing document IDs
    existing_items = db.get(include=[])
    existing_ids = set(existing_items.get("ids", []))
    print(f"📦 Existing documents in DB: {len(existing_ids)}")

    # Filter only new documents
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"🆕 Adding {len(new_chunks)} new documents.")
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
    else:
        print("✅ No new documents to add.")


def assign_chunk_ids(chunks):
    """Assigns unique IDs to document chunks based on source and index."""
    last_source_id = None
    chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page_or_row = chunk.metadata.get("page", chunk.metadata.get("row_index", "0"))
        current_source_id = f"{source}:{page_or_row}"

        if current_source_id == last_source_id:
            chunk_index += 1
        else:
            chunk_index = 0

        chunk.metadata["id"] = f"{current_source_id}:{chunk_index}"
        last_source_id = current_source_id

    return chunks


def clear_database():
    """Resets the ChromaDB database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🗑️ Database cleared.")


if __name__ == "__main__":
    main()
