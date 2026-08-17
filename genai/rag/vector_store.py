# genai/rag/vector_store.py

import os
import pickle
import faiss

from .document_loader import load_policy_document
from .chunker import chunk_documents
from .embeddings import PolicyEmbedder

from src.config import get, resolve_path


# --------------------------------------------------
# Paths -- resolved against PROJECT ROOT
# --------------------------------------------------

PDF_PATH = str(
    resolve_path(
        get(
            "rag",
            "pdf_path",
            default="data/raw/company_policy.pdf"
        )
    )
)

VECTOR_DIR = str(
    resolve_path(
        get(
            "rag",
            "vector_dir",
            default="data/processed/policy_vector_store"
        )
    )
)


INDEX_PATH = os.path.join(
    VECTOR_DIR,
    "policy.index"
)

CHUNKS_PATH = os.path.join(
    VECTOR_DIR,
    "chunks.pkl"
)


# --------------------------------------------------
# Create Vector Store
# --------------------------------------------------

def create_vector_store():

    print("\n" + "=" * 60)
    print("CREATING POLICY VECTOR STORE")
    print("=" * 60)


    # --------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------

    print("\n[1] Loading policy PDF...")

    pages = load_policy_document(
        PDF_PATH
    )

    print(
        "Pages loaded:",
        len(pages)
    )


    # --------------------------------------------------
    # 2. Create chunks
    # --------------------------------------------------

    print("\n[2] Creating chunks...")

    chunks = chunk_documents(
        pages
    )

    print(
        "Chunks created:",
        len(chunks)
    )


    # --------------------------------------------------
    # 3. Load embedding model
    # --------------------------------------------------

    print(
        "\n[3] Loading embedding model..."
    )

    embedder = PolicyEmbedder()


    # --------------------------------------------------
    # 4. Extract chunk text
    # --------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    # --------------------------------------------------
    # 5. Generate embeddings
    # --------------------------------------------------

    print(
        "\n[4] Creating embeddings..."
    )

    embeddings = (
        embedder.embed_documents(
            texts
        )
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )


    # --------------------------------------------------
    # 6. Create FAISS index
    # --------------------------------------------------

    print(
        "\n[5] Creating FAISS index..."
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings.astype(
            "float32"
        )
    )

    print(
        "Vectors stored:",
        index.ntotal
    )


    # --------------------------------------------------
    # 7. Create directory
    # --------------------------------------------------

    os.makedirs(
        VECTOR_DIR,
        exist_ok=True
    )


    # --------------------------------------------------
    # 8. Save FAISS index
    # --------------------------------------------------

    faiss.write_index(
        index,
        INDEX_PATH
    )

    print(
        "\nFAISS index saved:"
    )

    print(
        INDEX_PATH
    )


    # --------------------------------------------------
    # 9. Save chunks
    # --------------------------------------------------

    with open(
        CHUNKS_PATH,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )


    print(
        "\nChunks saved:"
    )

    print(
        CHUNKS_PATH
    )


    # --------------------------------------------------
    # SUCCESS
    # --------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "VECTOR STORE CREATED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    create_vector_store()