# genai/rag/retriever.py
import pickle
import faiss

from .embeddings import PolicyEmbedder
from src.config import get, resolve_path

VECTOR_DIR = resolve_path(get("rag", "vector_dir", default="data/processed/policy_vector_store"))
INDEX_PATH = VECTOR_DIR / "policy.index"
CHUNKS_PATH = VECTOR_DIR / "chunks.pkl"


class PolicyRetriever:

    def __init__(self):

        print("Loading FAISS index...")

        self.index = faiss.read_index(str(INDEX_PATH))

        print("FAISS vectors:", self.index.ntotal)

        with open(CHUNKS_PATH, "rb") as file:
            self.chunks = pickle.load(file)

        self.embedder = PolicyEmbedder()

        print("Retriever ready.")


    def search(self, query, top_k=3):

        # Convert query into embedding
        query_embedding = self.embedder.embed_query(query)

        # FAISS expects 2D array
        query_embedding = query_embedding.reshape(1, -1).astype("float32")

        # Search similar policy chunks
        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            chunk = self.chunks[index].copy()
            chunk["similarity_score"] = float(score)
            results.append(chunk)

        return results


if __name__ == "__main__":

    retriever = PolicyRetriever()

    query = "What retention offers are available for an important customer?"

    results = retriever.search(query, top_k=3)

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)

    print("\nQuery:")
    print(query)

    for i, result in enumerate(results, start=1):

        print("\n" + "-" * 60)
        print("Result:", i)
        print("Page:", result["page"])
        print("Similarity:", round(result["similarity_score"], 4))

        print("\nPolicy:")
        print(result["text"][:800])

    print("\n" + "=" * 60)
    print("RETRIEVAL COMPLETED")
    print("=" * 60)