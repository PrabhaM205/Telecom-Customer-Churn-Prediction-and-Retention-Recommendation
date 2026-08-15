from sentence_transformers import SentenceTransformer


class PolicyEmbedder:

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded.")

    def embed_documents(self, texts):
        """
        Convert policy chunks into numerical vectors.
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings

    def embed_query(self, query):
        """
        Convert a user query into a vector.
        """

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        return embedding[0]


if __name__ == "__main__":

    embedder = PolicyEmbedder()

    sample_texts = [
        "Customer retention discount policy",
        "Device upgrade and fee waiver policy"
    ]

    embeddings = embedder.embed_documents(
        sample_texts
    )

    print("\n" + "=" * 60)
    print("EMBEDDING TEST")
    print("=" * 60)

    print("Number of texts:", len(sample_texts))
    print("Embedding shape:", embeddings.shape)

    print("\nFirst vector:")
    print(embeddings[0][:10])

    print("\n" + "=" * 60)
    print("EMBEDDING COMPLETED")
    print("=" * 60)