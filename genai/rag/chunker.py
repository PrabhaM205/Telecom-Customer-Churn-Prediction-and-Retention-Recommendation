def chunk_documents(pages, chunk_size=1000, overlap=150):
    """
    Split policy pages into smaller overlapping chunks.
    """

    chunks = []

    chunk_id = 0

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": chunk_text
                })

                chunk_id += 1

            start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    from document_loader import load_policy_document

    pdf_path = "data/raw/company_policy.pdf"

    pages = load_policy_document(pdf_path)

    chunks = chunk_documents(pages)

    print("\n" + "=" * 60)
    print("CHUNKING TEST")
    print("=" * 60)

    print("Total pages:", len(pages))
    print("Total chunks:", len(chunks))

    for chunk in chunks[:3]:

        print("\n" + "-" * 60)
        print("Chunk ID:", chunk["chunk_id"])
        print("Page:", chunk["page"])
        print("-" * 60)

        print(chunk["text"][:500])

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETED")
    print("=" * 60)