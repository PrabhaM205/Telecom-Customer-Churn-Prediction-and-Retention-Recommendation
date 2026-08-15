from pathlib import Path
from pypdf import PdfReader

print("DOCUMENT LOADER STARTED")


def load_policy_document(pdf_path):

    pdf_path = Path(pdf_path)

    print("Checking file:", pdf_path)

    if not pdf_path.exists():
        print("ERROR: PDF FILE NOT FOUND")
        return []

    print("PDF FOUND")

    reader = PdfReader(str(pdf_path))

    print("PDF opened successfully")
    print("Total PDF pages:", len(reader.pages))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        print(f"Reading page {page_number}...")

        text = page.extract_text()

        if text and text.strip():

            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


if __name__ == "__main__":

    print("MAIN FUNCTION STARTED")

    policy_path = "data/raw/company_policy.pdf"

    pages = load_policy_document(policy_path)

    print("\n" + "=" * 60)
    print("POLICY EXTRACTION RESULT")
    print("=" * 60)

    print("Extracted pages:", len(pages))

    for page in pages[:2]:

        print(f"\n--- PAGE {page['page']} ---")
        print(page["text"][:1000])

    print("\nDONE")