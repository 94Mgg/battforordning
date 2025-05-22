import os
import json
import re
from pathlib import Path
import tiktoken
from unstructured.partition.pdf import partition_pdf

from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

# ===== SETTINGS =====
# Source and output directories
PDF_FOLDER = Path(r"C:\Users\MickiGrunzig\OneDrive - Zolo International Trading\Dokumenter\Batteriforordningen chatgpt\PDFer")
OUTPUT_FOLDER = Path(r"C:\Users\MickiGrunzig\OneDrive - Zolo International Trading\Dokumenter\Batteriforordningen chatgpt\PDFer\JSONL_data")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Tokenizer settings
encoding = tiktoken.get_encoding("cl100k_base")  # Base encoder
MAX_TOKENS = 150       # Desired chunk size in tokens
OVERLAP_TOKENS = 50    # Number of tokens to overlap between chunks
MIN_WORDS = 4          # Minimum words per chunk to avoid tiny chunks

# Only merge these element types
MERGEABLE_TYPES = {"NarrativeText", "ListItem", "Table"}

# ===== Utility functions =====
def simplify_filename(filename: str) -> str:
    """
    Create a clean .jsonl filename from the PDF stem.
    """
    name = Path(filename).stem.lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name + ".jsonl"


def extract_markdown_from_table(el):
    """
    Convert a table element's HTML to Markdown using pandas.
    Falls back to raw text on error.
    """
    try:
        html = el.metadata.text_as_html
        if not html:
            return el.text or ""
        soup = BeautifulSoup(html, "html.parser")
        table = pd.read_html(StringIO(str(soup)))[0]
        return table.to_markdown(index=False)
    except Exception:
        return el.text or ""


def split_into_token_chunks(text: str,
                            max_tokens: int = MAX_TOKENS,
                            overlap: int = OVERLAP_TOKENS) -> list[str]:
    """
    Split `text` into overlapping chunks of up to `max_tokens` tokens,
    with `overlap` tokens repeated between consecutive chunks.
    Ensures each chunk has at least MIN_WORDS words.

    Sliding window approach:
    - Step size = max_tokens - overlap
    - For sequence length L and window W, windows start at 0, step, 2*step, ...
    """
    tokens = encoding.encode(text)
    chunks = []
    step = max_tokens - overlap
    # Slide window
    for start in range(0, max(len(tokens) - overlap, 0), step):
        chunk_tokens = tokens[start:start + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        # Filter out very small chunks
        if len(chunk_text.split()) >= MIN_WORDS:
            chunks.append(chunk_text)
    return chunks


def parse_pdf_to_chunks(pdf_path: Path) -> list[dict]:
    """
    Read a PDF with partition_pdf, merge desired element types per page,
    detect current article number, then split into overlapping token chunks.
    Returns list of dicts with content, page, source, type, and article metadata.
    """
    elements = partition_pdf(
        filename=str(pdf_path),
        infer_table_structure=True,
        strategy="hi_res"
    )

    # Group text blocks by page
    grouped_by_page: dict[int, list[str]] = {}
    for el in elements:
        if el.category not in MERGEABLE_TYPES:
            continue
        page = el.metadata.page_number
        grouped_by_page.setdefault(page, [])
        if el.category == "Table":
            content = extract_markdown_from_table(el)
        else:
            content = (el.text or "").strip()
        if content:
            grouped_by_page[page].append(content)

    # Build overlapping token chunks with article detection
    all_chunks: list[dict] = []
    article_pattern = re.compile(r"^Artikel\s+(\d+)", re.IGNORECASE)
    current_article = None

    for page, blocks in grouped_by_page.items():
        # Merge blocks into single page text
        full_text = "\n\n".join(blocks)
        # Detect article number in page text
        for line in full_text.splitlines():
            m = article_pattern.match(line)
            if m:
                current_article = m.group(1)
                break  # use first occurrence on this page

        # Split into overlapping chunks
        token_chunks = split_into_token_chunks(full_text)
        # Package each chunk with metadata
        for chunk in token_chunks:
            all_chunks.append({
                "content": chunk,
                "page": page,
                "source": pdf_path.name,
                "article": current_article,
                "type": "MergedText"
            })
    return all_chunks


def main():
    """
    Main entrypoint: find all PDFs, parse them to chunks, and save as JSONL.
    """
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {PDF_FOLDER}\n")

    for pdf_path in pdf_files:
        print(f"Parsing: {pdf_path.name}")
        try:
            chunks = parse_pdf_to_chunks(pdf_path)
            print(f"  Created {len(chunks)} overlapping token chunks")
            if chunks:
                output_filename = simplify_filename(pdf_path.name)
                output_path = OUTPUT_FOLDER / output_filename
                with open(output_path, "w", encoding="utf-8") as f:
                    for chunk in chunks:
                        json.dump(chunk, f, ensure_ascii=False)
                        f.write("\n")
                print(f"  Saved to: {output_path}")
        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {e}")

    print("\n✅ Done processing all PDFs.")

if __name__ == "__main__":
    main()
