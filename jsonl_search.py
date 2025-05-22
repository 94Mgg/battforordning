import json
from pathlib import Path
from rapidfuzz import fuzz

# =========================================
# Script: jsonl_search_fixed_path.py
# Beskrivelse:
#   Script til at søge i en fast defineret JSONL-fil.
#   Spørger brugeren kun om søgeord og valgfri fuzzy-threshold.
# Konfiguration:
#   Ændr JSONL_FILE til den korrekte sti til din JSONL-filsti.
# =========================================

# Hardkodet sti til din JSONL-fil
JSONL_FILE = Path(r"C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/Batteriforordningen chatgpt/PDFer/JSONL_data/celex_02023r1542-20240718_da_txt.jsonl")


def search_jsonl_file(jsonl_file: Path, keyword: str, fuzzy_threshold: int = None):
    """
    Søger i den angivne .jsonl-fil efter 'keyword'.
    Hvis fuzzy_threshold er sat, anvendes partial_ratio >= threshold.
    Udskriver filnavn, linje- og sidereference samt snippet for hver match.
    """
    if not jsonl_file.exists():
        print(f"JSONL-filen blev ikke fundet: {jsonl_file}")
        return

    keyword_lower = keyword.lower()
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = data.get("content", "")
            content_lower = content.lower()

            match = False
            if fuzzy_threshold is not None:
                score = fuzz.partial_ratio(keyword_lower, content_lower)
                if score >= fuzzy_threshold:
                    match = True
            else:
                if keyword_lower in content_lower:
                    match = True

            if match:
                page = data.get("page", "?")
                print(f"[Match i {jsonl_file.name} | side {page} | linje {line_no}]\n{content}\n")


def main():
    print("===== JSONL Search (fast filsti) =====")
    if not JSONL_FILE.exists():
        print(f"JSONL-filen findes ikke: {JSONL_FILE}")
        return

    # Spørg om søgeord
    keyword = input("Indtast nøgleord eller frase, der skal søges efter: ").strip()
    if not keyword:
        print("Ingen søgeord indtastet. Afslutter.")
        return

    # Spørg om fuzzy-threshold (valgfri)
    fuzzy_input = input("Vil du bruge fuzzy match? Indtast threshold (0-100) eller tryk Enter for eksakt match: ").strip()
    fuzzy_threshold = None
    if fuzzy_input:
        try:
            fuzzy_threshold = int(fuzzy_input)
            if not (0 <= fuzzy_threshold <= 100):
                raise ValueError
        except ValueError:
            print("Ugyldig threshold. Bruger eksakt match i stedet.")
            fuzzy_threshold = None

    print(f"\nSøger efter '{keyword}' i filen: {JSONL_FILE.name} (fuzzy={fuzzy_threshold})...\n")
    search_jsonl_file(JSONL_FILE, keyword, fuzzy_threshold)
    print("\n=== Færdig ===")

if __name__ == "__main__":
    main()