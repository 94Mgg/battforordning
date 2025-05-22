import warnings
warnings.filterwarnings("ignore")
import re
import os
from pathlib import Path
import json
from rapidfuzz import fuzz
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import tiktoken

# === CONFIGURATION ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR-OPENAI-API-KEY-HERE"
CHROMA_FOLDER = Path("chroma_store")
JSONL_FILE    = Path(
    r"C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/"
    "Batteriforordningen chatgpt/PDFer/JSONL_data/"
    "celex_02023r1542-20240718_da_txt.jsonl"
)

# === RAG SETUP ===
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore     = Chroma(persist_directory=str(CHROMA_FOLDER), embedding_function=embedding_model)
llm             = ChatOpenAI(model="gpt-4", temperature=0.05, openai_api_key=OPENAI_API_KEY)

system_prompt = """
Du er en erfaren EU-juridisk konsulent med speciale i Batteriforordningen (EU 2023/1542 mv.).
Altid svar med præcise kildehenvisninger (side, artikel nr og stykke nr) og formuler dig klart og professionelt.
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"fetch_k":100, "k":15, "lambda_mult":0.5}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt},
    input_key="query"
)

# === WINDOWED FUZZY SEARCH ===
def search_jsonl_context(jsonl_file: Path, keyword: str, fuzzy_threshold: int = None, window: int = 5):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    key = keyword.lower()
    match_scores = []
    for i, e in enumerate(entries):
        txt = e.get("content", "").lower()
        score = fuzz.partial_ratio(key, txt)
        if fuzzy_threshold is not None and score >= fuzzy_threshold:
            match_scores.append((i, score))
        elif fuzzy_threshold is None and key in txt:
            match_scores.append((i, score))
    match_scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = match_scores[:5]
    indices = set()
    for i, _ in top_matches:
        start = max(0, i - window)
        end   = min(len(entries), i + window + 1)
        indices.update(range(start, end))
    return [entries[i] for i in sorted(indices)]

# === DYNAMIC KEYWORDS EXTRACTION ===
def extract_keywords(query, llm):
    prompt = f"""
    Udtræk op til 3 relevante søgeord eller fraser fra følgende spørgsmål hvis nødvendigt og relevnt, som er optimale til at finde information i dokumenterne. Undgå generiske ord som
    "bruger", "og", "hvis man" ogs.:

    Spørgsmål: "{query}"

    Returner kun søgeordene eller fraserne, adskilt med semikolon.
    """

    response = llm.invoke([
        SystemMessage(content="Du er ekspert i søgeordsanalyse og EU Batteriforordningen."),
        HumanMessage(content=prompt)
    ])

    keywords = [kw.strip() for kw in response.content.split(";")]
    return keywords

def get_full_article(jsonl_file: Path, article_number: str):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    article_chunks = []
    capture = False
    article_start = f"artikel {article_number}".lower()

    for entry in entries:
        content_lower = entry["content"].lower()
        if article_start in content_lower:
            capture = True
        elif capture and "artikel" in content_lower and article_start not in content_lower:
            break  # Stop når næste artikel starter
        if capture:
            article_chunks.append(entry["content"])

    return "\n\n".join(article_chunks).strip()

# === TOKEN LIMITER FUNCTION ===
def truncate_context(texts, max_tokens=6000, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    token_count = 0
    truncated = []
    for text in texts:
        tokens = enc.encode(text)
        if token_count + len(tokens) > max_tokens:
            break
        truncated.append(text)
        token_count += len(tokens)
    return truncated

# === CHAT FUNCTION ===
def chat():
    print("Bot is ready! Type 'exit' to quit.")
    article_pattern = re.compile(r'artikel (\d+)', re.IGNORECASE)

    while True:
        query = input("> ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        match = article_pattern.search(query)
        if match:
            article_num = match.group(1)
            print(f"[DEBUG] Henter hele artikel {article_num}.")
            article_text = get_full_article(JSONL_FILE, article_num)

            if article_text:
                final_prompt = f"""
                Du er en erfaren EU-juridisk konsulent med speciale i Batteriforordningen (EU 2023/1542 mv.).
                Nedenfor er den komplette tekst for artikel {article_num}.

                Artikel {article_num}:
                {article_text}

                Besvar følgende spørgsmål klart og professionelt:
                "{query}"
                """

                final_ans = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=final_prompt)
                ])

                answer = final_ans.content
                print("\n[Answer]\n", answer, "\n")
            else:
                print(f"\n[Answer]\nArtikel {article_num} blev ikke fundet.\n")
        else:
            # Standard RAG + fuzzy ved andre spørgsmål
            keywords = extract_keywords(query, llm)
            print(f"[DEBUG] Udtrukne nøgleord til fuzzy search: {keywords}")

            rag_result = qa_chain.invoke({"query": query})
            docs = rag_result.get("source_documents", [])
            print(f"[DEBUG] RAG hentede {len(docs)} chunks.")

            fuzzy_chunks = []
            for kw in keywords:
                fuzzy_chunks += search_jsonl_context(JSONL_FILE, kw, fuzzy_threshold=65, window=3)

            fuzzy_chunks_unique = {c["content"]: c for c in fuzzy_chunks}.values()
            print(f"[DEBUG] Fuzzy fandt {len(fuzzy_chunks_unique)} unikke chunks.")

            combined_texts = [d.page_content for d in docs] + [c["content"] for c in fuzzy_chunks_unique if c["content"] not in [d.page_content for d in docs]]
            combined_texts = truncate_context(combined_texts, max_tokens=6000, model="gpt-4")
            context_str = "\n\n".join(combined_texts)

            final_prompt = qa_prompt.format(context=context_str, question=query)

            final_ans = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=final_prompt)
            ])

            answer = final_ans.content
            print("\n[Answer]\n", answer, "\n")

if __name__ == "__main__":
    chat()
