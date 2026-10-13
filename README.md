# AskGraph

AskGraph is a small, local document question-answering project using LangGraph. It retrieves a relevant paragraph from Markdown files and returns it with a source citation. It runs without an API key, vector database, or language model: answers are **extractive**, not generated.

## How it works

1. Load Markdown documents and split them into paragraphs.
2. Rank passages with TF-IDF cosine similarity.
3. A LangGraph workflow routes to either a cited answer or a no-match response.

Retrieval is lexical: synonyms and ambiguous questions can fail, and a shared word does not guarantee that a passage answers the question. The no-match branch only detects zero token overlap, not every unsupported answer. It is a learning project, not a production RAG system.

## Run

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python askgraph.py "What is the time complexity of binary search?"
```

To use your own material, put `.md` files in a directory and pass `--docs path/to/directory`. Each non-heading paragraph becomes a separate retrievable passage. The source file and paragraph index appear in the answer.

## Evaluate

```bash
python evaluate.py
pip install pytest
python -m pytest -q
```

The bundled sample has 12 questions over 12 passages in four short study-note documents. The local run retrieved the expected source and evidence in the top result for **12/12 questions**. This is a small, hand-written test set aligned with the sample documents; it is not a general accuracy benchmark. The script checks both the expected filename and a specific phrase in the first retrieved passage. The unit tests also exercise citations and the no-match path.
