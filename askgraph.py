"""A small, local document question-answering workflow built with LangGraph."""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

STOP = set('a an and are as at be by can did do does for from how in is it its of on or the to was what when where which who why with'.split())


def terms(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP]


def load_documents(folder: Path) -> list[dict[str, str]]:
    documents = []
    for path in sorted(folder.glob('*.md')):
        # Paragraphs are retrievable units; the filename becomes the citation.
        for index, paragraph in enumerate(re.split(r'\n\s*\n', path.read_text(encoding='utf-8').strip())):
            paragraph = paragraph.strip()
            if paragraph and not paragraph.startswith('#'):
                documents.append({'source': path.name, 'chunk': str(index), 'text': paragraph})
    return documents


def retrieve(question: str, docs: list[dict[str, str]], limit: int = 3) -> list[dict]:
    """Rank chunks by TF-IDF cosine similarity; omit chunks with no overlap."""
    counts = [Counter(terms(doc['text'])) for doc in docs]
    query = Counter(terms(question))
    if not query or not docs:
        return []
    vocabulary = set(query).union(*(set(count) for count in counts))
    idf = {word: math.log((1 + len(docs)) / (1 + sum(word in c for c in counts))) + 1 for word in vocabulary}
    qvec = {word: n * idf[word] for word, n in query.items()}
    qnorm = math.sqrt(sum(v * v for v in qvec.values()))
    results = []
    for doc, count in zip(docs, counts):
        weights = {word: n * idf[word] for word, n in count.items()}
        norm = math.sqrt(sum(v * v for v in weights.values()))
        score = sum(value * weights.get(word, 0) for word, value in qvec.items()) / (qnorm * norm) if norm else 0
        if score > 0:
            results.append({**doc, 'score': score})
    return sorted(results, key=lambda item: item['score'], reverse=True)[:limit]


class QAState(TypedDict, total=False):
    question: str
    docs: list[dict]
    matches: list[dict]
    answer: str


def search(state: QAState) -> dict:
    return {'matches': retrieve(state['question'], state['docs'])}


def route(state: QAState) -> str:
    return 'answer' if state['matches'] else 'no_match'


def answer(state: QAState) -> dict:
    # Extractive answers keep every claim traceable to the user's files.
    top = state['matches'][0]
    return {'answer': f"{top['text']}\n\nSource: {top['source']} (chunk {top['chunk']})"}


def no_match(state: QAState) -> dict:
    return {'answer': 'No matching evidence found in the supplied documents.'}


def build_graph():
    graph = StateGraph(QAState)
    graph.add_node('search', search)
    graph.add_node('answer', answer)
    graph.add_node('no_match', no_match)
    graph.add_edge(START, 'search')
    graph.add_conditional_edges('search', route, {'answer': 'answer', 'no_match': 'no_match'})
    graph.add_edge('answer', END)
    graph.add_edge('no_match', END)
    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description='Ask a question about local Markdown documents.')
    parser.add_argument('question', help='Question to search for')
    parser.add_argument('--docs', type=Path, default=Path(__file__).parent / 'docs')
    args = parser.parse_args()
    if not args.docs.is_dir():
        parser.error(f'document folder does not exist: {args.docs}')
    documents = load_documents(args.docs)
    if not documents:
        parser.error(f'no Markdown paragraphs found in: {args.docs}')
    print(build_graph().invoke({'question': args.question, 'docs': documents})['answer'])


if __name__ == '__main__':
    main()
