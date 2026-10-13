"""Measure top-1 document retrieval and evidence matching on a small fixture."""
import json
from pathlib import Path
from askgraph import build_graph, load_documents

ROOT = Path(__file__).parent
cases = json.loads((ROOT / 'evaluation/questions.json').read_text())
docs = load_documents(ROOT / 'docs')
graph = build_graph()
source_hits = evidence_hits = 0
for case in cases:
    result = graph.invoke({'question': case['question'], 'docs': docs})
    matches = result['matches']
    source_ok = bool(matches) and matches[0]['source'] == case['source']
    evidence_ok = source_ok and case['contains'].lower() in matches[0]['text'].lower()
    source_hits += source_ok
    evidence_hits += evidence_ok
    print(f"{'PASS' if evidence_ok else 'FAIL'} {case['question']}")
print(f'Top-1 source accuracy: {source_hits}/{len(cases)} ({source_hits / len(cases):.1%})')
print(f'Top-1 evidence accuracy: {evidence_hits}/{len(cases)} ({evidence_hits / len(cases):.1%})')
