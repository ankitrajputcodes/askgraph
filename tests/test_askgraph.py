from askgraph import build_graph, retrieve


def test_retrieves_and_cites_matching_passage():
    docs = [{'source': 'notes.md', 'chunk': '1', 'text': 'Binary search takes logarithmic time.'}]
    result = build_graph().invoke({'question': 'binary search', 'docs': docs})
    assert result['matches'][0]['source'] == 'notes.md'
    assert 'Source: notes.md' in result['answer']


def test_no_match_does_not_invent_answer():
    docs = [{'source': 'notes.md', 'chunk': '1', 'text': 'Binary search takes logarithmic time.'}]
    result = build_graph().invoke({'question': 'photosynthesis', 'docs': docs})
    assert result['matches'] == []
    assert 'No matching evidence' in result['answer']


def test_empty_documents_and_stopwords():
    assert retrieve('binary search', []) == []
    assert retrieve('what is the', [{'source': 'x', 'chunk': '1', 'text': 'binary search'}]) == []
