from app.services import pdf_rag


def test_generated_answer_labels_evidence_and_history(monkeypatch):
    store=pdf_rag.PdfRagStore()
    c=pdf_rag.PdfChunk('A randomized trial enrolled 120 patients with cancer and reported treatment outcomes.', 'paper.pdf', 2, 0)
    store.chunks=[c]
    monkeypatch.setattr(store,'retrieve',lambda *args,**kwargs:[(c,1.0)])
    seen=[]
    monkeypatch.setattr(pdf_rag,'_answer_with_model',lambda question,context,history: seen.append(history) or 'The trial enrolled 120 people.')
    result=pdf_rag.answer_question(store,'How many?', [{'role':'user','content':'Describe the trial'}])
    assert result.answer.startswith('AI answer:')
    assert '[paper.pdf, page 2]' in result.answer
    assert result.cited_chunks==[c] and seen[0][0]['content']=='Describe the trial'


def test_failed_generation_returns_cited_passages(monkeypatch):
    store=pdf_rag.PdfRagStore()
    c=pdf_rag.PdfChunk('Cancer treatment outcomes were measured in a randomized trial of 120 patients.', 'paper.pdf', 3, 0)
    store.chunks=[c]
    monkeypatch.setattr(store,'retrieve',lambda *args,**kwargs:[(c,1.0)])
    monkeypatch.setattr(pdf_rag,'_answer_with_model',lambda *args: (_ for _ in ()).throw(RuntimeError()))
    result=pdf_rag.answer_question(store,'What was measured?')
    assert result.error=='evidence_only' and result.cited_chunks==[c]


def test_semantic_retrieval_works_without_keyword_overlap(monkeypatch):
    import numpy as np
    class FakeEmbedder:
        def encode(self,texts,**kwargs): return np.ones((len(texts),3))
    store=pdf_rag.PdfRagStore()
    c=pdf_rag.PdfChunk('Oncology therapies improve survival among participants in this randomized study.', 'paper.pdf', 1, 0)
    store.chunks=[c]
    monkeypatch.setattr(pdf_rag,'_get_embedder',lambda: FakeEmbedder())
    from app.services import reranker
    monkeypatch.setattr(reranker,'_load_cross_encoder',lambda name: object())
    monkeypatch.setattr(reranker,'_predict_with_timeout',lambda *args,**kwargs: [1.0])
    assert store.retrieve('cancer medication')[0][0] is c


def test_graph_links_terms_to_document_pages():
    store=pdf_rag.PdfRagStore()
    store.chunks=[pdf_rag.PdfChunk('Cancer therapy improves survival in this randomized trial. The cancer therapy was evaluated in patients.', 'paper.pdf', 3, 0)]
    graph=pdf_rag.knowledge_graph(store)
    assert graph['nodes'] and graph['edges']
    assert all(n['sources'][0]['page_num']==3 for n in graph['nodes'])
    assert all(e['source_ref']['doc_name']=='paper.pdf' for e in graph['edges'])


def test_cold_reader_starts_asynchronously(monkeypatch):
    calls=[]
    monkeypatch.setattr(pdf_rag, '_reader', None)
    monkeypatch.setattr(pdf_rag, 'start_reader_warmup', lambda: calls.append(True))
    import pytest
    with pytest.raises(RuntimeError, match='local_reader_loading'):
        pdf_rag._answer_with_model('Question', 'Context')
    assert calls == [True]
