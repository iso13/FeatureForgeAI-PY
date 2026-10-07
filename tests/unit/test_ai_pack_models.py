from featureforge.packs.ai.models import (
    CONTRACT_CLAUSES,
    INDEMNITY_CLAUSE,
    BiasSimulatingModel,
    RAGEngine,
    Sample,
    check_parity,
)


def test_parity_check():
    model = BiasSimulatingModel()
    white = Sample("White")
    result = check_parity(white, model.predict(white), baseline=0.5, threshold=0.06)
    assert result.delta == 0.05 and result.passed
    assert not check_parity(white, 0.55, 0.5, 0.02).passed
    assert "Result: FAIL" in check_parity(white, 0.55, 0.5, 0.02).render()


def test_rag_retrieved_ids_reflect_the_query():
    engine = RAGEngine()
    engine.index_documents(CONTRACT_CLAUSES)
    engine.request_summary(["payment"])
    assert engine.retrieved_ids == ["C002"]  # TS returned every indexed id


def test_rag_flags_missing_types():
    engine = RAGEngine()
    engine.index_documents(CONTRACT_CLAUSES)
    summary = engine.request_summary(["payment", "indemnity"])
    assert "incomplete" in summary and "indemnity" in summary
    assert any("missing clause types: indemnity" in line for line in engine.logs)
    engine.add_clause(INDEMNITY_CLAUSE)
    assert INDEMNITY_CLAUSE.content in engine.request_summary(["indemnity"])
