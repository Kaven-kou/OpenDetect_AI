import time
from concurrent.futures import ThreadPoolExecutor

from opendetect_ai.tools import rag_tool


def test_vectorstore_is_initialized_once_across_threads(monkeypatch, tmp_path) -> None:
    created = []

    class FakeChroma:
        def __init__(self, **kwargs) -> None:
            time.sleep(0.01)
            created.append(kwargs)

    monkeypatch.setattr(rag_tool, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_tool, "Chroma", FakeChroma)
    monkeypatch.setattr(rag_tool, "_get_embeddings", lambda: object())
    monkeypatch.setattr(rag_tool, "_vectorstore", None)

    with ThreadPoolExecutor(max_workers=8) as executor:
        stores = list(executor.map(lambda _: rag_tool._get_vectorstore(), range(32)))

    assert len(created) == 1
    assert all(store is stores[0] for store in stores)
