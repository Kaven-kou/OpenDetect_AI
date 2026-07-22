import time
from concurrent.futures import ThreadPoolExecutor

import fitz
import pytest

from opendetect_ai.tools import rag_tool


def _write_layout_pdf(path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 50), "Table 1: Model accuracy by split", fontsize=10)
    x_positions = [72, 180, 288]
    y_positions = [70, 95, 120, 145]
    for x_pos in x_positions:
        page.draw_line((x_pos, y_positions[0]), (x_pos, y_positions[-1]))
    for y_pos in y_positions:
        page.draw_line((x_positions[0], y_pos), (x_positions[-1], y_pos))
    for y_pos, row in zip(
        [87, 112, 137],
        [("Split", "Accuracy"), ("Train", "95%"), ("Test", "91%")],
    ):
        for x_pos, value in zip([82, 190], row):
            page.insert_text((x_pos, y_pos), value, fontsize=9)

    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 160, 80), False)
    pixmap.clear_with(180)
    page.insert_image(fitz.Rect(72, 190, 300, 310), pixmap=pixmap)
    page.insert_text(
        (72, 330),
        "Figure 2: Architecture overview of the proposed model.",
        fontsize=10,
    )

    second = doc.new_page(width=612, height=792)
    second.insert_text(
        (72, 72),
        "As shown in Figure 2, the encoder feeds the classifier.",
        fontsize=10,
    )
    second.insert_text(
        (72, 96),
        "Table 1 reports the held-out accuracy.",
        fontsize=10,
    )
    doc.save(path)
    doc.close()


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


def test_build_pdf_documents_preserves_page_numbers(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    doc = fitz.open()
    first = doc.new_page()
    first.insert_text((72, 72), "First page discusses vision transformers.")
    second = doc.new_page()
    second.insert_text((72, 72), "Second page reports experimental results.")
    doc.save(pdf_path)
    doc.close()

    documents, ids, pages = rag_tool._build_pdf_documents(
        str(pdf_path),
        title="Paper",
        arxiv_id="1234.5678",
        authors="A. Author",
        published="2026-01-01",
        source="local",
    )

    assert pages == 2
    assert {item.metadata["page"] for item in documents} == {1, 2}
    assert len(ids) == len(set(ids)) == len(documents)


def test_build_pdf_documents_structures_tables_and_links_figures(tmp_path) -> None:
    pdf_path = tmp_path / "layout-paper.pdf"
    _write_layout_pdf(pdf_path)

    documents, _, pages = rag_tool._build_pdf_documents(
        str(pdf_path),
        title="Layout Paper",
        arxiv_id="1234.5678",
        authors="A. Author",
        published="2026-01-01",
        source="local",
    )

    table = next(doc for doc in documents if doc.metadata["element_type"] == "table")
    assert pages == 2
    assert table.metadata["page"] == 1
    assert table.metadata["element_number"] == "1"
    assert table.metadata["element_title"] == "Model accuracy by split"
    assert table.metadata["reference_pages"] == "2"
    assert table.metadata["parser_version"] == rag_tool._PARSER_VERSION
    assert "| Split | Accuracy |" in table.page_content
    assert "| Test | 91% |" in table.page_content
    assert "第 2 页：Table 1 reports the held-out accuracy." in table.page_content

    figure = next(doc for doc in documents if doc.metadata["element_type"] == "figure")
    assert figure.metadata["page"] == 1
    assert figure.metadata["element_number"] == "2"
    assert figure.metadata["element_title"] == "Architecture overview of the proposed model."
    assert figure.metadata["image_detected"] is True
    assert figure.metadata["image_xref"] > 0
    assert figure.metadata["element_bbox"]
    assert figure.metadata["reference_pages"] == "2"
    assert "第 2 页：As shown in Figure 2" in figure.page_content


def test_table_to_markdown_escapes_pipes_and_pads_rows() -> None:
    markdown = rag_tool._table_to_markdown([
        ["Method", "Score"],
        ["A | B", "0.9"],
        ["C"],
    ])

    assert "| A \\| B | 0.9 |" in markdown
    assert "| C |  |" in markdown


def test_caption_parser_supports_roman_table_numbers() -> None:
    parsed = rag_tool._parse_caption("Table IV. Ablation study")
    assert parsed == {
        "kind": "table",
        "number": "IV",
        "label": "Table IV",
        "title": "Ablation study",
        "caption": "Table IV. Ablation study",
    }
    assert rag_tool._parse_caption("Table models are useful") is None


def test_parser_version_controls_reingestion() -> None:
    class FakeVectorstore:
        def __init__(self, parser_version):
            self.parser_version = parser_version

        def get(self, **kwargs):
            return {
                "ids": ["paper__chunk_0"],
                "metadatas": [{"parser_version": self.parser_version}],
            }

    assert rag_tool._paper_already_ingested(FakeVectorstore(1), "id", "Paper") is False
    assert (
        rag_tool._paper_already_ingested(
            FakeVectorstore(rag_tool._PARSER_VERSION), "id", "Paper"
        )
        is True
    )


def test_document_write_uses_public_chroma_collection(monkeypatch) -> None:
    calls = []

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[float(index)] for index, _text in enumerate(texts)]

    class FakeCollection:
        def upsert(self, **kwargs):
            calls.append(kwargs)

    documents = [rag_tool.Document(page_content="content")]
    monkeypatch.setattr(rag_tool, "_get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(rag_tool, "_get_write_collection", lambda: FakeCollection())
    rag_tool._add_documents_fast(object(), documents, ["chunk-1"])

    assert calls[0]["ids"] == ["chunk-1"]
    assert calls[0]["documents"] == ["content"]
    assert calls[0]["embeddings"] == [[0.0]]


def test_build_pdf_documents_rejects_corrupt_pdf(tmp_path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-not-a-real-document")

    with pytest.raises(fitz.FileDataError):
        rag_tool._build_pdf_documents(
            str(bad), title="Bad", arxiv_id="", authors="",
            published="", source="local",
        )


def test_download_pdf_rejects_non_pdf_response(monkeypatch) -> None:
    class FakeResponse:
        headers = {}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"<html>not a pdf</html>"

    monkeypatch.setattr(rag_tool._PDF_SESSION, "get", lambda *args, **kwargs: FakeResponse())
    path, error = rag_tool._download_pdf("https://arxiv.org/pdf/1234.5678")

    assert path is None
    assert "有效的 PDF" in error


def test_download_pdf_rejects_oversized_content_length(monkeypatch) -> None:
    class FakeResponse:
        headers = {"Content-Length": str(2 * 1024 * 1024)}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(rag_tool, "OPENDETECT_MAX_PDF_MB", 1)
    monkeypatch.setattr(rag_tool._PDF_SESSION, "get", lambda *args, **kwargs: FakeResponse())
    path, error = rag_tool._download_pdf("https://arxiv.org/pdf/1234.5678")

    assert path is None
    assert "超过 1 MB" in error


def test_pdf_url_allowlist_rejects_http_and_unknown_hosts() -> None:
    assert rag_tool._is_allowed_pdf_url("https://arxiv.org/pdf/1234.5678") is True
    assert rag_tool._is_allowed_pdf_url("http://arxiv.org/pdf/1234.5678") is False
    assert rag_tool._is_allowed_pdf_url("https://evil.example/paper.pdf") is False
