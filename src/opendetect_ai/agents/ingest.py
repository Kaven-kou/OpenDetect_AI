"""
Ingest Agent —— OpenDetect_AI
负责下载论文 PDF、解析文本、分块存入 Chroma 向量库。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from opendetect_ai.state import AgentState, PaperMeta

MAX_RETRY = 2  # 单篇论文最多重试次数，超过后永久放弃
from opendetect_ai.tools.progress import push_progress
from opendetect_ai.tools.rag_tool import ingest_local_pdf, ingest_paper


def ingest_node(state: AgentState) -> dict:
    _tid = state.get("thread_id", "default")
    """
    Ingest Agent 节点：
    1. 从 state.papers_to_ingest 取出待处理论文
    2. 逐篇下载 PDF 并存入向量库
    3. 更新 ingested_count 和 search_results 的 ingested 标记
    """
    # ── 优先处理本地 PDF（手动上传）─────────────────────────
    local_path = state.get("local_pdf_path", "")
    if local_path:
        print(f"[Ingest] 处理本地 PDF: {local_path}")
        result = ingest_local_pdf.invoke({"file_path": local_path})
        if result.get("status") == "ok":
            chunks = result.get("chunks", 0)
            msg = f"本地 PDF 入库成功，{chunks} 个文本块" if chunks > 0 else "已存在，跳过"
        else:
            msg = f"本地 PDF 入库失败: {result.get('message')}"
        return {
            "ingested_count": state.get("ingested_count", 0) + (1 if result.get("chunks", 0) > 0 else 0),
            "local_pdf_path": "",   # 清空，避免重复处理
            "messages": [AIMessage(content=msg)],
            "error": "" if result.get("status") == "ok" else msg,
        }
    
    papers: list[PaperMeta] = state.get("papers_to_ingest", [])

    if not papers:
        return {
            "ingested_count": state.get("ingested_count", 0),
            "messages": [AIMessage(content="没有待入库的论文。")],
        }

    # ── 过滤掉已入库的 ─────────────────────────────────────────
    pending = [p for p in papers if not p.ingested]
    if not pending:
        return {
            "ingested_count": state.get("ingested_count", 0),
            "messages": [AIMessage(content="所有论文均已入库，无需重复处理。")],
        }

    push_progress(_tid, f"📥 开始入库，共 {len(pending)} 篇...")
    print(f"[Ingest] 待入库论文数: {len(pending)}")

    # ── 逐篇处理 ───────────────────────────────────────────────
    success, skipped_papers, failed = [], [], []

    for paper in pending:
        # 没有 PDF 链接则跳过
        if not paper.pdf_url:
            print(f"[Ingest] 跳过（无 PDF 链接）: {paper.title}")
            paper.ingested = True        # ← 加这行，标记已处理，不再重试
            failed.append(paper.title)
            continue

        push_progress(_tid, f"⬇️ 下载中：{paper.title[:40]}...")
        print(f"[Ingest] 正在处理: {paper.title}")
        result = ingest_paper.invoke({
            "title":     paper.title,
            "pdf_url":   paper.pdf_url,
            "arxiv_id":  paper.arxiv_id,
            "authors":   paper.authors,
            "published": paper.published,
        })

        if result.get("status") == "ok":
            paper.ingested = True
            chunks = result.get("chunks", 0)
            skipped = result.get("skipped", False)
            if skipped:
                skipped_papers.append(f"↩ {paper.title}（已存在，跳过）")
            else:
                success.append(f"✓ {paper.title}（{chunks} 个文本块）")
            push_progress(_tid, f"✓ 入库成功：{paper.title[:35]}（{chunks} 块）")
            print(f"[Ingest] 入库成功: {paper.title}，{chunks} 块")
        else:
            paper.ingested = True   # ← 失败也标记，避免重试死循环
            msg = result.get("message", "未知错误")
            failed.append(f"✗ {paper.title}（{msg}）")
            push_progress(_tid, f"✗ 入库失败：{paper.title[:35]}（{msg[:30]}）")
            print(f"[Ingest] 入库失败: {paper.title}，{msg}")

    # ── 汇总消息 ───────────────────────────────────────────────
    total_ingested = state.get("ingested_count", 0) + len(success)

    lines = [f"入库完成，新增 {len(success)} 篇，跳过 {len(skipped_papers)} 篇，失败 {len(failed)} 篇。\n"]
    if success:
        lines.append("新增：")
        lines.extend(success)
    if skipped_papers:
        lines.append("跳过：")
        lines.extend(skipped_papers)
    if failed:
        lines.append("\n失败：")
        lines.extend(failed)
    summary = "\n".join(lines)
    print(f"[Ingest] {summary}")

    retriable_failed = []
    for p in pending:
        was_success = any(p.title in s for s in success + skipped_papers)
        if was_success:
            continue

        # 没有 arxiv_id 且 PDF 下载失败 → 付费墙/无开放获取，永远无法重试，直接放弃
        if not p.arxiv_id and not p.pdf_url:
            print(f"[Ingest] 放弃（无 arxiv_id 无 PDF 链接）: {p.title}")
            continue

        # 没有 arxiv_id 但有 PDF 链接 → 付费墙导致下载失败，重试无意义
        if not p.arxiv_id and p.pdf_url:
            print(f"[Ingest] 放弃（付费墙，无 arxiv_id）: {p.title}")
            continue

        p.retry_count = getattr(p, "retry_count", 0) + 1
        if p.retry_count <= MAX_RETRY:
            retriable_failed.append(p)
            print(f"[Ingest] 标记可重试: {p.title}（第 {p.retry_count} 次失败，上限 {MAX_RETRY}）")
        else:
            print(f"[Ingest] 放弃重试: {p.title}（已失败 {p.retry_count} 次，超过上限）")

    has_error = bool(failed) and not success and not skipped_papers
    return {
        "papers_to_ingest": papers,
        "ingested_count":   total_ingested,
        "failed_papers":    retriable_failed,
        "error": f"入库失败 {len(failed)} 篇（可能为付费墙限制），已跳过" if has_error else "",
        "messages": [AIMessage(content=summary)],
    }