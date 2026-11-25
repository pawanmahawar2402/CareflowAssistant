"""M4 — Document loaders (one per format, with a text fallback) and structure-aware chunking."""
from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from careflow.config import settings

log = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 900
CHUNK_OVERLAP_CHARS = 120


@dataclass
class Chunk:
    source: str       # file stem, used as the citation prefix
    title: str        # document title
    section: str      # heading path, e.g. "Pre-authorization > Urgent requests"
    index: int        # chunk index within the document
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.source}#{self.index}"


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_html(path: Path) -> str:
    raw = _load_text(path)
    raw = re.sub(r"(?is)<(script|style).*?</\1>", "", raw)
    for n in range(1, 7):  # keep headings as markdown so chunking stays structure-aware
        raw = re.sub(rf"(?is)<h{n}[^>]*>(.*?)</h{n}>", lambda m, n=n: "\n" + "#" * n + " " + m.group(1) + "\n", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</tr>", "\n", raw)
    return html.unescape(re.sub(r"<[^>]+>", "", raw))


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


_LOADERS = {".md": _load_text, ".txt": _load_text, ".html": _load_html, ".htm": _load_html, ".pdf": _load_pdf}


def load_document(path: Path) -> str:
    """Load by extension; on any loader failure fall back to a plain-text read."""
    loader = _LOADERS.get(path.suffix.lower(), _load_text)
    try:
        text = loader(path)
        if text.strip():
            return text
    except Exception as exc:
        log.warning("loader for %s failed (%s); falling back to plain text", path.name, exc)
    return _load_text(path)


# ── Structure-aware chunking ──────────────────────────────────────────────────

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _sections(text: str) -> list[tuple[str, str, str]]:
    """Split markdown into (doc_title, section_path, body) by headings."""
    title, stack, out, buf = "", [], [], []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            out.append((title, " > ".join(h for _, h in stack), body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            flush()
            level, head = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = head
            stack[:] = [(lv, h) for lv, h in stack if lv < level] + [(level, head)]
        else:
            buf.append(line)
    flush()
    return out


def _split_long(body: str) -> list[str]:
    """Paragraph-first split with a small overlap for oversize sections."""
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n|\n(?=[-*] )", body):
        if cur and len(cur) + len(para) > MAX_CHUNK_CHARS:
            parts.append(cur.strip())
            cur = cur[-CHUNK_OVERLAP_CHARS:]
        cur += "\n" + para
    if cur.strip():
        parts.append(cur.strip())
    return parts


def chunk_document(path: Path) -> list[Chunk]:
    text = load_document(path)
    chunks: list[Chunk] = []
    for title, section, body in _sections(text):
        for piece in _split_long(body):
            # Prefix the section path so chunks stay self-describing for dense + BM25 search.
            chunks.append(Chunk(path.stem, title or path.stem, section, len(chunks),
                                f"{section}\n{piece}" if section else piece))
    return chunks


def iter_corpus() -> list[Path]:
    root = settings.corpus_dir
    return sorted(p for p in root.iterdir() if p.suffix.lower() in _LOADERS)
