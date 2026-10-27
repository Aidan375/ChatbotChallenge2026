"""Indexer. Chunk the scraped text and write it to the store.

Run from inside submission_repo, after build/scrape.py:

    python -m build.index

This WIPES the whole index (text and image chunks) and rebuilds the text
part. Run `python -m build.images` straight after it to put the image
chunks back; that step reuses data/descriptions.json, so it costs nothing
for images already described.

Metadata (year, page_type, category, site) is stored on every chunk so the
bot can filter later. Adding a field later means rebuilding everything.
"""
import json
import re
from pathlib import Path

from bot.store import add_to_store, get_store

CHUNK_SIZE, OVERLAP = 800, 150

# The scraper joins a page's text into one long line, so there are no
# headings left to split on. Splitting at sentence ends instead keeps a
# date and the event it belongs to in the same chunk far more often than
# cutting every 800 characters regardless of where a word ends.
SENTENCE_END = re.compile(r"(?<=[.!?。])\s+|\s+(?=[A-Z][a-z]+:\s)")


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Chunks of about `size` characters, cut at sentence ends, overlapping.

    Overlap exists because a fact split across a boundary is lost: a date
    at char 798 and its event name at char 802 land in different chunks
    and neither answers the question.
    """
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text.strip() else []

    # break into sentences, and hard-split any sentence longer than `size`
    pieces = []
    for s in SENTENCE_END.split(text):
        s = s.strip()
        while len(s) > size:
            cut = s.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(s[:cut])
            s = s[cut:].strip()
        if s:
            pieces.append(s)

    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > size:
            chunks.append(cur)
            # carry the last `overlap` characters (whole words) into the next
            tail = cur[-overlap:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = f"{tail} {p}"
        else:
            cur = f"{cur} {p}".strip()
    if cur.strip():
        chunks.append(cur)
    return chunks


def build_index(pages: list[dict], reset: bool = True):
    texts, metas, ids = [], [], []
    seen, copies = set(), 0
    for page in pages:
        # Some pages are published twice under different URLs. Index the
        # text once, or the copy takes up places in every search result.
        body = " ".join(page["text"].split())
        if body in seen:
            copies += 1
            continue
        seen.add(body)
        title = page.get("title", "")
        for i, piece in enumerate(chunk(page["text"])):
            # Start every chunk with the page title. A chunk from the middle of a
            # page ("Project leader: Pranay Agarwal...") doesn't say which project
            # it is about, so a question naming the project can't find it.
            texts.append(f"{title}\n{piece}")
            metas.append({
                "url":       page["url"],
                "title":     title,
                "position":  i,
                "kind":      "text",
                "year":      page.get("year", 0) or 0,
                "page_type": page.get("page_type", ""),
                "category":  page.get("category", ""),
                "site":      page.get("site", ""),
            })
            ids.append(f"t{len(ids)}")
    store = get_store(reset=reset)
    add_to_store(store, texts, metas, ids=ids)
    print(f"indexed {len(texts)} text chunks from {len(pages) - copies} pages "
          f"({copies} duplicate pages skipped)")
    from build.notes import index_notes   # the wipe above removed the team notes too
    index_notes(store)
    print("next: python -m build.images   (puts the image chunks back)")
    return store


OLDER_SCRAPE = Path("data/pages_2026-09-24.json")


def older_copies(pages: list[dict]) -> list[dict]:
    """Pages that changed or disappeared since the 24 Sep scrape, as they were
    then. Questions may have been written against the older text (the 2025-26
    funding deadline was replaced by the 2026-27 one), so keep both versions."""
    if not OLDER_SCRAPE.exists():
        return []
    now = {p["url"]: " ".join(p["text"].split()) for p in pages}
    out = []
    for p in json.loads(OLDER_SCRAPE.read_text(encoding="utf-8")):
        if now.get(p["url"]) != " ".join(p["text"].split()):
            out.append(dict(p, title=f'{p.get("title", "")} (older version of the page, as of 24 Sep 2026)'))
    print(f"{len(out)} older page versions kept from {OLDER_SCRAPE}")
    return out


if __name__ == "__main__":
    pages = json.loads(Path("data/pages.json").read_text(encoding="utf-8"))
    build_index(pages + older_copies(pages))
