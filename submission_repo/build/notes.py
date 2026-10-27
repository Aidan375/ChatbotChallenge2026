"""Your team's own notes about the building (Level 5), indexed next to the
website text and image descriptions.

One plain text file per zone in data/notes/, named exactly as the zone is
signed:

    data/notes/Makerspace A.txt
    data/notes/Open Event Area.txt

Write plain facts, with a blank line between paragraphs. Each paragraph
becomes one chunk tagged with its zone. Name the zone and the thing in each
paragraph ("There are 56 lockers at the front of Makerspace A"), because that
is what the search matches on.

    python -m build.notes      # after adding or editing notes: seconds, no image calls

build/index.py runs this at the end too, because it wipes the whole index.
"""
from pathlib import Path

from bot.store import add_to_store, get_store

NOTES_DIR = Path("data/notes")


def note_chunks():
    texts, metas, ids = [], [], []
    if not NOTES_DIR.exists():
        return texts, metas, ids
    for f in sorted(NOTES_DIR.glob("*.txt")):
        zone = f.stem
        paragraphs = [" ".join(p.split()) for p in f.read_text(encoding="utf-8-sig").split("\n\n")]
        for j, p in enumerate(p for p in paragraphs if p):
            texts.append(f"Team notes from inside the Inno Wing, {zone} (counted in person)\n{p}")
            metas.append({"url": f"notes:{zone}", "title": f"Team notes: {zone}",
                          "kind": "note", "zone": zone, "position": j,
                          "year": 0, "site": "notes"})
            ids.append(f"note_{zone}_{j}")
    return texts, metas, ids


def index_notes(store=None) -> None:
    store = store or get_store()
    try:
        store.delete(where={"kind": "note"})   # so re-running does not duplicate
    except Exception:                          # noqa: BLE001
        pass
    texts, metas, ids = note_chunks()
    if texts:
        add_to_store(store, texts, metas, ids=ids)
    print(f"indexed {len(texts)} note chunks from {NOTES_DIR}")


if __name__ == "__main__":
    index_notes()
