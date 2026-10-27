"""Image descriptions that are not in the vector index (for example because
the embedding limit ran out while indexing) are written to
data/keyword_only.json, so the bot's keyword search can still use them.

    python -m build.keyword_only      # no AI calls at all; a few seconds

bot/answer.py adds these chunks to its keyword search when the file exists.
"""
import json
from pathlib import Path

from bot.store import get_store
from build import images as I

OUT = Path("data/keyword_only.json")


def main() -> None:
    got = get_store().get(where={"kind": "image"}, include=["metadatas"])
    indexed = {m.get("image") for m in got["metadatas"] if m.get("how") == "described"}
    descriptions = json.loads(I.CACHE_PATH.read_text(encoding="utf-8"))
    missing = {s: t for s, t in descriptions.items() if s not in indexed}
    images = json.loads(Path("data/images.json").read_text(encoding="utf-8"))

    # Reuse the normal chunking in build/images.py, but catch the chunks
    # instead of embedding them.
    captured = {"texts": [], "metas": []}

    class NoStore:
        def delete(self, where=None):
            pass

    def capture(store, texts, metas, ids=None):
        captured["texts"] += texts
        captured["metas"] += metas

    I.get_store = lambda reset=False: NoStore()
    I.add_to_store = capture
    I.index_descriptions(missing, images)          # prints a line; nothing is embedded
    rows = [{"text": t, "metadata": m}
            for t, m in zip(captured["texts"], captured["metas"]) if m.get("how") == "described"]
    OUT.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"{len(missing)} described images are not in the vector index -> "
          f"{len(rows)} keyword-only chunks saved to {OUT}")


if __name__ == "__main__":
    main()
