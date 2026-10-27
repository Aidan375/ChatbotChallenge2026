"""Shows what the counting path does for one question (uses 2 AI calls).

    python debug_count.py
"""
import sys
import time

from bot import answer as A
from bot.llm import chat

q = sys.argv[1] if len(sys.argv) > 1 else \
    "How many workshop posters are on the Innovation Academy websites for 2025?"
t = time.time()
chunks = A.retrieve_for_count(q)
print(f"context: {len(chunks)} chunks, {sum(len(c['text']) for c in chunks)} chars, "
      f"{time.time() - t:.1f}s to fetch")
numbered = "\n\n".join(f"[{i + 1}] " + A.format_context([c]) for i, c in enumerate(chunks))
t = time.time()
try:
    reply = chat([{"role": "system", "content": A.COUNT_PROMPT},
                  {"role": "user", "content": f"Context:\n{numbered}\n\nQuestion: {q}"}],
                 model=A.CONFIG["model"], max_tokens=A.CONFIG["count_tokens"])
    print(f"--- AI reply ({time.time() - t:.1f}s) ---")
    print(reply)
    print("--- number found:", A.COUNT_ANSWER.findall(reply or ""))
except Exception as exc:                                  # noqa: BLE001
    print(f"--- AI call FAILED after {time.time() - t:.1f}s: {type(exc).__name__}: {exc}")
