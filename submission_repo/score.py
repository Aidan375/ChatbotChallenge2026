"""Score the bot on dev_set.json, the way the grader would.

    cd submission_repo
    python score.py            # fast: all questions in one Python process
    python score.py --real     # slow but honest: one `python main.py` per
                               # question, so start-up time is counted too
    python score.py --level 3  # only one level

For each question it prints the answer, the expected answer, the time,
and two checks:
  in context?  did the expected answer's words reach the model at all?
               NO = retrieval problem (scraping, images, chunking, search)
  mark         1 / 0.5 / 0, a rough automatic version of the rubric.
               Read the answers yourself as well: the real marking is
               by meaning, so "February 27 2026" for "27 February 2026" is
               right even if this script calls it 0.5.
"""
import argparse
import json
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(errors="replace")

STOP = {"the", "a", "an", "of", "and", "in", "on", "at", "to"}


def words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def mark(got: str, expected: str) -> float:
    g, e = set(words(got)), words(expected)
    if not e:
        return 0.0
    hit = sum(w in g for w in e) / len(e)
    return 1.0 if hit == 1 else 0.5 if hit >= 0.5 else 0.0


def in_context(question: str, expected: str) -> str:
    from bot.answer import format_context, retrieve_mixed
    ctx = set(words(format_context(retrieve_mixed(question))))
    e = words(expected)
    share = sum(w in ctx for w in e) / max(1, len(e))
    return "yes" if share == 1 else "part" if share >= 0.5 else "NO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="run main.py once per question")
    ap.add_argument("--level", type=int, help="only questions of this level")
    ap.add_argument("--file", default="dev_set.json")
    args = ap.parse_args()

    dev = json.load(open(args.file, encoding="utf-8"))
    if args.level:
        dev = [d for d in dev if d["level"] == args.level]
    if not args.real:
        from bot.answer import rag_answer

    total, slowest, by_level = 0.0, 0.0, {}
    for d in dev:
        t = time.time()
        if args.real:
            r = subprocess.run([sys.executable, "main.py", d["question"]],
                               capture_output=True, text=True, encoding="utf-8")
            got = r.stdout.strip()
            if r.stderr.strip():
                print("   stderr:", r.stderr.strip()[-300:])
        else:
            got = rag_answer(d["question"])
        secs = time.time() - t
        m = mark(got, d["answer"])
        total += m
        slowest = max(slowest, secs)
        by_level.setdefault(d["level"], []).append(m)
        flag = "  OVER 30s!" if secs > 30 else ""
        print(f"\nLv{d['level']}  {d['question']}")
        print(f"   got:      {got}")
        print(f"   expected: {d['answer']}")
        print(f"   mark {m}   {secs:.1f}s{flag}   in context? {in_context(d['question'], d['answer'])}")

    print("\n" + "=" * 60)
    for lv in sorted(by_level):
        ms = by_level[lv]
        print(f"  Lv{lv}: {sum(ms)}/{len(ms)}")
    print(f"  total {total}/{len(dev)}   slowest {slowest:.1f}s")


if __name__ == "__main__":
    main()
