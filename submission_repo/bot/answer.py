"""Your chatbot.

Two functions must exist with these exact names and signatures:
rag_answer and rag_answer_batch. main.py calls them and prints whatever
they return, one line per question.

How an answer is made
1. Retrieve from three places and merge them:
     - meaning search (embeddings), good when the question is worded
       differently from the page
     - keyword search (BM25), good for names, dates and exact phrases
       like "Pitch New Tech Ideas", which embeddings often miss
     - a few chunks forced from image descriptions, because the index holds
       thousands of text chunks and few image chunks, so a plain search
       almost never surfaces an image (Level 3 needs one)
2. Add the chunks either side of the best hits, so a date cut off from its
   heading by a chunk boundary still arrives with it.
3. Ask the model for a bare answer, then tidy the format in code.
4. Guard the 30 second limit: a quick no-context answer is prepared in
   parallel and used if the main path errors or runs out of time.
   No live web search anywhere: everything comes from data/chroma.
"""
import json
import math
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from bot.llm import chat
from bot.store import get_store, query

CONFIG = {
    "k": 8,             # chunks from meaning (embedding) search
    "k_keyword": 8,     # chunks from keyword search
    "k_image": 3,       # pictures forced from meaning search over image descriptions
    "k_image_kw": 3,    # pictures forced from keyword search over image descriptions
    "neighbours": 5,    # add the chunks either side of this many top text hits
    "per_page": 3,      # at most this many search hits from any one page
    "max_chars": 24000, # cap on context size; bigger prompts are slower
    "deadline": 20,     # seconds; the limit is 30 per question, including start-up
    "k_count": 20,      # chunks from each search for "how many" questions
    "count_chars": 40000,  # context cap for the counting path
    "count_tokens": 1500,  # room for the model to list the items before the number
    "model": None,      # None = gpt-4o-mini (bot/llm.py default)
}

SYSTEM_PROMPT = """You answer trivia questions about the HKU Tam Wing Fan Innovation
Wing (Inno Wing) and the HKU Innovation Academy, using context taken from their
websites and from descriptions of images on those websites.

How to answer:
- If the context contains the answer, use it. Where the context disagrees with
  what you think you know, the context is correct.
- If the context does not contain it, answer from your own general knowledge.
- General-knowledge questions (what something is also known as, what an
  abbreviation stands for, a definition): give the standard textbook answer,
  even if the context mentions a related term.
- Pages often mention several years or editions. Use the one the question asks
  about, and read every chunk before deciding.
- A few pages appear twice: as they are now, and as an "older version of the
  page, as of 24 Sep 2026". Use the current version, unless the question asks
  about a year, round or edition that only the older version mentions.
- Reply with the answer only: no sentence around it, no explanation, no
  preamble, no source.
- How many: reply with a number in digits and nothing else.
- Several items (teams, winners, materials): give all of them, separated by
  commas, on one line.
- Words written on something: copy them exactly as the context spells them.
- Never reply that you do not know and never hedge. A wrong guess and a refusal
  score the same, so always commit to your best answer."""

CLOSED_BOOK_PROMPT = """You answer trivia questions about the HKU Tam Wing Fan Innovation
Wing at the University of Hong Kong. Reply with the answer only, as short as
possible: no sentence, no explanation. For how many, a number in digits. Never
refuse: always give your best guess."""

# --------------------------------------------------------------------------
# Keyword search (BM25), built from the documents already in data/chroma
# --------------------------------------------------------------------------
TOKEN = re.compile(r"[a-z0-9]+|[一-鿿]")
STOP = set("""a an the of to in on for and or is are was were be been by with at from
as that this these those what which who whom whose when where why how many much
does did do it its into than then there their they them you your i we our about
can could would should will shall may might has have had not no""".split())


def stem(t: str) -> str:
    """Crude word-ending strip, so "winning" and "WINNERS" match each other."""
    for suf in ("ings", "ing", "ers", "er", "es", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def tokens(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOP]


class KeywordIndex:
    """Classic BM25 over every chunk in the store. Built once per process."""

    def __init__(self, docs: list[str], metas: list[dict], k1=1.4, b=0.75):
        self.docs, self.metas, self.k1, self.b = docs, metas, k1, b
        self.tf = [Counter(tokens(d)) for d in docs]
        self.lens = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.post[t].append(i)
        self.n = len(docs)
        # where each chunk sits on its page, for fetching neighbours
        self.by_pos = {}
        self.by_pic = {}
        for i, m in enumerate(metas):
            if m.get("kind", "text") == "text":
                self.by_pos[(m.get("url"), m.get("position"))] = i
            elif m.get("image") is not None and m.get("position") is not None:
                # each described picture's chunks, to fetch its main description
                self.by_pic[(m.get("image"), m.get("position"))] = i

    def search(self, question: str, k: int, kind: str = None) -> list[dict]:
        scores = defaultdict(float)
        for t in set(tokens(question)):
            hits = self.post.get(t)
            if not hits:
                continue
            idf = math.log(1 + (self.n - len(hits) + 0.5) / (len(hits) + 0.5))
            for i in hits:
                f = self.tf[i][t]
                norm = 1 - self.b + self.b * self.lens[i] / self.avg
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        ranked = sorted(scores, key=scores.get, reverse=True)
        if kind:
            ranked = [i for i in ranked if self.metas[i].get("kind", "text") == kind]
        return [self.item(i, scores[i]) for i in ranked[:k]]

    def item(self, i: int, score: float = 0.0) -> dict:
        return {"text": self.docs[i], "metadata": self.metas[i], "distance": -score}

    def zone_notes(self, question: str) -> list[dict]:
        """Every team-notes chunk for a zone the question names. Notes are few
        and short, and hold facts counted in person, so they always go in."""
        q = question.lower()
        return [self.item(i) for i, m in enumerate(self.metas)
                if m.get("kind") == "note" and m.get("zone")
                and m["zone"].lower() in q]

    def main_chunk(self, meta: dict):
        """The first chunk of a described picture: the full description, which
        holds the transcribed text. A close-up chunk alone can miss it."""
        i = self.by_pic.get((meta.get("image"), 0))
        return self.item(i) if i is not None else None

    def neighbours(self, meta: dict) -> list[dict]:
        url, pos = meta.get("url"), meta.get("position")
        if pos is None or meta.get("kind", "text") != "text":
            return []
        out = []
        for p in (pos - 1, pos + 1):
            i = self.by_pos.get((url, p))
            if i is not None:
                out.append(self.item(i))
        return out


# The competition's own page lists example questions with their answers. It
# is not about the Inno Wing, and it pulls wrong answers (its 1st-round
# funding date was given for a 2nd-round question), so it is never used.
SKIP_PAGES = ("/aichallenge", "ai-chatbot-challenge-inno-trivia")


def skipped(meta: dict) -> bool:
    url = meta.get("url") or ""
    return any(s in url for s in SKIP_PAGES)


_keyword = None
_keyword_lock = threading.Lock()


def keyword_index() -> KeywordIndex:
    global _keyword
    with _keyword_lock:
        if _keyword is None:
            got = get_store().get(include=["documents", "metadatas"])
            pairs = list(zip(got["documents"] or [], got["metadatas"] or []))
            # Descriptions that could not be embedded are still searchable by keyword.
            extra = Path("data/keyword_only.json")
            if extra.exists():
                try:
                    pairs += [(r["text"], r["metadata"])
                              for r in json.loads(extra.read_text(encoding="utf-8"))]
                except Exception:                       # noqa: BLE001
                    pass
            keep = [(d, m) for d, m in pairs if not skipped(m)]
            _keyword = KeywordIndex([d for d, _ in keep], [m for _, m in keep])
    return _keyword


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve(question: str, k: int = None, where: dict = None) -> list[dict]:
    """Meaning search only. Kept separate so my_test.py can check whether
    the right page was fetched at all (a retrieval failure) as opposed to
    fetched but answered wrongly (a prompt failure)."""
    global _vector_ok
    k = k or CONFIG["k"]
    if not _vector_ok:
        return []
    try:
        hits = query(get_store(), question, k=k + 6, where=where)   # spare room for skipped pages
    except Exception as exc:                            # noqa: BLE001
        # Meaning search needs an embedding call. If that fails (for example
        # the embedding limit is used up), carry on with keyword search only.
        _vector_ok = False
        print(f"[warning] meaning search unavailable, keyword search only: {str(exc)[:150]}",
              file=sys.stderr)
        return []
    return [c for c in hits if not skipped(c["metadata"])][:k]


_vector_ok = True


def key(c: dict) -> str:
    """Identity of a chunk by its body (the first line is the page title).
    Some pages exist twice on the site under different URLs; keying on the
    body stops the same text taking two places in the context."""
    return " ".join(c["text"].split("\n", 1)[-1].split())[:300]


def retrieve_mixed(question: str) -> list[dict]:
    """Meaning + keyword search fused by rank, then images, then neighbours."""
    kw = keyword_index()
    vec = retrieve(question)
    words = kw.search(question, CONFIG["k_keyword"])
    try:
        vec_images = retrieve(question, k=CONFIG["k_image"] * 3, where={"kind": "image"})
    except Exception:                                   # noqa: BLE001
        vec_images = []
    kw_images = kw.search(question, 40, kind="image")
    # One chunk per picture: a picture is split into several chunks, and
    # without this one picture can take every image slot.
    images, pictures = [], set()
    for found, limit in ((vec_images, CONFIG["k_image"]), (kw_images, CONFIG["k_image_kw"])):
        taken = 0
        for c in found:
            pic = c["metadata"].get("image") or key(c)
            if pic in pictures:
                continue
            pictures.add(pic)
            images.append(c)
            taken += 1
            if taken >= limit:
                break

    # Reciprocal rank fusion: a chunk ranked high by either search rises.
    score, chunk_of = defaultdict(float), {}
    for ranked in (vec, words):
        for rank, c in enumerate(ranked):
            score[key(c)] += 1 / (60 + rank)
            chunk_of.setdefault(key(c), c)
    fused, per_page = [], defaultdict(int)
    for k_ in sorted(score, key=score.get, reverse=True):
        c = chunk_of[k_]
        page = c["metadata"].get("image") or c["metadata"].get("url")
        if per_page[page] < CONFIG["per_page"]:   # stop one page crowding out the rest
            per_page[page] += 1
            fused.append(c)

    merged, seen = [], set()

    def add(c):
        if key(c) not in seen:
            seen.add(key(c))
            merged.append(c)

    top_text = {key(c) for c in fused
                if c["metadata"].get("kind", "text") == "text"}
    top_text = [k_ for k_ in (key(c) for c in fused) if k_ in top_text][:CONFIG["neighbours"]]
    top_text += [key(c) for c in words[:3]]          # best keyword hits always get neighbours
    # Image chunks first into `seen`, so they are never squeezed out of the
    # context by text: Level 3 is 40% of the preliminary round.
    picked_images = []
    for c in images:
        for c2 in (kw.main_chunk(c["metadata"]), c):
            if c2 is not None and key(c2) not in seen:
                seen.add(key(c2))
                picked_images.append(c2)
    image_chars = sum(len(c["text"]) for c in picked_images)

    for c in kw.zone_notes(question):                 # team notes first, never trimmed
        add(c)
    for c in fused:
        add(c)
        if key(c) in top_text:
            for n in kw.neighbours(c["metadata"]):
                add(n)

    # keep the text under the size cap, best first, then the images
    out, total = [], 0
    room = max(8000, CONFIG["max_chars"] - image_chars)
    for c in merged:
        if total + len(c["text"]) > room and out:
            continue
        out.append(c)
        total += len(c["text"])
    return out + picked_images


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        m = c["metadata"]
        if m.get("kind") == "image":
            head = f"[image on {m.get('url', '?')}]"
        else:
            year = f" | {m['year']}" if m.get("year") else ""
            head = f"[{m.get('title', '')} | {m.get('url', '?')}{year}]"
        parts.append(f"{head}\n{c['text']}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Output format
# --------------------------------------------------------------------------
REFUSALS = re.compile(
    r"^(i (do not|don't) know|the (context|text|image|information) "
    r"(does not|doesn't) (provide|contain|specify|mention|say)[^.]*\.?)\s*",
    re.I)
PREFIXES = re.compile(
    r"^(the answer is|answer:|it is|there are|there is|they are)\s+", re.I)
# Only "how many": "number of" also covers phone and room numbers.
COUNT_Q = re.compile(r"\bhow many\b", re.I)
UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
WORD = re.compile(r"[a-z]+", re.I)
# whole numbers only: the "3" in "3D" is not a number of its own
DIGITS = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\w])|(?<![\w.,])\d+(?![\w])")


def word_numbers(text: str) -> list[tuple[int, str]]:
    """Numbers written as words, as (position, value) pairs.
    "seven" -> 7, "thirty-one" -> 31, "one hundred and twenty" -> 120."""
    words = list(WORD.finditer(text))
    out, i = [], 0
    while i < len(words):
        w = words[i].group().lower()
        if w not in UNITS and w not in TENS:
            i += 1
            continue
        start, value, j, last_end = words[i].start(), 0, i, None
        while j < len(words):
            m = words[j]
            w = m.group().lower()
            gap = text[last_end:m.start()] if last_end is not None else ""
            if last_end is not None and gap.strip(" -"):
                break                                   # words must touch: space or hyphen
            if w in UNITS and value % 10 == 0 and (value % 100 == 0 or UNITS[w] < 10):
                value += UNITS[w]
            elif w in TENS and value % 100 == 0:
                value += TENS[w]
            elif w == "hundred" and 0 < value < 100:
                value *= 100
            elif (w == "and" and value and value % 100 == 0 and j + 1 < len(words)
                  and words[j + 1].group().lower() in {**UNITS, **TENS}):
                pass                                    # "one hundred and twenty"
            else:
                break
            last_end = m.end()
            j += 1
        out.append((start, str(value)))
        i = j
    return out


def all_numbers(text: str) -> list[tuple[int, str]]:
    """Every number in the text, words and digits, in reading order."""
    found = word_numbers(text) + [(m.start(), m.group().replace(",", ""))
                                  for m in DIGITS.finditer(text)]
    return sorted(found)


def count_answer(out: str, question: str) -> str | None:
    """The number a how-many answer gives, or None if it gives none: the
    first number in the reply, word or digit, skipping any number that
    already appears in the question (a year, the 3 in 3D)."""
    in_question = {n for _, n in all_numbers(question)}
    for _, n in all_numbers(out):
        if n not in in_question:
            return n
    return None


LOOKS_LIKE_REFUSAL = re.compile(
    r"not (provide|contain|specify|mention)|do not know|don't know|unknown|"
    r"no information|cannot (determine|be determined)", re.I)


def tidy(reply: str, question: str) -> str:
    """Strip the prose the rubric penalises, put the answer on one line, and
    return a bare number when a number was asked for. A correct fact
    wrapped in hedging is marked partial."""
    # one line only: main.py prints one line per question
    lines = [re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", ln).strip()
             for ln in (reply or "").splitlines()]
    out = ", ".join(ln for ln in lines if ln)
    out = out.replace("**", "").strip().strip('"').strip()
    out = REFUSALS.sub("", out).strip()
    out = PREFIXES.sub("", out).strip()
    out = re.sub(r"^(my best guess is|best guess:|probably|i think)\s+", "", out, flags=re.I).strip()
    out = out.strip('"').strip()

    if COUNT_Q.search(question):
        n = count_answer(out, question)
        if n is not None:
            return n

    # one trailing full stop on a short answer reads as a sentence, not a value
    if out.endswith(".") and out.count(".") == 1 and len(out.split()) <= 6:
        out = out[:-1]
    return out.strip()


# --------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------
def ask_model(system: str, user: str) -> str:
    return chat([{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                model=CONFIG["model"], max_tokens=200)


def answer_with_context(question: str) -> str:
    context = format_context(retrieve_mixed(question))
    user = f"Context:\n{context}\n\nQuestion: {question}"
    answer = tidy(ask_model(SYSTEM_PROMPT, user), question)

    # A refusal scores zero, so if the model refused anyway, make it commit.
    if not answer or LOOKS_LIKE_REFUSAL.search(answer):
        answer = tidy(ask_model(SYSTEM_PROMPT, user + "\n\nAnswer with your single "
                                "best guess. Output only the answer."), question)
    return answer


# --------------------------------------------------------------------------
# Counting ("how many") questions, Level 4: list the evidence, then count
# --------------------------------------------------------------------------
COUNT_PROMPT = """You answer "how many" trivia questions about the HKU Tam Wing Fan
Innovation Wing (Inno Wing) and the HKU Innovation Academy, using the numbered
context items below (website text and descriptions of images).

Work in two steps.
1. Under the heading ITEMS, list every distinct thing in the context that the
   question is counting, one per line, as:
   - <what it is, in at most 8 words> | <how many of it, if the context says; otherwise 1> | [N]
   where [N] is the context item it came from. Count the same thing once even
   when several context items mention it (the same poster on two pages, the
   same machine described twice). Leave out anything that does not meet every
   condition in the question (year, type, place).
2. The last line must be exactly: ANSWER: <the total, in digits>
If a context item states the total directly and it is larger than your list,
use the stated total. Never answer 0 unless nothing at all matches."""

YEAR = re.compile(r"\b(20\d\d)\b")
POSTER = re.compile(r"\bposters?\b", re.I)
COUNT_ANSWER = re.compile(r"ANSWER:\s*\**\s*(\d[\d,]*)", re.I)


def filtered_posters(question: str) -> list[dict]:
    """Every described poster matching the question's year, one chunk per
    picture (its main description, shortened). Vector search alone returns
    the k nearest, which caps the count at k; a metadata filter returns all."""
    y = YEAR.search(question)
    year = int(y.group(1)) if y else None
    kw = keyword_index()      # every chunk, including keyword-only descriptions
    best = {}
    for d, m in zip(kw.docs, kw.metas):
        if m.get("kind") != "image" or m.get("type") != "poster":
            continue
        if year is not None and m.get("year") != year:
            continue
        pic = m.get("image") or d[:80]
        if pic not in best or (m.get("position") or 0) < (best[pic]["metadata"].get("position") or 0):
            # Short entries, so every poster fits: the page it is on and the
            # first lines of its description say what it is.
            best[pic] = {"text": " ".join(d[:350].split()), "metadata": m, "distance": 0.0}
    return list(best.values())


def retrieve_for_count(question: str) -> list[dict]:
    kw = keyword_index()
    found = kw.zone_notes(question)
    found += filtered_posters(question) if POSTER.search(question) else []
    found += retrieve(question, k=CONFIG["k_count"])
    found += kw.search(question, CONFIG["k_count"])
    for c in kw.search(question, 3):                  # keep lists that run across chunks
        found += kw.neighbours(c["metadata"])
    out, seen, total = [], set(), 0
    for c in found:
        k_ = key(c)
        if k_ in seen or total + len(c["text"]) > CONFIG["count_chars"]:
            continue
        seen.add(k_)
        out.append(c)
        total += len(c["text"])
    return out


def answer_by_counting(question: str):
    """A number in digits, or None if the model did not give one."""
    chunks = retrieve_for_count(question)
    numbered = "\n\n".join(f"[{i + 1}] " + format_context([c]) for i, c in enumerate(chunks))
    reply = chat([{"role": "system", "content": COUNT_PROMPT},
                  {"role": "user", "content": f"Context:\n{numbered}\n\nQuestion: {question}"}],
                 model=CONFIG["model"], max_tokens=CONFIG["count_tokens"])
    found = COUNT_ANSWER.findall(reply or "")
    return found[-1].replace(",", "") if found else None


def closed_book(question: str) -> str:
    return tidy(ask_model(CLOSED_BOOK_PROMPT, question), question)


def in_background(fn, *args) -> dict:
    """Run fn in a daemon thread (one that cannot keep the program alive
    after main.py has printed). Returns a dict that fills in when done."""
    box = {}

    def run():
        try:
            box["value"] = fn(*args)
        except Exception as exc:                        # noqa: BLE001
            box["error"] = exc
    box["thread"] = threading.Thread(target=run, daemon=True)
    box["thread"].start()
    return box


def rag_answer(question: str) -> str:
    """One question in, one answer out. 30 second budget."""
    started = time.time()
    backup = in_background(closed_book, question)       # safety net
    main = in_background(answer_with_context, question)
    # "How many" questions also go down the counting path, in parallel, so
    # the normal answer is still there if counting is slow or fails.
    count = in_background(answer_by_counting, question) if COUNT_Q.search(question) else None

    main["thread"].join(CONFIG["deadline"])
    answer = main.get("value") or ""
    if count is not None:
        count["thread"].join(max(0.1, CONFIG["deadline"] - (time.time() - started)))
        answer = count.get("value") or answer
    if not answer:
        left = max(0.5, CONFIG["deadline"] + 2 - (time.time() - started))
        backup["thread"].join(left)
        answer = backup.get("value") or ""
    return " ".join(answer.split())                      # never more than one line


def rag_answer_batch(questions: list[str]) -> list[str]:
    """Many questions in, the same number of answers out, in order.
    Each question has its own try, so one failure cannot blank the rest."""
    out = []
    for q in questions:
        try:
            out.append(rag_answer(q))
        except Exception:                                # noqa: BLE001
            out.append("")
    return out
