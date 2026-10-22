"""Image pipeline. Workshop 2 blocks 2 and 3.

Downloads collected images, describes the informative ones with the vision
model, caches every description, and indexes them next to the scraped text.

Run from inside submission_repo, after build/scrape.py and build/index.py:

    python -m build.images            # describe the top IMAGE_BUDGET images
    python -m build.images --alt-only # index alt/caption text only, no vision calls
    python -m build.images --budget 300

Your own photos of the building (Level 5) go in data/photos/, one folder
per zone, named exactly as the zone is signed:

    data/photos/Brainstorming Area/IMG_0001.jpg
    data/photos/Makerspace A/IMG_0002.jpg

Every photo found there is described too (they are not counted against the
budget), and its chunks carry zone="Brainstorming Area" and type="photo".

Describing is slow and metered, so this file is resumable: descriptions are
cached in data/descriptions.json by image URL and never regenerated.
"""
import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import re
import threading
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps

# iPhone photos are often .heic, which Pillow cannot open by itself.
# pillow-heif teaches it how. If it is not installed, .heic photos are
# skipped with a message instead of crashing the whole run.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

from bot.llm import describe_image
from bot.store import add_to_store, get_store

# --------------------------------------------------------------------
# The prompt. Whatever this does not ask for cannot be retrieved later.
# --------------------------------------------------------------------
DESCRIPTION_PROMPT = """You are describing one image from the HKU Tam Wing Fan
Innovation Wing websites. The description will be stored and searched later, so
it must be factual, specific and self-contained.

Report only what is visible in this image. Never state a fact you cannot see,
and never give a number you have not counted.

Style rules:
- Plain declarative sentences. No hedging: no "appears to be", "likely",
  "possibly". No confidence levels.
- Do not describe lighting, mood, atmosphere, or where the camera is standing.
- No preamble and no closing summary.

1. WHAT IT IS
One line: photograph, poster, screenshot, diagram, logo or headshot, and the
subject in a few words.

2. TEXT
Copy every readable word exactly as written: spelling, capitalisation,
punctuation and any Chinese characters unchanged. Do not translate, correct or
summarise. Give each one's position, for example: the heading at the top reads
"PITCH NEW TECH IDEAS 2025". Include titles, headings, body text, names,
dates, team names, award names, captions, labels, logos and model numbers.
Write [unreadable] for anything you cannot make out. Never guess at a partly
visible word. On a poster, transcribe the whole poster, including small print.

3. OBJECTS AND COUNTS
For any group of similar items, list them one by one from left to right with
their colour and position, then give the total on a separate line. Count what
is actually in the frame. Do not assume rows are equal length, that a group
forms a grid, or that colours are grouped. If a group is partly out of frame or
occluded, say so and give the number you can see.

4. LAYOUT
State what is next to, in front of, behind, above or facing what.

5. PEOPLE AND ACTIVITY
How many people, and what they are doing. Do not guess at anyone's identity
unless their name is written in the image.

6. COLOURS
- Overall colour scheme in one line, for example: dark blue background with
  white and yellow text.
- Every sign, poster, notice, label and screen: quote its main words exactly
  (up to 10 words), then give its background colour and its text colour, in the
  form: the sign reading "<words>": <colour> background, <colour> text.
- Large objects (furniture, machines, walls, floor): what each is, and its
  colour.
Use common colour names (blue, dark blue, red, light grey...)."""

# A short second pass for images described before section 6 existed, so their
# colours can be added without paying for a whole new description.
COLOUR_PROMPT = """List the colours in this image. The answers will be searched
later for questions like "what colour is the sign that says ...?" or "what
colour scheme does this poster use?". Report only what is visible; no hedging,
no preamble.

6. COLOURS
- Overall colour scheme in one line, for example: dark blue background with
  white and yellow text.
- Every sign, poster, notice, label and screen: quote its main words exactly
  (up to 10 words), then give its background colour and its text colour, in the
  form: the sign reading "<words>": <colour> background, <colour> text.
- Large objects (furniture, machines, walls, floor): what each is, and its
  colour.
Use common colour names (blue, dark blue, red, light grey...)."""

TEXT_PROMPT = """Transcribe every readable word in this image and nothing else.

Copy each exactly as written: spelling, capitalisation, punctuation and any
Chinese characters unchanged. Do not translate, correct or summarise. Give each
one's position, for example: the heading at the top reads "AWARD WINNERS".
Include headings, body text, names, dates, team names, award names, captions,
labels and model numbers. Write [unreadable] for anything you cannot make out.
Never guess at a partly visible word. If there is no text, write
"no text visible"."""

CACHE_PATH = Path("data/descriptions.json")
COLOUR_PATH = Path("data/colours.json")   # colour pass, keyed by image URL
IN_DESCRIPTION = "(in description)"      # marker: colours are in the main description
PHOTO_DIR = Path("data/photos")
PAGES_PATH = Path("data/pages.json")
PART_MAX = 2500            # characters per indexed image chunk
IMAGE_DIR = Path("data/_images")
IMAGE_BUDGET = 600          # how many images to describe, highest priority first
MAX_EDGE = 2000             # downscale before sending; keeps payloads sane
TILE_MIN_EDGE = 1400        # tile anything bigger than this for a second text pass
TILE_TOP_N = 150            # only the top-ranked website images (and your own photos)
                            # get the 4 close-up passes; the rest get one call
WORKERS = 4                 # parallel descriptions; raise carefully, 429 = too many

_lock = threading.Lock()

# WordPress serves the same picture at many sizes: foo-768x512.jpg, foo-150x150.jpg
SIZE_SUFFIX = re.compile(r"-(\d{2,4})x(\d{2,4})(?=\.[A-Za-z]{3,4}$)")
SKIP_WORDS = ("logo", "favicon", "icon", "avatar", "spinner", "placeholder",
              "cropped-", "sprite", "arrow", "bullet")
GOOD_WORDS = ("poster", "pitch", "award", "winner", "competition", "exhibit",
              "showcase", "programme", "program", "schedule", "timetable",
              "notice", "flyer", "infographic", "banner", "slide")


# --------------------------------------------------------------------
# Choosing which images are worth a vision call
# --------------------------------------------------------------------
def canonical(src: str) -> str:
    """Collapse WordPress size variants so one picture is described once."""
    return SIZE_SUFFIX.sub("", src.split("?")[0])


def declared_size(src: str):
    m = SIZE_SUFFIX.search(src)
    return (int(m.group(1)), int(m.group(2))) if m else None


def priority(im: dict) -> int:
    """Higher is more worth describing."""
    blob = " ".join(str(im.get(f, "")) for f in ("src", "alt", "caption", "page")).lower()
    if any(w in blob for w in SKIP_WORDS):
        return -1
    if im["src"].lower().split("?")[0].endswith((".svg", ".gif")):
        return -1
    size = declared_size(im["src"])
    if size and max(size) < 300:            # thumbnails carry nothing readable
        return -1

    score = 0
    score += 6 * sum(w in blob for w in GOOD_WORDS)
    if size:
        w, h = size
        if h > w * 1.2:                     # portrait: usually a poster
            score += 5
        if max(w, h) >= 768:
            score += 2
    if not str(im.get("alt", "")).strip():  # no alt text: the only way in is vision
        score += 3
    if "/news" in blob or "/event" in blob or "/award" in blob:
        score += 2
    return score


def shortlist(images: list[dict], budget: int) -> list[dict]:
    """Deduplicate by canonical URL, drop junk, keep the best `budget`."""
    best: dict = {}
    for im in images:
        key = canonical(im["src"])
        p = priority(im)
        if p < 0:
            continue
        keep = best.get(key)
        # prefer the largest variant of the same picture
        if keep is None or (declared_size(im["src"]) or (0, 0)) > (declared_size(keep["src"]) or (0, 0)):
            im = dict(im, _priority=p)
            best[key] = im
    ranked = sorted(best.values(), key=lambda i: -i["_priority"])
    print(f"{len(images)} image records -> {len(best)} distinct images -> describing {min(budget, len(ranked))}")
    return ranked[:budget]


# --------------------------------------------------------------------
# Fetching and preparing
# --------------------------------------------------------------------
def fetch(src: str) -> Path | None:
    """Download once, rotate upright, downscale. Returns a local path."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    dst = IMAGE_DIR / (hashlib.sha1(src.encode()).hexdigest()[:16] + ".jpg")
    if dst.exists():
        return dst
    try:
        # WordPress keeps the full-size original next to the resized copies
        # (foo-768x512.jpg -> foo.jpg). Small print on a poster is only
        # readable at full size, so try the original first.
        r = None
        for url in dict.fromkeys([canonical(src), src]):
            try:
                r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                break
            except Exception:                             # noqa: BLE001
                r = None
        if r is None:
            raise RuntimeError("could not download")
        im = Image.open(io.BytesIO(r.content))
        im = ImageOps.exif_transpose(im).convert("RGB")   # phone photos are rotated
        im.thumbnail((MAX_EDGE, MAX_EDGE))
        im.save(dst, quality=90)
        return dst
    except Exception as exc:                              # noqa: BLE001
        print("  fetch failed:", src[:70], exc)
        return None


def tiles(path: Path) -> list[Path]:
    """Four overlapping quarters, so small print is near full resolution."""
    im = Image.open(path)
    W, H = im.size
    if max(W, H) < TILE_MIN_EDGE:
        return []
    ox, oy = int(W * 0.15), int(H * 0.15)
    boxes = {
        "top-left":     (0, 0, W // 2 + ox, H // 2 + oy),
        "top-right":    (W // 2 - ox, 0, W, H // 2 + oy),
        "bottom-left":  (0, H // 2 - oy, W // 2 + ox, H),
        "bottom-right": (W // 2 - ox, H // 2 - oy, W, H),
    }
    out = []
    for name, box in boxes.items():
        p = path.with_name(f"{path.stem}_{name}.jpg")
        if not p.exists():
            im.crop(box).save(p, quality=90)
        out.append(p)
    return out


def prepare_local(path: Path) -> Path | None:
    """Rotate and downscale one of your own photos into IMAGE_DIR."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    dst = IMAGE_DIR / (hashlib.sha1(str(path).encode()).hexdigest()[:16] + ".jpg")
    if not dst.exists():
        try:
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            im.thumbnail((MAX_EDGE, MAX_EDGE))
            im.save(dst, quality=90)
        except Exception as exc:                          # noqa: BLE001
            print("  could not open", path, exc)
            return None
    return dst


def describe_one(im: dict) -> tuple[str, str] | None:
    src = im["src"]
    zone = im.get("zone")
    path = prepare_local(Path(im["local"])) if im.get("local") else fetch(src)
    if path is None:
        return None
    prompt = DESCRIPTION_PROMPT
    if zone:
        prompt = (f"This photo was taken inside the Inno Wing, in the zone signed "
                  f'"{zone}". Begin your description with: LOCATION: {zone}\n\n'
                  + DESCRIPTION_PROMPT)
    try:
        parts = [describe_image(str(path), prompt)]
        if not parts[0].strip():
            # An empty reply must not be cached, or this image would never
            # be described: returning None means the next run retries it.
            print("  empty description, will retry next run:", src[:70])
            return None
        for t in (tiles(path) if im.get("_tiles", True) else []):
            parts.append(f"[{t.stem.split('_')[-1]}] " + describe_image(str(t), TEXT_PROMPT))
    except Exception as exc:                              # noqa: BLE001
        print("  describe failed:", src[:70], exc)
        return None
    return src, "\n".join(parts)


# --------------------------------------------------------------------
# Main steps
# --------------------------------------------------------------------
def colour_one(im: dict) -> tuple[str, str] | None:
    """The short colours-only pass for an image described before section 6."""
    src = im["src"]
    path = prepare_local(Path(im["local"])) if im.get("local") else fetch(src)
    if path is None:
        return None
    try:
        text = describe_image(str(path), COLOUR_PROMPT)
    except Exception as exc:                              # noqa: BLE001
        print("  colour pass failed:", src[:70], exc)
        return None
    if not text.strip():
        return None
    return src, text


def load_colours() -> dict:
    return json.loads(COLOUR_PATH.read_text(encoding="utf-8")) if COLOUR_PATH.exists() else {}


def run_pool(fn, todo: list[dict], store: dict, path: Path, label: str) -> None:
    """Run fn over todo in parallel, saving results into store (and to disk
    every 20) so a stop or a used-up limit loses at most 19."""
    done = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for result in pool.map(fn, todo):
            done += 1
            if result:
                src, text = result
                with _lock:
                    store[src] = text
                    if label == "described":
                        colours[src] = IN_DESCRIPTION
            if done % 20 == 0:
                save_json(store, path)
                if label == "described":
                    save_json(colours, COLOUR_PATH)
                print(f"  {done}/{len(todo)} {label}")
    save_json(store, path)
    if label == "described":
        save_json(colours, COLOUR_PATH)


colours: dict = {}


def describe_all(images: list[dict], budget: int) -> dict:
    """Order matters, because the daily limit can run out part-way:
    1. your own photos (full description, close-ups, colours)
    2. the colour pass for images described before colours were asked for
    3. new website images, best first (close-ups only for the top TILE_TOP_N)"""
    global colours
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    colours = load_colours()
    photos = [im for im in own_photos() if im["src"] not in cache]
    ranked = shortlist(images, budget)
    for rank, im in enumerate(ranked):
        im["_tiles"] = rank < TILE_TOP_N
    by_src = {im["src"]: im for im in own_photos() + ranked}
    by_src.update({im["src"]: im for im in images if im["src"] not in by_src})
    need_colour = [by_src.get(src) or {"src": src} for src in cache
                   if src not in colours and not src.startswith("photo:")]
    need_colour += [im for im in own_photos() if im["src"] in cache and im["src"] not in colours]
    new = [im for im in ranked if im["src"] not in cache]
    print(f"{len(cache)} already described: {len(photos)} new photos, "
          f"{len(need_colour)} need the colour pass, {len(new)} new website images to do")

    if photos:
        print("1/3 your photos")
        run_pool(describe_one, photos, cache, CACHE_PATH, "described")
    if need_colour:
        print("2/3 colour pass")
        run_pool(colour_one, need_colour, colours, COLOUR_PATH, "coloured")
    if new:
        print("3/3 new website images")
        run_pool(describe_one, new, cache, CACHE_PATH, "described")
    print(f"{len(cache)} descriptions cached, {sum(1 for v in colours.values() if v != IN_DESCRIPTION)} colour passes")
    return cache


def own_photos() -> list[dict]:
    """Records for every photo in data/photos/<zone>/."""
    out = []
    if PHOTO_DIR.exists():
        for f in sorted(PHOTO_DIR.rglob("*")):
            if f.suffix.lower() == ".heic" and not HEIC_OK:
                print("  skipped", f.name, "- run: pip install pillow-heif")
                continue
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".webp"}:
                zone = f.parent.name if f.parent != PHOTO_DIR else ""
                out.append({"src": "photo:" + f.relative_to(PHOTO_DIR).as_posix(),
                            "local": str(f), "zone": zone, "alt": "", "caption": "",
                            "page": "photo taken in the Inno Wing"})
    if out:
        print(f"{len(out)} of your own photos found in {PHOTO_DIR}")
    return out


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


UPLOAD_YEAR = re.compile(r"/uploads/(20\d\d)/")


def kind_of(text: str, im: dict) -> str:
    """poster / photo / other, from the first line the model wrote."""
    if im.get("local"):
        return "photo"
    head = text[:300].lower()
    if re.search(r"poster|flyer|leaflet|infographic|programme|schedule", head):
        return "poster"
    if re.search(r"photograph|photo", head):
        return "photo"
    return "other"


def split_parts(text: str) -> list[str]:
    """One chunk per description part (main description, each tile), and
    no chunk longer than PART_MAX. A long chunk blurs its embedding."""
    parts, out = re.split(r"\n(?=\[(?:top|bottom)-(?:left|right)\] )", text), []
    for part in parts:
        part = part.strip()
        while len(part) > PART_MAX:
            cut = part.rfind("\n", 0, PART_MAX)
            cut = cut if cut > PART_MAX // 2 else PART_MAX
            out.append(part[:cut])
            part = part[cut:].strip()
        if part:
            out.append(part)
    return out


def index_descriptions(descriptions: dict, images: list[dict]) -> None:
    """Index two things: every image's alt/caption text (free), and the
    descriptions we paid for. All get kind='image' so they can be filtered,
    plus type (poster/photo/other), year, and zone for your own photos."""
    by_src = {im["src"]: im for im in images + own_photos()}
    # The same picture often sits on the home page AND on its own event
    # page. Prefer the event page: its title says what the image is about.
    pages = {}
    for im in images:
        key, old = canonical(im["src"]), pages.get(canonical(im["src"]))
        if old is None or len(urlparse(old["page"]).path) < len(urlparse(im["page"]).path):
            pages[key] = im
    page_info = {}
    if PAGES_PATH.exists():
        for p in json.loads(PAGES_PATH.read_text(encoding="utf-8")):
            page_info[p["url"]] = p

    def meta_for(im: dict, src: str, text: str, kind: str) -> dict:
        page = page_info.get(im.get("page", ""), {})
        m = UPLOAD_YEAR.search(src)
        return {
            "url":   im.get("page", src),
            "title": page.get("title", ""),
            "image": src,
            "kind":  "image",
            "type":  kind_of(text, im) if kind == "described" else "caption",
            "how":   kind,
            "year":  int(m.group(1)) if m else (page.get("year", 0) or 0),
            "site":  page.get("site", ""),
            "zone":  im.get("zone", ""),
        }

    texts, metas, ids = [], [], []
    colour_texts = load_colours()

    # 1. the descriptions, one chunk per part (plus a colours chunk if any)
    for n, (src, text) in enumerate(descriptions.items()):
        im = pages.get(canonical(src)) or by_src.get(src) or {}
        title = page_info.get(im.get("page", ""), {}).get("title", "")
        where = (f"Photo taken in the Inno Wing, {im['zone']}" if im.get("zone")
                 else f"Image on the page \"{title}\" ({im.get('page', '?')})")
        head = " ".join(filter(None, [im.get("alt"), im.get("caption")]))
        meta = meta_for(im, src, text, "described")
        parts = split_parts(text)
        extra = colour_texts.get(src)
        if extra and extra != IN_DESCRIPTION:
            parts.append("COLOURS\n" + extra.strip())
        for j, part in enumerate(parts):
            texts.append(f"{where}\n{head}\n{part}".replace("\n\n\n", "\n\n"))
            metas.append(dict(meta, position=j))
            ids.append(f"img_{n}_{j}")

    # 2. alt and caption for everything else, which costs nothing
    seen = {canonical(s) for s in descriptions}
    for n, (key, im) in enumerate(pages.items()):
        if key in seen:
            continue
        head = " ".join(filter(None, [im.get("alt"), im.get("caption")])).strip()
        if len(head) < 15:
            continue
        title = page_info.get(im.get("page", ""), {}).get("title", "")
        texts.append(f"Image on the page \"{title}\" ({im.get('page', '?')}) captioned: {head}")
        metas.append(meta_for(im, im["src"], head, "caption"))
        ids.append(f"imgcap_{n}")

    store = get_store(reset=False)
    try:
        store.delete(where={"kind": "image"})   # so re-running does not duplicate
    except Exception:
        pass
    add_to_store(store, texts, metas, ids=ids)
    print(f"indexed {len(texts)} image chunks ({len(descriptions)} images described, "
          f"{sum(1 for m in metas if m['how'] == 'caption')} caption-only)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=IMAGE_BUDGET)
    ap.add_argument("--alt-only", action="store_true",
                    help="skip vision calls, index alt/caption text only")
    args = ap.parse_args()

    images = json.loads(Path("data/images.json").read_text(encoding="utf-8"))
    descriptions = {} if args.alt_only else describe_all(images, args.budget)
    if args.alt_only and CACHE_PATH.exists():
        descriptions = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    index_descriptions(descriptions, images)
