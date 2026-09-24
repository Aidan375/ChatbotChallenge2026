"""Scraper. Workshop 1 block 4.

Crawl both websites, extract the readable text, and collect image
records in the same pass. Rebuilds data/pages.json and data/images.json.

Run it from inside the submission_repo folder, so data/ ends up where
build/index.py and the grader expect it:

    cd submission_repo
    python -m build.scrape

Both sites are WordPress sites and publish a sitemap, so we read the
sitemap first and only fall back to crawling if that fails.
"""
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

SITES = [
    "https://innowings.engg.hku.hk",
    "https://innoacademy.engg.hku.hk",
]


def fetch(url: str, tries: int = 3) -> requests.Response:
    """Download a URL, retrying a couple of times if the server is slow."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r
        except Exception:
            if attempt == tries:
                raise
            time.sleep(2 * attempt)   # wait a bit longer each time


def sitemap_locs(xml_bytes: bytes) -> list[str]:
    """Return the text of every <loc> tag in a sitemap XML file."""
    ns = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    return [loc.text.strip() for loc in ET.fromstring(xml_bytes).iter(ns + "loc")
            if loc.text]


# ---------------------------------------------------------------------------
# Step 1: get the list of page URLs
# ---------------------------------------------------------------------------

def urls_from_sitemap(site: str) -> list[tuple[str, str]]:
    """Read the site's sitemap and return (url, page_type) for every page.

    /sitemap.xml on these sites is an *index*: it doesn't list pages
    directly, it lists smaller sitemaps (one for posts, one for pages,
    one for tags, one for authors...). We open the posts and pages ones
    and skip tags/categories/authors, because those are just listing
    pages that repeat content found elsewhere.

    page_type is "news" for WordPress posts (dated news/event articles)
    and "page" for normal pages (About, Facilities, ...). build/index.py
    stores it so the bot can filter by it later.
    """
    try:
        sub_sitemaps = sitemap_locs(fetch(site + "/sitemap.xml").content)
    except Exception as exc:
        print("no sitemap for", site, exc)
        return []

    urls = []
    for sm in sub_sitemaps:
        # WordPress names them wp-sitemap-posts-post-1.xml (news posts) and
        # wp-sitemap-posts-page-1.xml (pages), so "-posts-" keeps both.
        if "-posts-" not in sm:
            continue
        page_type = "news" if "-posts-post-" in sm else "page"
        try:
            urls += [(u, page_type) for u in sitemap_locs(fetch(sm).content)]
        except Exception as exc:
            print("skipped sitemap", sm, exc)
    return urls


def crawl(start_url: str, max_pages: int = 500) -> list[str]:
    """Fallback: follow links to find every page URL on the same site."""
    seen, queue, out = set(), [start_url], []
    domain = urlparse(start_url).netloc

    while queue and len(out) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = requests.get(url, timeout=20).text
        except Exception:
            continue
        out.append(url)

        for a in BeautifulSoup(html, "html.parser").select("a[href]"):
            link = urljoin(url, a["href"]).split("#")[0]
            if urlparse(link).netloc == domain and link not in seen:
                queue.append(link)

    return out


# ---------------------------------------------------------------------------
# Step 2: pull the useful text + images out of one page
# ---------------------------------------------------------------------------

# Where the real content lives on these two sites. Both are built with a
# page builder called Elementor, which wraps each page's own content in
# <div data-elementor-type="wp-post"> (or "wp-page"). Checked on real
# pages with the browser's Inspect tool. The rest are generic fallbacks.
CONTENT_SELECTORS = [
    '[data-elementor-type="wp-post"]',
    '[data-elementor-type="wp-page"]',
    ".entry-content",
    "main",
    "#content",
]

# Pages shorter than this after cleaning are skipped. On these sites they
# are password-protected pages ("This content is password protected")
# or photo galleries with only a heading.
MIN_TEXT_CHARS = 80


def wp_metadata(site: str) -> dict:
    """Ask WordPress for the date and categories of every post and page.

    The pages themselves don't show a publish date anywhere in their HTML,
    but every WordPress site has a built-in data feed (the "REST API",
    under /wp-json/) that lists them. We read it once, 100 items at a
    time, and return {url: {"year": 2025, "category": "SIG Projects"}}.
    """
    def get_all(path):
        items, page = [], 1
        while True:
            try:
                r = fetch(f"{site}/wp-json/wp/v2/{path}&per_page=100&page={page}")
            except Exception:
                break                       # ran past the last page, or no API
            batch = r.json()
            if not batch:
                break
            items += batch
            if page >= int(r.headers.get("X-WP-TotalPages", page)):
                break
            page += 1
        return items

    cats = {c["id"]: BeautifulSoup(c["name"], "html.parser").get_text()
            for c in get_all("categories?_fields=id,name")}
    meta = {}
    for kind in ("posts", "pages"):
        for item in get_all(f"{kind}?_fields=link,date,categories"):
            meta[item["link"].rstrip("/")] = {
                "year": int(item["date"][:4]),
                "category": ", ".join(cats.get(c, "") for c in item.get("categories", [])
                                      if cats.get(c) and cats[c] != "Uncategorized"),
            }
    return meta


def extract(html: str, url: str) -> dict:
    """Return {"url", "title", "text", "images": [...]} for one page."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    # Titles end in " – Innovation Wing" / " – Innovation Academy"; drop that.
    title = re.sub(r"\s+[–-]\s+Innovation (Wing|Academy)$", "", title)

    # Throw away parts that are never real content.
    for tag in soup.select("nav, footer, script, style, noscript, form"):
        tag.decompose()
    # Remove the site-wide header, but not a heading block inside an
    # article (WordPress puts the post title in one of those).
    for tag in soup.select("header"):
        if not tag.find_parent(["article", "main"]):
            tag.decompose()
    # "Related posts" card grids. Each one repeats another page's title
    # and first lines ("Bambu Lab 3D Printer Read More »") on hundreds of
    # pages, which is exactly the noise the workshop warned about.
    for tag in soup.select(".elementor-posts-container, .elementor-widget-posts, "
                           ".elementor-post-navigation"):
        tag.decompose()

    body = None
    for sel in CONTENT_SELECTORS:
        body = soup.select_one(sel)
        if body:
            break
    if body is None:            # nothing matched: use what's left
        body = soup.body or soup

    images = []
    for img in body.select("img"):   # only images inside the content
        src = img.get("src")
        if not src:
            continue
        fig = img.find_parent("figure")
        images.append({
            "src":     urljoin(url, src),     # relative -> absolute
            "alt":     img.get("alt", ""),
            "caption": (fig.find("figcaption").get_text(strip=True)
                        if fig and fig.find("figcaption") else ""),
            "page":    url,
        })

    return {
        "url":    url,
        "title":  title,
        "text":   body.get_text(" ", strip=True),
        "images": images,
    }


# ---------------------------------------------------------------------------
# Step 3: run everything and save
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    # Don't crash if the terminal can't display a character (e.g. an emoji).
    sys.stdout.reconfigure(errors="replace")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    WORKERS = 5   # pages downloaded at the same time; keep it small to be polite

    def download(url, page_type, site, meta):
        page = extract(fetch(url).text, url)
        info = meta.get(url.rstrip("/"), {})
        page["page_type"] = page_type
        page["year"] = info.get("year", 0)          # 0 = unknown
        page["category"] = info.get("category", "")
        page["site"] = urlparse(site).netloc.split(".")[0]   # "innowings" / "innoacademy"
        time.sleep(0.2)
        return page

    pages, skipped, too_short = [], [], []
    for site in SITES:
        urls = urls_from_sitemap(site) or [(u, "") for u in crawl(site)]
        meta = wp_metadata(site)
        print(f"{site}: {len(urls)} URLs, dates for {len(meta)}")

        # Instead of downloading one page, waiting, then the next, we keep
        # WORKERS downloads going at once. Same pages, much less waiting.
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            jobs = {pool.submit(download, u, t, site, meta): u for u, t in urls}
            for i, job in enumerate(as_completed(jobs), 1):
                url = jobs[job]
                try:
                    page = job.result()
                    if len(page["text"]) >= MIN_TEXT_CHARS:
                        pages.append(page)
                    else:
                        too_short.append(url)
                except Exception as exc:
                    print("skipped", url, exc)
                    skipped.append(url)
                if i % 50 == 0 or i == len(urls):
                    print(f"  {i}/{len(urls)} done")

    pages.sort(key=lambda p: p["url"])   # same order every run

    # encoding="utf-8" matters on Windows: without it Python uses the old
    # Windows character set, which can't store emoji or Chinese text.
    Path("data").mkdir(exist_ok=True)
    Path("data/pages.json").write_text(
        json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")

    images = [im for p in pages for im in p["images"]]
    Path("data/images.json").write_text(
        json.dumps(images, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"{len(pages)} pages, {len(images)} images, "
          f"{len(skipped)} failed to download, {len(too_short)} empty/password-protected")
    print(f"pages with no year: {sum(1 for p in pages if not p['year'])}")

    # Quick sanity check: the start of a few pages should be real
    # content, not menu words like "Home About Events Contact".
    for p in pages[:3]:
        print("\n---", p["url"], "\n", p["text"][:200])
