"""Pack the code and the vector database into submission.zip for Codabench.

    cd submission_repo
    python make_submission.py             # without your .env (API key)
    python make_submission.py --with-env  # only if the organisers say the
                                          # grader needs your key in the zip

What goes in: main.py, bot/, build/, requirements.txt, .env.example and
data/chroma (the vector database the bot answers from). No __pycache__,
no downloaded images, no scraped JSON (the bot does not read it at runtime).
"""
import argparse
import zipfile
from pathlib import Path

INCLUDE = ["main.py", "requirements.txt", ".env.example", "check_setup.py",
           "bot", "build", "data/chroma"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-env", action="store_true")
    ap.add_argument("--submission-key", action="store_true",
                    help="put .env.submission (the key issued for grading) in the zip as .env")
    ap.add_argument("--out", default="submission.zip")
    args = ap.parse_args()

    if not Path("data/chroma/chroma.sqlite3").exists():
        raise SystemExit("data/chroma is missing: build the index first (see HOW_TO_BUILD.md).")
    if args.submission_key and not Path(".env.submission").exists():
        raise SystemExit(".env.submission is missing.")
    items = INCLUDE + ([".env"] if args.with_env and not args.submission_key else [])
    n = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in items:
            p = Path(item)
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
            for f in files:
                if "__pycache__" in f.parts or f.suffix == ".bak":
                    continue
                z.write(f, f.as_posix())
                n += 1
        if args.submission_key:
            z.write(".env.submission", ".env")   # grading key, stored under the name the bot reads
            n += 1
    size = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}: {n} files, {size:.1f} MB"
          + ("  (grading key inside)" if args.submission_key else "" if args.with_env else "  (no .env inside)"))


if __name__ == "__main__":
    main()
