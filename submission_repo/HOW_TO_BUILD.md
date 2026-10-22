# How to build and submit the bot

Do this in the VS Code terminal (Terminal > New Terminal), which opens
PowerShell. Every command is run from inside `submission_repo`.

## 0. Open the folder and switch on the venv

The "venv" is the private copy of Python and packages for this project.

```
cd "C:\Coding stuff save here\Git Bash save location\HKU hackathon Git Bash save location\ChatbotChallenge2026\submission_repo"
..\venv\Scripts\Activate.ps1
```

`(venv)` should now show at the start of the line. (In Git Bash instead:
`source ../venv/Scripts/activate`.)

## 1. Rebuild the text index (about 5 to 10 minutes)

```
python -m build.index
```

Cuts every scraped page into chunks and stores them in `data/chroma`.
This wipes the image chunks too, so always run step 2 after it.

Optional, before this: `python -m build.scrape` re-downloads both websites
(your copy is from 24 Sep). Only worth it if the sites have changed.

## 2. Describe the images (long: start small)

```
python -m build.images --budget 150
```

Downloads the most useful-looking website images (posters first),
describes each with the vision model, and indexes the descriptions.
Every description is saved in `data/descriptions.json`, so you can stop it
(Ctrl+C) and rerun at any time without paying twice. Once 150 works, run
it again with `--budget 600` (or more) and leave it going.

Your own photos: put them in `data/photos/<zone name>/` (for example
`data/photos/Makerspace A/IMG_0012.jpg`) and rerun this step.

## 3. Test

```
python score.py
python my_test.py
python main.py "What were the winning teams in Pitch New Tech Ideas in 2025?"
```

`score.py` runs `dev_set.json` and shows, for each question, whether the
answer reached the bot at all ("in context?"). `python score.py --real`
also times start-up, like the grader. Nothing should take over 30s.

## 4. Pack for Codabench

```
python make_submission.py
```

Makes `submission.zip` with the code and `data/chroma`, without your `.env`.
