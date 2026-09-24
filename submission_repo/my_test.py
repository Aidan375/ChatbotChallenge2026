"""Quick self-test for the bot, on the REAL Inno Wing sites.

    cd submission_repo
    python my_test.py

dev_set.json is about the workshop's made-up sample Centre, so its
answers don't exist on the real sites. These questions were written from
real pages instead. Add your own as you find facts on the sites.

For each question it shows two things:
  found page?  was the page holding the answer among the chunks retrieved?
  answer ok?   did the final answer contain the expected text?

found page = NO   -> a retrieval problem (scraping, chunking, k, metadata)
found page = yes but answer ok = NO -> a prompt problem (bot/answer.py)
"""
import time

from bot.answer import rag_answer, retrieve

# (question, text the answer should contain, part of the URL that has it)
TESTS = [
    ("Who supervised the Hacking Wong videogame project?",
     "Chim", "hw-videogame"),
    ("Which course was the Hacking Wong game made for?",
     "COMP3329", "hw-videogame"),
    ("Who was the project leader of Hacking Wong?",
     "Pranay", "hw-videogame"),
    ("What is Alan Chau's role at the Innovation Wing?",
     "Tutor", "/alan"),
    ("Which sections of the Inno Wing safety document should all users read?",
     "7", "/safety"),
    ("Who can attend the TechTalk on learning to simulate and understand the 3D world?",
     "HKU community", "3dworld"),
]


def main():
    passed = 0
    for question, expected, url_part in TESTS:
        t0 = time.time()
        chunks = retrieve(question)
        found = any(url_part in c["metadata"].get("url", "") for c in chunks)
        answer = rag_answer(question)
        ok = expected.lower() in answer.lower()
        passed += ok
        print(f"\nQ: {question}")
        print(f"   answer: {answer[:150]}")
        print(f"   expected to contain: {expected}")
        print(f"   found page? {'yes' if found else 'NO'}   "
              f"answer ok? {'yes' if ok else 'NO'}   ({time.time() - t0:.1f}s)")
        if not found:
            print("   pages it retrieved instead:")
            for c in chunks:
                print("     ", c["metadata"].get("url"))
    print(f"\n{passed}/{len(TESTS)} answers contained the expected text")


if __name__ == "__main__":
    main()
