# the llm judge: for each eval prompt, is the small answer good enough, is the large one, which is better.
# same three questions i answer by hand in app/label.py, so the two can be compared with cohen's kappa.
# reuses generate.py's cache, retries and pacing. usage: python -m pipeline.judge (needs JUDGE_* in .env)
import argparse
import csv
import hashlib
import json
import os
import re

from pipeline.generate import DB_PATH, cache_key, connect_db, run_calls, strip_thoughts
from pipeline.label_set import LABEL_SET_PATH

# shortcut: the local judge has an 8k context, so long prompts and answers get trimmed.
# the judge can miss problems at the end of a very long answer. raise these with a bigger judge.
# the marker says the grader cut it, otherwise the judge reads it as the model getting cut off.
PROMPT_CHARS = 3000
ANSWER_CHARS = 4000

JUDGE_TEMPLATE = """You are grading two AI answers to the same user request.

USER REQUEST:
{prompt}

ANSWER A:
{a}

ANSWER B:
{b}

For each answer, decide if it is good enough: it does what the user asked, is correct, and a typical user would be satisfied without asking again. Length and style don't matter unless the user asked for them. An empty answer, or one the model itself stopped mid-sentence, is not good enough. Text marked "[trimmed by the grader]" was cut by us to save space, not by the model, so don't count it against the answer.
Then say which answer is better overall, or tie.

Reply with only this JSON, nothing else:
{{"a_good": "yes" or "no", "b_good": "yes" or "no", "better": "a" or "b" or "tie"}}"""


def trim(text, limit):
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n[trimmed by the grader]"


def a_role_for(conversation_hash):
    # random a/b per prompt to cancel the judge's position bias. a different salt from the
    # labeling page, so the judge and i don't always see the same model in the same slot.
    return "small" if int(hashlib.md5((conversation_hash + "judge").encode()).hexdigest(), 16) % 2 == 0 else "large"


def build_prompt(prompt, small_answer, large_answer, a_role):
    a, b = (small_answer, large_answer) if a_role == "small" else (large_answer, small_answer)
    return JUDGE_TEMPLATE.format(prompt=trim(prompt, PROMPT_CHARS), a=trim(a, ANSWER_CHARS), b=trim(b, ANSWER_CHARS))


def parse_verdict(raw):
    # small models wrap json in ```json fences or echo code first, so try each {...} until one fits
    for match in re.finditer(r"\{.*?\}", raw or "", flags=re.DOTALL):
        try:
            verdict = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        ok = (isinstance(verdict, dict) and verdict.get("a_good") in ("yes", "no")
              and verdict.get("b_good") in ("yes", "no") and verdict.get("better") in ("a", "b", "tie"))
        if ok:
            return verdict
    return None


def to_roles(verdict, a_role):
    # a/b back to small/large
    b_role = "large" if a_role == "small" else "small"
    good = {a_role: verdict["a_good"] == "yes", b_role: verdict["b_good"] == "yes"}
    better = {"a": a_role, "b": b_role, "tie": "tie"}[verdict["better"]]
    return good["small"], good["large"], better


def labeled_hashes(path=LABEL_SET_PATH):
    with open(path, newline="") as f:
        return {row["conversation_hash"] for row in csv.DictReader(f)}


def load_pairs(con):
    # only prompts where both models answered without an api error. cut-off "" answers stay in.
    return con.execute("""
        select e.conversation_hash, e.prompt, s.answer, l.answer
        from eval_set e
        join answers s on s.conversation_hash = e.conversation_hash and s.role = 'small' and s.error is null
        join answers l on l.conversation_hash = e.conversation_hash and l.role = 'large' and l.error is null
        order by e.conversation_hash
    """).fetchall()


def write_judgments(con, pairs, cache, model):
    records = []
    for conversation_hash, prompt, small_answer, large_answer in pairs:
        a_role = a_role_for(conversation_hash)
        record = cache.get(cache_key(model, build_prompt(prompt, small_answer, large_answer, a_role)))
        if record is None or record.get("error"):
            continue
        verdict = parse_verdict(strip_thoughts(record.get("answer")))
        if verdict is None:
            records.append((conversation_hash, model, a_role, None, None, None, True))
        else:
            records.append((conversation_hash, model, a_role, *to_roles(verdict, a_role), False))
    con.execute("""
        create or replace table judgments (
            conversation_hash varchar, judge_model varchar, a_role varchar,
            small_good boolean, large_good boolean, better varchar, parse_failed boolean
        )
    """)
    if records:
        con.executemany("insert into judgments values (?, ?, ?, ?, ?, ?, ?)", records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    # judge my 200 hand-labeled pairs first, check kappa, and only then spend ~35h on the rest
    parser.add_argument("--labeled-only", action="store_true")
    args = parser.parse_args()
    con = connect_db(DB_PATH, read_only=True)
    pairs = load_pairs(con)
    con.close()
    if args.labeled_only:
        if not os.path.exists(LABEL_SET_PATH):
            raise SystemExit("no label set yet, run python -m pipeline.label_set first")
        wanted = labeled_hashes()
        pairs = [p for p in pairs if p[0] in wanted]
    rows = [(h, build_prompt(p, s, l, a_role_for(h))) for h, p, s, l in pairs]
    cache, model = run_calls("judge", rows, args.limit)
    con = connect_db(DB_PATH)
    write_judgments(con, pairs, cache, model)
    judged = con.execute("select count(*) from judgments").fetchone()[0]
    # long prompts error more often, so say how many got skipped, the kappa set could lose hard cases
    print(f"{len(pairs)} pairs, {judged} judged, {len(pairs) - judged} errored or not run yet")
    print(con.execute("""
        select count(*) judged, count(*) filter (where parse_failed) parse_failed,
               avg(small_good::int) small_good_rate, avg(large_good::int) large_good_rate
        from judgments
    """).df())
    con.close()


if __name__ == "__main__":
    main()
