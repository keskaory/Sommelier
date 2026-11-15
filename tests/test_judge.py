# judge.py: verdicts parse even when wrapped in junk, a/b maps back to the right model, long text gets trimmed.
import duckdb

from pipeline.generate import cache_key
from pipeline.judge import (ANSWER_CHARS, PROMPT_CHARS, a_role_for, build_prompt, parse_verdict, to_roles, trim,
                            write_judgments)


def test_parse_verdict_handles_fences_and_rejects_junk():
    raw = 'sure!\n```json\n{"a_good": "yes", "b_good": "no", "better": "a"}\n```'
    assert parse_verdict(raw) == {"a_good": "yes", "b_good": "no", "better": "a"}
    assert parse_verdict('{"a_good": "maybe", "b_good": "no", "better": "a"}') is None
    assert parse_verdict("no json here") is None
    assert parse_verdict(None) is None
    echoed = 'here is code: {x: 1} and the verdict {"a_good": "no", "b_good": "yes", "better": "b"}'
    assert parse_verdict(echoed) == {"a_good": "no", "b_good": "yes", "better": "b"}


def test_to_roles_maps_a_and_b_back():
    verdict = {"a_good": "yes", "b_good": "no", "better": "b"}
    assert to_roles(verdict, "small") == (True, False, "large")
    assert to_roles(verdict, "large") == (False, True, "small")
    assert to_roles({"a_good": "no", "b_good": "no", "better": "tie"}, "small") == (False, False, "tie")


def test_build_prompt_puts_the_right_answer_in_a_and_trims():
    long_answer = "x" * (ANSWER_CHARS + 500)
    text = build_prompt("q", "small says hi", long_answer, "small")
    assert text.index("small says hi") < text.index("ANSWER B")
    assert "[trimmed by the grader]" in text
    assert len(build_prompt("q" * (PROMPT_CHARS + 500), "a", "b", "small")) < PROMPT_CHARS + 2000
    assert trim("short", 10) == "short"


def test_a_role_is_stable_and_mixed():
    assert a_role_for("h1") == a_role_for("h1")
    assert {a_role_for(f"h{i}") for i in range(50)} == {"small", "large"}


def test_write_judgments_maps_and_flags_parse_failures():
    con = duckdb.connect()
    pairs = [("h1", "q1", "s1", "l1"), ("h2", "q2", "s2", "l2"), ("h3", "q3", "s3", "l3")]
    model = "judge-m"
    cache = {}
    for h, raw, error in [("h1", '{"a_good": "yes", "b_good": "yes", "better": "tie"}', None),
                          ("h2", "garbage", None), ("h3", None, "http 400")]:
        p = next(x for x in pairs if x[0] == h)
        key = cache_key(model, build_prompt(p[1], p[2], p[3], a_role_for(h)))
        cache[key] = {"answer": raw, "error": error}
    write_judgments(con, pairs, cache, model)
    rows = con.execute("select conversation_hash, small_good, large_good, better, parse_failed from judgments order by 1").fetchall()
    # h3 errored, so it's skipped (main prints how many)
    assert rows == [("h1", True, True, "tie", False), ("h2", None, None, None, True)]
