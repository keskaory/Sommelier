# load_prompts.py on a tiny fake wildchat file
import duckdb
import pandas as pd
import pytest

from pipeline.load_prompts import dedupe_prompts, load_first_prompts, sample_prompts


TEMPLATE = "as a prompt generator for an image ai, follow these rules exactly. " * 3


def make_row(conv_id, content, language="English", role="user", toxic=False, redacted=False, ts="2023-05-01"):
    turn = {"content": content, "language": language, "role": role}
    return {
        "conversation_hash": conv_id,
        "hashed_ip": "ip_" + conv_id,
        "timestamp": pd.Timestamp(ts),
        "conversation": [turn, {"content": "answer", "language": "English", "role": "assistant"}],
        "toxic": toxic,
        "redacted": redacted,
    }


@pytest.fixture
def con(tmp_path):
    rows = [
        make_row("a", "how do i sort a list in python?", ts="2023-05-02"),
        make_row("b", "How do I  sort a list in Python?", ts="2023-05-01"),  # dupe of a, but earlier
        make_row("c", "bonjour", language="French"),
        make_row("d", "bad stuff", toxic=True),
        make_row("e", "my name is [NAME]", redacted=True),
        make_row("f", "   "),
        make_row("g", "write a poem about rain"),
        make_row("h", "assistant spoke first", role="assistant"),
        make_row("i", "润色如下内容：我们的报告总结了很多数据", ts="2023-05-03"),  # mostly chinese, tagged english
        make_row("j", "what does 你好 mean?"),  # english with a little chinese, should stay
        make_row("m", "لطفا متن زیر را بازنویسی کن"),  # persian, tagged english
        make_row("k", TEMPLATE + "a cat on the moon", ts="2023-05-04"),
        make_row("l", TEMPLATE + "a dog in the rain", ts="2023-05-05"),  # same template, later
    ]
    path = tmp_path / "fake.parquet"
    pd.DataFrame(rows).to_parquet(path)
    connection = duckdb.connect()
    load_first_prompts(connection, str(path))
    yield connection
    connection.close()


def ids(con, table):
    return sorted(r[0] for r in con.execute(f"select conversation_hash from {table}").fetchall())


def test_filters_drop_non_english_flagged_blank_and_non_user(con):
    assert ids(con, "first_prompts") == ["a", "b", "g", "j", "k", "l"]


def test_dedupe_keeps_earliest_copy(con):
    dedupe_prompts(con)
    assert ids(con, "unique_prompts") == ["b", "g", "j", "k"]


def test_sample_is_capped_and_repeatable(con):
    dedupe_prompts(con)
    sample_prompts(con, 1)
    first = ids(con, "prompts")
    sample_prompts(con, 1)
    assert len(first) == 1
    assert ids(con, "prompts") == first
