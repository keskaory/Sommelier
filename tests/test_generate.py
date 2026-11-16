# generate.py against a fake api, no network or keys
import io
import json
import urllib.error

import duckdb
import pytest

import pipeline.generate as gen


@pytest.fixture
def con(monkeypatch):
    monkeypatch.setenv("SMALL_BASE_URL", "http://fake")
    monkeypatch.setenv("SMALL_MODEL", "fake-8b")
    monkeypatch.setenv("SMALL_RPM", "60000")  # no real waiting in tests
    monkeypatch.setattr(gen, "load_dotenv", lambda: None)  # don't read the real .env
    monkeypatch.setattr(gen.time, "sleep", lambda seconds: None)
    connection = duckdb.connect()
    connection.execute("""
        create table eval_set as select * from (values ('a', 'hi there'), ('b', 'write a poem'))
        t(conversation_hash, prompt)
    """)
    yield connection
    connection.close()


def fake_ok(config, prompt):
    return {
        "model": config["model"],
        "choices": [{"message": {"content": "answer to " + prompt}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }


def http_error(code):
    return urllib.error.HTTPError("http://fake", code, "err", {}, io.BytesIO(b""))


def test_answers_land_in_table_and_cache(con, tmp_path):
    gen.generate(con, "small", call=fake_ok, cache_dir=str(tmp_path))
    rows = con.execute("select conversation_hash, answer, completion_tokens from answers order by 1").fetchall()
    assert rows == [("a", "answer to hi there", 4), ("b", "answer to write a poem", 4)]
    lines = (tmp_path / "small.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["model"] == "fake-8b"


def test_cached_prompts_are_not_called_again(con, tmp_path):
    gen.generate(con, "small", call=fake_ok, cache_dir=str(tmp_path))

    def must_not_call(config, prompt):
        raise AssertionError("should have used the cache")

    gen.generate(con, "small", call=must_not_call, cache_dir=str(tmp_path))
    assert con.execute("select count(*) from answers").fetchone()[0] == 2


def test_rate_limit_is_retried_then_succeeds(con, tmp_path):
    calls = {"n": 0}

    def flaky(config, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http_error(429)
        return fake_ok(config, prompt)

    gen.generate(con, "small", limit=1, call=flaky, cache_dir=str(tmp_path))
    assert con.execute("select count(*), count(error) from answers").fetchone() == (1, 0)


def test_bad_request_is_recorded_not_retried(con, tmp_path):
    def too_long(config, prompt):
        raise http_error(400)

    gen.generate(con, "small", call=too_long, cache_dir=str(tmp_path))
    assert con.execute("select count(*), count(error) from answers").fetchone() == (2, 2)


def test_daily_cap_stops_cleanly_and_keeps_what_it_has(con, tmp_path):
    def always_limited(config, prompt):
        raise http_error(429)

    gen.generate(con, "small", call=always_limited, cache_dir=str(tmp_path))
    assert con.execute("select count(*) from answers").fetchone()[0] == 0
    assert not (tmp_path / "small.jsonl").exists() or (tmp_path / "small.jsonl").read_text() == ""


def test_model_version_is_kept(con, tmp_path):
    gen.generate(con, "small", limit=1, call=fake_ok, cache_dir=str(tmp_path))
    assert con.execute("select model_reported from answers").fetchone()[0] == "fake-8b"


def test_bad_key_stops_the_run_and_caches_nothing(con, tmp_path):
    def unauthorized(config, prompt):
        raise http_error(401)

    gen.generate(con, "small", call=unauthorized, cache_dir=str(tmp_path))
    assert con.execute("select count(*) from answers").fetchone()[0] == 0
    gen.generate(con, "small", call=fake_ok, cache_dir=str(tmp_path))  # fixed key, nothing was cached as done
    assert con.execute("select count(*), count(error) from answers").fetchone() == (2, 0)


def test_broken_last_cache_line_is_skipped(con, tmp_path):
    gen.generate(con, "small", limit=1, call=fake_ok, cache_dir=str(tmp_path))
    with open(tmp_path / "small.jsonl", "a") as f:
        f.write('{"key": "half a li')  # what a ctrl-c mid-write looks like
    gen.generate(con, "small", call=fake_ok, cache_dir=str(tmp_path))
    assert con.execute("select count(*), count(error) from answers").fetchone() == (2, 0)

    def must_not_call(config, prompt):
        raise AssertionError("both answers should be cached by now")

    gen.generate(con, "small", call=must_not_call, cache_dir=str(tmp_path))


def test_network_error_is_retried(con, tmp_path):
    calls = {"n": 0}

    def flaky(config, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("connection refused")
        return fake_ok(config, prompt)

    gen.generate(con, "small", limit=1, call=flaky, cache_dir=str(tmp_path))
    assert con.execute("select count(*), count(error) from answers").fetchone() == (1, 0)


def test_retry_after_header_is_used():
    error = urllib.error.HTTPError("http://fake", 429, "slow down", {"Retry-After": "7"}, io.BytesIO(b""))
    assert gen.wait_seconds(error, attempt=3) == 7
    assert gen.wait_seconds(urllib.error.URLError("x"), attempt=1) == 120


def test_garbled_response_is_cached_as_bad_response(con, tmp_path):
    gen.generate(con, "small", limit=1, call=lambda config, prompt: {}, cache_dir=str(tmp_path))
    assert con.execute("select error from answers").fetchone()[0] == "bad response"


def test_huge_retry_after_is_capped():
    error = urllib.error.HTTPError("http://fake", 429, "daily cap", {"Retry-After": "86400"}, io.BytesIO(b""))
    assert gen.wait_seconds(error, attempt=0) == 600


def test_thoughts_are_stripped_from_the_answer():
    assert gen.strip_thoughts("<thought>* topic: x</thought>\n\nA hash table maps keys.") == "A hash table maps keys."
    assert gen.strip_thoughts("no thinking here") == "no thinking here"
    assert gen.strip_thoughts("<thought>still going when it got cut off") == ""
    assert gen.strip_thoughts(None) is None


def test_parallel_calls_answer_every_prompt_once(con, tmp_path, monkeypatch):
    monkeypatch.setenv("SMALL_WORKERS", "3")
    con.execute("insert into eval_set select 'x' || i, 'prompt ' || i from range(10) t(i)")
    seen = []

    def counting(config, prompt):
        seen.append(prompt)
        return fake_ok(config, prompt)

    gen.generate(con, "small", call=counting, cache_dir=str(tmp_path))
    # 12 = the fixture's 2 prompts + the 10 added here
    assert sorted(seen) == sorted(set(seen)) and len(seen) == 12
    assert con.execute("select count(*), count(distinct conversation_hash) from answers").fetchone() == (12, 12)
    assert len((tmp_path / "small.jsonl").read_text().splitlines()) == 12


def test_stop_in_one_worker_stops_new_calls(con, tmp_path, monkeypatch):
    monkeypatch.setenv("SMALL_WORKERS", "2")
    con.execute("insert into eval_set select 'x' || i, 'prompt ' || i from range(20) t(i)")
    calls = {"n": 0}

    def bad_key(config, prompt):
        calls["n"] += 1
        raise http_error(401)

    gen.generate(con, "small", call=bad_key, cache_dir=str(tmp_path))
    assert calls["n"] == 2  # just the 2 already in flight, not all 22


def test_changing_the_token_cap_changes_the_cache_key(monkeypatch):
    before = gen.cache_key("m", "p")
    monkeypatch.setattr(gen, "MAX_TOKENS", 1024)
    assert gen.cache_key("m", "p") != before


def test_prompt_that_keeps_500ing_is_skipped_not_cached_and_run_goes_on(con, tmp_path):
    def broken_on_poem(config, prompt):
        if prompt == "write a poem":
            raise http_error(500)
        return fake_ok(config, prompt)

    gen.generate(con, "small", call=broken_on_poem, cache_dir=str(tmp_path))
    cached = [json.loads(line)["conversation_hash"] for line in open(tmp_path / "small.jsonl")]
    assert cached == ["a"]  # b not cached, so the next run tries it again
