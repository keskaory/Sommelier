# gets one answer per eval prompt from the small or large model, caches it, fills the `answers` table.
# usage: python -m pipeline.generate small (or large). --limit 20 for a quick try.
import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import duckdb
from dotenv import load_dotenv

DB_PATH = "data/sommelier.duckdb"
CACHE_DIR = "data/cache"
# shortcut: answers stop at 1024 tokens to stay inside free-tier budgets, same cap for both models.
# 512 cut off 2 of the first 3 test answers, way too many.
MAX_TOKENS = 1024
MAX_RETRIES = 5


class StopRun(Exception):
    # bad key, daily cap or server down. no point going on
    pass


def model_config(role):
    load_dotenv()
    prefix = role.upper()
    return {
        "base_url": os.environ[f"{prefix}_BASE_URL"],
        "model": os.environ[f"{prefix}_MODEL"],
        "api_key": os.environ.get(f"{prefix}_API_KEY", "none"),
        "rpm": int(os.environ.get(f"{prefix}_RPM", "20")),
    }


def cache_key(model, prompt):
    # model + prompt, so the same call never happens twice
    return hashlib.md5(f"{model}\n{prompt}".encode()).hexdigest()


def load_cache(path):
    cache = {}
    if not os.path.exists(path):
        return cache
    line = ""
    with open(path) as f:
        for line in f:
            # ctrl-c can leave half a line. skip it, that call gets redone
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping a broken line in {path}")
                continue
            cache[record["key"]] = record
    # close off the broken line or the next record gets stuck onto it
    if line and not line.endswith("\n"):
        with open(path, "a") as f:
            f.write("\n")
    return cache


def append_cache(path, record):
    # one json per line. a crash only loses the call in flight
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def call_chat(config, prompt):
    body = json.dumps({
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }).encode()
    request = urllib.request.Request(
        config["base_url"].rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config['api_key']}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}  # parse_response turns this into a "bad response" error


def wait_seconds(error, attempt):
    # use retry-after if the server sends it (groq does), else wait 1, 2, 4, 8 min, max 10
    retry_after = error.headers.get("Retry-After") if isinstance(error, urllib.error.HTTPError) else None
    if retry_after is not None and retry_after.isdigit():
        return min(int(retry_after), 600)
    return min(60 * 2 ** attempt, 600)


def answer_one(config, prompt, call):
    # 429, 5xx, network errors: retry.
    # 401/403/404: key or model id is wrong. stop before every prompt gets that error cached.
    # other 4xx (like prompt too long): retrying won't help, cache it as an error.
    for attempt in range(MAX_RETRIES):
        try:
            data = call(config, prompt)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise StopRun(f"http {e.code}: check the api key and model id in .env")
            if e.code != 429 and e.code < 500:
                return {"answer": None, "error": f"http {e.code}"}
            error = e
        except OSError as e:  # dropped connection, timeout, refused
            error = e
        else:
            return parse_response(data)
        if attempt < MAX_RETRIES - 1:
            time.sleep(wait_seconds(error, attempt))
    raise StopRun("still rate limited or the server is down (likely the daily cap)")


def parse_response(data):
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        return {"answer": None, "error": "bad response"}
    usage = data.get("usage") or {}
    return {
        "answer": message.get("content"),
        "finish_reason": choice.get("finish_reason"),
        "model_reported": data.get("model"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "error": None,
    }


def generate(con, role, limit=None, call=call_chat, cache_dir=CACHE_DIR):
    config = model_config(role)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{role}.jsonl")
    cache = load_cache(cache_path)
    rows = con.execute("select conversation_hash, prompt from eval_set order by conversation_hash").fetchall()
    todo = [(h, p) for h, p in rows if cache_key(config["model"], p) not in cache]
    print(f"{role}: {len(rows) - len(todo)} cached, {len(todo)} to go ({config['model']})")
    if limit is not None:
        todo = todo[:limit]

    for i, (conversation_hash, prompt) in enumerate(todo):
        try:
            result = answer_one(config, prompt, call)
        except StopRun as e:
            print(f"stopped after {i} calls: {e}. rerun later, the cache picks up where this left off.")
            break
        key = cache_key(config["model"], prompt)
        record = {"key": key, "conversation_hash": conversation_hash, "model": config["model"], **result}
        append_cache(cache_path, record)
        cache[key] = record
        time.sleep(60 / config["rpm"])  # stay under the requests-per-minute cap

    write_answers(con, role, config["model"], rows, cache)


def write_answers(con, role, model, rows, cache):
    # error rows have answer = null, filter on `error is null` later
    records = []
    for conversation_hash, prompt in rows:
        record = cache.get(cache_key(model, prompt))
        if record is not None:
            records.append((conversation_hash, role, model, record.get("model_reported"),
                            record.get("answer"), record.get("finish_reason"),
                            record.get("prompt_tokens"), record.get("completion_tokens"), record.get("error")))
    # not create or replace, that would wipe the other model's answers
    con.execute("""
        create table if not exists answers (
            conversation_hash varchar, role varchar, model varchar, model_reported varchar,
            answer varchar, finish_reason varchar,
            prompt_tokens integer, completion_tokens integer, error varchar
        )
    """)
    con.execute("delete from answers where role = ?", [role])
    if records:
        con.executemany("insert into answers values (?, ?, ?, ?, ?, ?, ?, ?, ?)", records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["small", "large"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    con = duckdb.connect(DB_PATH)
    generate(con, args.role, args.limit)
    print(con.execute("""
        select role, model, count(*) answers, count(error) errors,
               count(*) filter (where finish_reason = 'length') cut_off
        from answers group by 1, 2
    """).df())
    con.close()


if __name__ == "__main__":
    main()
