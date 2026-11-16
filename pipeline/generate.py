# gets one answer per eval prompt from the small or large model, caches it, fills the `answers` table.
# usage: python -m pipeline.generate small (or large). --limit 20 for a quick try.
import argparse
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import certifi
import duckdb
from dotenv import load_dotenv

DB_PATH = "data/sommelier.duckdb"
CACHE_DIR = "data/cache"
# gemma 4's thinking counts against this. at 1024 all 9 trial answers got cut off, at 4096 none did
MAX_TOKENS = 4096
MAX_RETRIES = 5


class StopRun(Exception):
    # bad key, daily cap or server down. no point going on
    pass


class SkipPrompt(Exception):
    # google 500s the same prompt over and over. skip it without caching so the next run tries again
    pass


def model_config(role):
    load_dotenv()
    prefix = role.upper()
    return {
        "base_url": os.environ[f"{prefix}_BASE_URL"],
        "model": os.environ[f"{prefix}_MODEL"],
        "api_key": os.environ.get(f"{prefix}_API_KEY", "none"),
        "rpm": int(os.environ.get(f"{prefix}_RPM", "20")),
        "workers": int(os.environ.get(f"{prefix}_WORKERS", "1")),
    }


def cache_key(model, prompt):
    # model + token cap + prompt, so the same call never happens twice,
    # and changing the cap doesn't reuse answers that got cut off under the old one
    return hashlib.md5(f"{model}\n{MAX_TOKENS}\n{prompt}".encode()).hexdigest()


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
    # python.org's mac python ships with no trusted certs, so every https call fails without this
    context = ssl.create_default_context(cafile=certifi.where())
    # calls took up to 260s in testing. a shorter timeout just retries and pays for the tokens again
    with urllib.request.urlopen(request, timeout=360, context=context) as response:
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
            # google's 429 body names the quota it hit (per minute vs per day), keep it for the stop message
            last_seen = f"http {e.code}: {e.read()[:300].decode(errors='replace')}"
        except OSError as e:  # dropped connection, timeout, refused
            error = e
            last_seen = repr(e)
        else:
            return parse_response(data)
        if attempt < MAX_RETRIES - 1:
            time.sleep(wait_seconds(error, attempt))
    if isinstance(error, urllib.error.HTTPError) and error.code >= 500:
        raise SkipPrompt(last_seen)
    raise StopRun(f"still rate limited or the server is down. last error: {last_seen}")


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


def load_rows(con):
    return con.execute("select conversation_hash, prompt from eval_set order by conversation_hash").fetchall()


def run_calls(role, rows, limit=None, call=call_chat, cache_dir=CACHE_DIR):
    # returns the cache after making every call that isn't cached yet
    config = model_config(role)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{role}.jsonl")
    cache = load_cache(cache_path)
    todo = [(h, p) for h, p in rows if cache_key(config["model"], p) not in cache]
    print(f"{role}: {len(rows) - len(todo)} cached, {len(todo)} to go ({config['model']})")
    if limit is not None:
        todo = todo[:limit]

    # one call takes 0.5-4 min but the api lets several run at once. so start a new one
    # every 60/rpm seconds and keep up to `workers` in flight. only this thread writes the cache.
    # ctrl-c waits for the calls in flight and those answers aren't saved, that's fine
    stop_reason = None
    skipped = 0
    running = {}
    next_todo = iter(todo)
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        while True:
            if stop_reason is None and len(running) < config["workers"]:
                item = next(next_todo, None)
                if item is not None:
                    running[pool.submit(answer_one, config, item[1], call)] = item
                    time.sleep(60 / config["rpm"])
                    continue
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                conversation_hash, prompt = running.pop(future)
                try:
                    result = future.result()
                except StopRun as e:
                    stop_reason = str(e)  # let the calls already running finish, start no more
                    continue
                except SkipPrompt:
                    skipped += 1
                    continue
                key = cache_key(config["model"], prompt)
                record = {"key": key, "conversation_hash": conversation_hash, "model": config["model"], **result}
                append_cache(cache_path, record)
                cache[key] = record
    if skipped:
        print(f"skipped {skipped} prompts that kept getting 500s, rerun to retry them")
    if stop_reason is not None:
        print(f"stopped: {stop_reason}. rerun later, the cache picks up where this left off.")
    return cache, config["model"]


def generate(con, role, limit=None, call=call_chat, cache_dir=CACHE_DIR):
    rows = load_rows(con)
    cache, model = run_calls(role, rows, limit, call, cache_dir)
    write_answers(con, role, model, rows, cache)


def strip_thoughts(raw):
    # gemma 4 writes its reasoning in <thought>...</thought> before the answer and it can't
    # be turned off. the judge should only see the answer. the cache keeps the raw reply.
    if raw is None:
        return None
    if "<thought>" in raw and "</thought>" not in raw:
        return ""  # cut off mid-thought, there's no answer
    return re.sub(r"<thought>.*?</thought>", "", raw, flags=re.DOTALL).strip()


def write_answers(con, role, model, rows, cache):
    # error rows have answer = null, filter on `error is null` later
    records = []
    for conversation_hash, prompt in rows:
        record = cache.get(cache_key(model, prompt))
        if record is not None:
            records.append((conversation_hash, role, model, record.get("model_reported"),
                            strip_thoughts(record.get("answer")), record.get("finish_reason"),
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


def connect_db(path, read_only=False):
    # small and large run as two processes. duckdb lets only one write at a time, so if the
    # other one is mid-write, wait a bit. every answer is already in the cache either way.
    for _ in range(10):
        try:
            return duckdb.connect(path, read_only=read_only)
        except duckdb.IOException:
            time.sleep(5)
    raise SystemExit("db is busy. rerun, everything is cached so it makes no new calls")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["small", "large"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    # don't hold the db open for hours, so small and large can run at the same time
    con = connect_db(DB_PATH, read_only=True)
    rows = load_rows(con)
    con.close()
    cache, model = run_calls(args.role, rows, args.limit)
    con = connect_db(DB_PATH)
    write_answers(con, args.role, model, rows, cache)
    print(con.execute("""
        select role, model, count(*) answers, count(error) errors,
               count(*) filter (where finish_reason = 'length') cut_off
        from answers group by 1, 2
    """).df())
    con.close()


if __name__ == "__main__":
    main()
