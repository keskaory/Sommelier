# step 3 of the pipeline (first half, the eval-set draw comes later): put every prompt in train, val or test, by user.
# reads `prompts` (steps 1-2) and adds a `split` column. router trains on train, the
# threshold is picked on val, and the reported numbers only ever come from test.
import hashlib

import duckdb

DB_PATH = "data/sommelier.duckdb"


def split_for(hashed_ip):
    # whole users go to one split, so a heavy user's similar prompts can't leak from
    # train into test. md5 instead of python's hash() because hash() changes every run.
    # 40/20/40: test needs ~2k labeled prompts (see analysis/power.py), val ~1k.
    bucket = int(hashlib.md5(hashed_ip.encode()).hexdigest(), 16) % 100
    if bucket < 40:
        return "train"
    if bucket < 60:
        return "val"
    return "test"


def add_splits(con):
    df = con.execute("select conversation_hash, hashed_ip from prompts").df()
    # a prompt with no user can't be split by user, fail loudly instead of guessing
    assert df["hashed_ip"].notna().all(), "found prompts with no hashed_ip"
    df["split"] = df["hashed_ip"].apply(split_for)
    # rebuild instead of altering so a rerun just overwrites
    con.execute("""
        create or replace table prompts as
        select p.conversation_hash, p.hashed_ip, p.timestamp, p.prompt, p.segment, df.split
        from prompts p
        join df using (conversation_hash)
    """)


def main():
    con = duckdb.connect(DB_PATH)
    add_splits(con)
    print(con.execute("""
        select split, count(*) prompts, count(distinct hashed_ip) users
        from prompts group by 1 order by 1
    """).df())
    con.close()


if __name__ == "__main__":
    main()
