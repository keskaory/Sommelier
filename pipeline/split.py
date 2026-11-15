# puts each prompt in train, val or test by user. train fits the router, val picks the
# threshold, test is only for the final numbers.
import hashlib

import duckdb

DB_PATH = "data/sommelier.duckdb"


def split_for(hashed_ip):
    # all of a user's prompts go to one split so their lookalike prompts can't leak into test.
    # md5 not hash(), hash() changes every run. 40/20/40 because test needs ~2k.
    bucket = int(hashlib.md5(hashed_ip.encode()).hexdigest(), 16) % 100
    if bucket < 40:
        return "train"
    if bucket < 60:
        return "val"
    return "test"


def add_splits(con):
    df = con.execute("select conversation_hash, hashed_ip from prompts").df()
    # can't split by user without a user
    assert df["hashed_ip"].notna().all(), "found prompts with no hashed_ip"
    df["split"] = df["hashed_ip"].apply(split_for)
    # rebuild the table so rerunning just overwrites it
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
