# split.py: one split per user, roughly 40/20/40
import duckdb

from pipeline.split import add_splits, split_for


def test_same_user_same_split_every_time():
    assert split_for("ip_abc") == split_for("ip_abc")


def test_proportions_are_roughly_40_20_40():
    splits = [split_for(f"user_{i}") for i in range(10_000)]
    assert abs(splits.count("train") / 10_000 - 0.40) < 0.02
    assert abs(splits.count("val") / 10_000 - 0.20) < 0.02
    assert abs(splits.count("test") / 10_000 - 0.40) < 0.02


def test_no_user_in_two_splits():
    con = duckdb.connect()
    con.execute("""
        create table prompts as
        select 'c' || i as conversation_hash, 'ip' || (i % 50) as hashed_ip,
               timestamp '2023-05-01' as timestamp, 'p' as prompt, 'other' as segment
        from range(500) t(i)
    """)
    add_splits(con)
    add_splits(con)  # rerun has to work now that the column exists
    users_in_many = con.execute("""
        select count(*) from (
            select hashed_ip from prompts group by 1 having count(distinct split) > 1
        )
    """).fetchone()[0]
    assert users_in_many == 0
    assert con.execute("select count(*) from prompts").fetchone()[0] == 500
