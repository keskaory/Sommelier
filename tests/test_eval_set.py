# eval_set.py: sizes, segment shares, same draw every run
import duckdb
import pytest

from pipeline.eval_set import draw_eval_set


@pytest.fixture
def con():
    # 1,000 test prompts: 70% writing, 30% coding. 500 train prompts, all q&a.
    connection = duckdb.connect()
    connection.execute("""
        create table prompts as
        select
            'c' || i as conversation_hash,
            'ip' || i as hashed_ip,
            case when i < 1000 then 'test' else 'train' end as split,
            case when i < 700 then 'writing' when i < 1000 then 'coding' else 'q&a' end as segment,
            'p' || i as prompt
        from range(1500) t(i)
    """)
    yield connection
    connection.close()


def counts(con):
    return dict(con.execute("select split || '/' || segment, count(*) from eval_set group by 1").fetchall())


def test_sizes_and_shares(con):
    draw_eval_set(con, {"train": 100, "val": 0, "test": 200})
    assert counts(con) == {"test/writing": 140, "test/coding": 60, "train/q&a": 100}


def test_same_draw_every_time(con):
    draw_eval_set(con, {"train": 100, "val": 0, "test": 200})
    first = con.execute("select conversation_hash from eval_set order by 1").fetchall()
    draw_eval_set(con, {"train": 100, "val": 0, "test": 200})
    assert con.execute("select conversation_hash from eval_set order by 1").fetchall() == first


def test_uneven_shares_stay_close_to_target():
    # 333/333/334 drawing 100 can't split evenly, so the total may be off by up to one per segment
    con = duckdb.connect()
    con.execute("""
        create table prompts as
        select 'c' || i as conversation_hash, 'ip' || i as hashed_ip, 'test' as split,
               case when i < 333 then 'a' when i < 666 then 'b' else 'c' end as segment, 'p' as prompt
        from range(1000) t(i)
    """)
    draw_eval_set(con, {"train": 0, "val": 0, "test": 100})
    assert abs(con.execute("select count(*) from eval_set").fetchone()[0] - 100) <= 3
