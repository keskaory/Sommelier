# draws the 4k prompts that actually get answered. reads `prompts`, writes `eval_set`.
import duckdb

DB_PATH = "data/sommelier.duckdb"
# test size from analysis/power.py: ~80% power at a 1 point loss if ~10% of answers disagree (a guess).
# val just picks a threshold and train fits a small logistic regression, 1k each is enough.
TARGETS = {"train": 1000, "val": 1000, "test": 2000}


def draw_eval_set(con, targets):
    # each segment keeps its real share inside each split, so cost numbers match real traffic.
    # rounding can leave a split a prompt or two off.
    con.execute(f"""
        create or replace table eval_set as
        with ranked as (
            select
                *,
                row_number() over (partition by split, segment order by md5(conversation_hash)) as rank_in_segment,
                count(*) over (partition by split, segment) as segment_size,
                count(*) over (partition by split) as split_size
            from prompts
        )
        select conversation_hash, hashed_ip, split, segment, prompt
        from ranked
        where rank_in_segment <= round(
            segment_size * (case split
                when 'train' then {int(targets["train"])}
                when 'val' then {int(targets["val"])}
                else {int(targets["test"])}
            end) / split_size
        )
    """)


def main():
    con = duckdb.connect(DB_PATH)
    draw_eval_set(con, TARGETS)
    print(con.execute("""
        pivot (select split, segment from eval_set) on split using count(*) group by segment order by segment
    """).df())
    print(con.execute("select split, count(*) n from eval_set group by 1 order by 1").df())
    con.close()


if __name__ == "__main__":
    main()
