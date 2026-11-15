# step 3 of the pipeline (second half): draw the prompts we'll actually get answers for.
# reads `prompts` (steps 1-3a) and writes `eval_set`. step 4 sends these to both models.
import duckdb

DB_PATH = "data/sommelier.duckdb"
# test size comes from analysis/power.py: ~2k gives ~80% power if the true loss is 1 point
# and the routed and large answers disagree on ~10% of prompts (a guess, rerun once we know).
# val only picks a threshold and train only fits a small logistic regression, so 1k each.
TARGETS = {"train": 1000, "val": 1000, "test": 2000}


def draw_eval_set(con, targets):
    # proportional stratified sample: inside each split, every segment keeps its real share,
    # so the cost estimate still reflects real traffic. md5 order = same draw every run.
    # rounding per segment means a split can come out a prompt or two off its target.
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
