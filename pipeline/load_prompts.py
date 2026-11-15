# step 1 of the pipeline: turn raw wildchat parquet shards into a clean table of ~20k prompts.
# everything later (segments, eval set, router) reads the `prompts` table this writes.
import duckdb

RAW_GLOB = "data/raw/*.parquet"
DB_PATH = "data/sommelier.duckdb"
SAMPLE_SIZE = 20_000
# chinese, japanese, korean and cyrillic letters. wildchat tags some of these prompts english.
NON_LATIN = r"[\x{4e00}-\x{9fff}\x{3040}-\x{30ff}\x{ac00}-\x{d7af}\x{0400}-\x{04ff}]"


def load_first_prompts(con, parquet_glob):
    # first message only, mvp is single-turn. toxic is already removed upstream,
    # but the filter is cheap insurance. redacted = pii was found, so those go too.
    # the non-latin check drops prompts that are mostly another language but still
    # tagged english. an english prompt quoting a little foreign text stays.
    con.execute(f"""
        create or replace table first_prompts as
        select
            conversation_hash,
            hashed_ip,
            timestamp,
            conversation[1].content as prompt
        from read_parquet('{parquet_glob}')
        where conversation[1].role = 'user'
          and conversation[1].language = 'English'
          and not toxic
          and not redacted
          and trim(conversation[1].content) <> ''
          and length(regexp_replace(conversation[1].content, '{NON_LATIN}', '', 'g'))
              >= 0.8 * length(conversation[1].content)
    """)


def dedupe_prompts(con):
    # two prompts count as duplicates if their first 150 chars match after lowercasing
    # and squashing spaces. that catches copy-paste templates (one midjourney template
    # was 11% of the sample, from 1k+ different users) that only differ at the end.
    # keep the earliest copy so the choice doesn't depend on file order.
    # duplicates left in would land in both train and test and inflate results.
    con.execute("""
        create or replace table unique_prompts as
        select conversation_hash, hashed_ip, timestamp, prompt
        from first_prompts
        qualify row_number() over (
            partition by left(regexp_replace(lower(trim(prompt)), '\\s+', ' ', 'g'), 150)
            order by timestamp, conversation_hash
        ) = 1
    """)


def sample_prompts(con, n):
    # ordering by a hash of the id gives the same "random" sample on every run,
    # unlike duckdb's sample which can change with thread count.
    con.execute(f"""
        create or replace table prompts as
        select conversation_hash, hashed_ip, timestamp, prompt
        from unique_prompts
        order by md5(conversation_hash)
        limit {int(n)}
    """)


def main():
    con = duckdb.connect(DB_PATH)
    load_first_prompts(con, RAW_GLOB)
    dedupe_prompts(con)
    sample_prompts(con, SAMPLE_SIZE)
    for table in ["first_prompts", "unique_prompts", "prompts"]:
        count = con.execute(f"select count(*) from {table}").fetchone()[0]
        print(f"{table}: {count}")
    # the in-between tables were only for counting, no need to keep them on disk
    con.execute("drop table first_prompts")
    con.execute("drop table unique_prompts")
    con.close()


if __name__ == "__main__":
    main()
