# turns the raw wildchat shards into a `prompts` table of 20k english prompts. everything after reads it.
import duckdb

RAW_GLOB = "data/raw/*.parquet"
DB_PATH = "data/sommelier.duckdb"
SAMPLE_SIZE = 20_000
# cjk, cyrillic and arabic letters. wildchat tags some of these prompts as english
NON_LATIN = r"[\x{4e00}-\x{9fff}\x{3040}-\x{30ff}\x{ac00}-\x{d7af}\x{0400}-\x{04ff}\x{0600}-\x{06ff}]"


def load_first_prompts(con, parquet_glob):
    # first message only. toxic rows are already gone upstream, the filter's just in case.
    # redacted = pii was found in it, so those go too.
    # last check drops prompts that are mostly another language but tagged english.
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
    # dupes = same first 150 chars after lowercasing and squashing spaces. exact match missed
    # copy-paste templates, one midjourney template alone was 11% of the sample.
    # keep the earliest copy. dupes left in end up in both train and test.
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
    # md5 order gives the same sample every run. duckdb's sample() can change with thread count
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
    # the middle tables were only for the counts above
    con.execute("drop table first_prompts")
    con.execute("drop table unique_prompts")
    con.close()


if __name__ == "__main__":
    main()
