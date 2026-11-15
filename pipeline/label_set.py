# picks the ~200 prompts i hand-label, once, after generation is done. writes data/labels/label_set.csv.
# run: python -m pipeline.label_set. it won't overwrite an existing set, so labels never point at a moving target.
import csv
import hashlib
import os

import duckdb

DB_PATH = "data/sommelier.duckdb"
LABEL_SET_PATH = "data/labels/label_set.csv"
N_LABELS = 200
N_RELABEL = 50  # re-labeled a week later (me-2) and by a friend, for rater agreement


def candidates(con):
    # train + val only, test stays untouched. both models answered without an api error.
    # cut-off answers ("") stay in, they're real failures the judge has to handle.
    return con.execute("""
        select e.conversation_hash, e.segment
        from eval_set e
        join answers s on s.conversation_hash = e.conversation_hash and s.role = 'small' and s.error is null
        join answers l on l.conversation_hash = e.conversation_hash and l.role = 'large' and l.error is null
        where e.split in ('train', 'val')
        order by md5(e.conversation_hash || 'label')
    """).fetchall()


def pick(rows, n=N_LABELS):
    # equal share per segment so the rarer, harder ones (math, coding) aren't drowned out by writing.
    # n // segments keeps the caps from adding up past n, then fill to n from whatever's left.
    per_segment = n // len({segment for _, segment in rows})
    picked = []
    taken = {}
    for conversation_hash, segment in rows:
        if taken.get(segment, 0) < per_segment:
            picked.append((conversation_hash, segment))
            taken[segment] = taken.get(segment, 0) + 1
    chosen = {h for h, _ in picked}
    for conversation_hash, segment in rows:
        if len(picked) >= n:
            break
        if conversation_hash not in chosen:
            picked.append((conversation_hash, segment))
    return picked[:n]


def relabel_hashes(picked, n=N_RELABEL):
    # a different order from round 1, so the re-label isn't just the first 50 i did while still learning
    hashes = sorted((h for h, _ in picked), key=lambda h: hashlib.md5((h + "relabel").encode()).hexdigest())
    return set(hashes[:n])


def freeze(con, path=LABEL_SET_PATH):
    if os.path.exists(path):
        print(f"{path} already exists, not touching it")
        return
    rows = candidates(con)
    if not rows:
        print("no clean answer pairs yet, finish generation first")
        return
    picked = pick(rows)
    relabel = relabel_hashes(picked)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["conversation_hash", "segment", "relabel"])
        for conversation_hash, segment in picked:
            writer.writerow([conversation_hash, segment, int(conversation_hash in relabel)])
    total = con.execute("select count(*) from eval_set where split in ('train', 'val')").fetchone()[0]
    print(f"picked {len(picked)} of {len(rows)} usable train+val prompts ({total - len(rows)} had no clean answer pair)")
    counts = {}
    for _, segment in picked:
        counts[segment] = counts.get(segment, 0) + 1
    print("per segment:", counts)


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    freeze(con)
    con.close()


if __name__ == "__main__":
    main()
