# the page i use to hand-label answer pairs. the judge gets checked against these labels.
# run python -m pipeline.label_set once first, then: streamlit run app/label.py
# answers show up as a and b with the model names hidden, so i can't favor the big one.
import csv
import hashlib
import os
from datetime import datetime

import duckdb
import streamlit as st

DB_PATH = "data/sommelier.duckdb"
LABEL_SET_PATH = "data/labels/label_set.csv"
LABELS_PATH = "data/labels/human.csv"
FIELDS = ["conversation_hash", "labeler", "a_role", "a_good", "b_good", "better", "labeled_at"]
# me = first pass on all 200. me-2 = same person a week later, friend = second person, both on the 50 relabel ones
LABELERS = ["me", "me-2", "friend"]


def read_label_set(labeler, path=LABEL_SET_PATH):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if labeler == "me":
        return [r["conversation_hash"] for r in rows]
    relabel = [r["conversation_hash"] for r in rows if r["relabel"] == "1"]
    return sorted(relabel, key=lambda h: hashlib.md5((h + "relabel").encode()).hexdigest())


def fetch_item(con, conversation_hash):
    return con.execute("""
        select e.segment, e.prompt, s.answer, l.answer
        from eval_set e
        join answers s on s.conversation_hash = e.conversation_hash and s.role = 'small'
        join answers l on l.conversation_hash = e.conversation_hash and l.role = 'large'
        where e.conversation_hash = ?
    """, [conversation_hash]).fetchone()


def a_role_for(conversation_hash):
    # coin flip per prompt that comes out the same every time, so a is the small model about half the time
    return "small" if int(hashlib.md5(conversation_hash.encode()).hexdigest(), 16) % 2 == 0 else "large"


def better_role(label):
    # labels say a/b/tie, analysis needs small/large/tie
    if label["better"] == "tie":
        return "tie"
    other = "large" if label["a_role"] == "small" else "small"
    return label["a_role"] if label["better"] == "a" else other


def load_labels(path=LABELS_PATH):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save_label(row, path=LABELS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def next_unlabeled(hashes, labels, labeler):
    done = {r["conversation_hash"] for r in labels if r["labeler"] == labeler}
    for conversation_hash in hashes:
        if conversation_hash not in done:
            return conversation_hash
    return None


def main():
    st.title("label answers")
    if not os.path.exists(LABEL_SET_PATH):
        st.warning("no label set yet. run `python -m pipeline.label_set` once generation is done.")
        return
    labeler = st.selectbox("labeler", LABELERS)
    hashes = read_label_set(labeler)
    labels = load_labels()
    done = len({r["conversation_hash"] for r in labels if r["labeler"] == labeler})
    st.progress(min(done / len(hashes), 1.0), text=f"{done} / {len(hashes)}")

    conversation_hash = next_unlabeled(hashes, labels, labeler)
    if conversation_hash is None:
        st.success("all done")
        return
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
    except duckdb.IOException:
        st.error("generation is writing to the db right now, refresh in a minute")
        return
    segment, prompt, small_answer, large_answer = fetch_item(con, conversation_hash)
    con.close()
    a_role = a_role_for(conversation_hash)
    a_answer, b_answer = (small_answer, large_answer) if a_role == "small" else (large_answer, small_answer)

    st.caption(f"segment: {segment}")
    # same bar the judge gets in pipeline/judge.py, so we grade against the same thing
    st.caption("good enough = does what was asked, is correct, and a typical user wouldn't need to ask again. "
               "length and style don't matter unless asked for. empty or stopped mid-sentence = no.")
    with st.expander("prompt", expanded=True):
        st.text(prompt)
    left, right = st.columns(2)
    with left:
        st.subheader("a")
        st.markdown(a_answer or "*(empty, got cut off)*")
    with right:
        st.subheader("b")
        st.markdown(b_answer or "*(empty, got cut off)*")

    with st.form(key=f"{labeler}-{conversation_hash}"):
        a_good = st.radio("is a good enough?", ["yes", "no"], index=None, horizontal=True)
        b_good = st.radio("is b good enough?", ["yes", "no"], index=None, horizontal=True)
        better = st.radio("which is better?", ["a", "b", "tie"], index=None, horizontal=True)
        if st.form_submit_button("save"):
            if None in (a_good, b_good, better):
                st.error("answer all three")
            else:
                save_label({
                    "conversation_hash": conversation_hash, "labeler": labeler, "a_role": a_role,
                    "a_good": a_good, "b_good": b_good, "better": better,
                    "labeled_at": datetime.now().isoformat(timespec="seconds"),
                })
                st.rerun()


if __name__ == "__main__":
    main()
