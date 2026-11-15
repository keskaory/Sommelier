# label_set.py and app/label.py: the set is balanced and frozen, relabel is 50 in its own order, labels decode right.
import csv

import duckdb

from app.label import a_role_for, better_role, load_labels, next_unlabeled, read_label_set, save_label
from pipeline.label_set import candidates, freeze, pick


def make_db():
    con = duckdb.connect()
    con.execute("""
        create table eval_set as select * from (values
            ('h1', 'train', 'coding', 'p1'),
            ('h2', 'val', 'writing', 'p2'),
            ('h3', 'test', 'q&a', 'p3'),
            ('h4', 'train', 'other', 'p4')
        ) t(conversation_hash, split, segment, prompt)
    """)
    con.execute("""
        create table answers as select * from (values
            ('h1', 'small', 'a1s', null), ('h1', 'large', 'a1l', null),
            ('h2', 'small', '', null), ('h2', 'large', 'a2l', null),
            ('h3', 'small', 'a3s', null), ('h3', 'large', 'a3l', null),
            ('h4', 'small', null, 'http 400'), ('h4', 'large', 'a4l', null)
        ) t(conversation_hash, role, answer, error)
    """)
    return con


def test_candidates_skip_test_and_api_errors_but_keep_cut_offs():
    assert sorted(h for h, _ in candidates(make_db())) == ["h1", "h2"]


def test_pick_caps_each_segment_then_tops_up():
    rows = [(f"w{i}", "writing") for i in range(10)] + [("m0", "math"), ("m1", "math")]
    picked = pick(rows, n=6)  # cap = 6 // 2 = 3 per segment
    segments = [s for _, s in picked]
    assert segments.count("math") == 2 and len(picked) == 6  # both math kept even though writing came first
    assert len({h for h, _ in picked}) == 6


def test_rare_segment_late_in_the_order_still_gets_its_share():
    # 3 segments, n=6 so cap = 2. math only shows up at the very end of the order.
    rows = [(f"w{i}", "writing") for i in range(10)] + [(f"c{i}", "coding") for i in range(10)]
    rows += [("m0", "math"), ("m1", "math")]
    segments = [s for _, s in pick(rows, n=6)]
    assert segments.count("math") == 2


def test_freeze_writes_once(tmp_path):
    path = tmp_path / "label_set.csv"
    freeze(make_db(), str(path))
    first = path.read_text()
    path.write_text(first + "tampered\n")
    freeze(make_db(), str(path))
    assert path.read_text().endswith("tampered\n")  # second run didn't overwrite


def test_relabel_set_is_only_the_flagged_ones(tmp_path):
    path = tmp_path / "label_set.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["conversation_hash", "segment", "relabel"])
        for i in range(6):
            writer.writerow([f"h{i}", "other", int(i < 2)])
    assert len(read_label_set("me", str(path))) == 6
    assert sorted(read_label_set("me-2", str(path))) == ["h0", "h1"]


def test_a_role_is_stable_and_mixed():
    assert a_role_for("h1") == a_role_for("h1")
    assert {a_role_for(f"h{i}") for i in range(50)} == {"small", "large"}


def test_better_role_decodes_a_and_b():
    assert better_role({"a_role": "small", "better": "a"}) == "small"
    assert better_role({"a_role": "small", "better": "b"}) == "large"
    assert better_role({"a_role": "large", "better": "b"}) == "small"
    assert better_role({"a_role": "large", "better": "tie"}) == "tie"


def test_labels_round_trip_and_next_skips_done(tmp_path):
    path = str(tmp_path / "labels.csv")
    save_label({"conversation_hash": "h1", "labeler": "me", "a_role": "small",
                "a_good": "yes", "b_good": "no", "better": "a", "labeled_at": "x"}, path)
    labels = load_labels(path)
    assert labels[0]["a_good"] == "yes"
    assert next_unlabeled(["h1", "h2"], labels, "me") == "h2"
    assert next_unlabeled(["h1", "h2"], labels, "me-2") == "h1"
