# segments.py: one example per rule plus the order cases that are easy to break
import duckdb
import pytest

from pipeline.segments import add_segments, segment_for


@pytest.mark.parametrize("prompt, expected", [
    ("c++ get all folders in a directory", "coding"),
    ("write a python function that reverses a list", "coding"),  # coding beats writing
    ("```\nprint(1)\n```", "coding"),
    ("solve for x: 2x + 3 = 7", "math"),
    ("Write an equation to solve this question", "math"),  # math beats writing
    ("hi", "chit-chat"),
    ("How are you?", "chit-chat"),  # chit-chat beats q&a
    ("write a poem about rain", "writing"),
    ("Summarize the next section for me", "writing"),
    ("what are common hazards related to working alone?", "q&a"),
    ("question for supplier of UPS?", "q&a"),
    ("galaxy mountain", "other"),
    ("a barcode scanner", "other"),  # "code" inside a word shouldn't count
])
def test_segment_for(prompt, expected):
    assert segment_for(prompt) == expected


def test_add_segments_keeps_rows_and_can_rerun():
    con = duckdb.connect()
    con.execute("""
        create table prompts as select * from (values
            ('a', 'ip1', timestamp '2023-05-01', 'hi'),
            ('b', 'ip2', timestamp '2023-05-01', 'write a poem')
        ) t(conversation_hash, hashed_ip, timestamp, prompt)
    """)
    add_segments(con)
    add_segments(con)  # second run has to work now that the column exists
    rows = con.execute("select conversation_hash, segment from prompts order by 1").fetchall()
    assert rows == [("a", "chit-chat"), ("b", "writing")]


@pytest.mark.parametrize("prompt, expected", [
    ('le puedes agregar un required a este campo <input type="text">', "coding"),
    ("fatal: [cnc_api_test1]: FAILED! traceback follows", "coding"),
    ("describe a beautiful sunset in every detail", "writing"),
    ('"Anti-Americanism is still American exceptionalism" Explain this quote.', "q&a"),
])
def test_segment_for_cases_found_in_other(prompt, expected):
    assert segment_for(prompt) == expected


def test_keywords_deep_in_a_long_prompt_are_ignored():
    prompt = "rewrite these points into a paragraph. " + "filler words here. " * 20 + "calculate the total"
    assert segment_for(prompt) == "writing"


def test_react_as_a_verb_is_not_coding():
    assert segment_for("What dataset do the markets react to the most?") != "coding"


@pytest.mark.parametrize("prompt", [
    "how to import a truck from china",
    "my class journal: today we learned about plants",
    "select one from the list below",
    "explain the function of the liver",
    "from now on you are a pirate. stay in character",
])
def test_code_words_in_plain_english_are_not_coding(prompt):
    assert segment_for(prompt) != "coding"


@pytest.mark.parametrize("prompt", [
    "import numpy as np\nx = np.zeros(3)",
    "fix this:\nfrom flask import Flask",
    "class Card { constructor(suit) {} }",
    "function add(a, b) { return a + b }",
])
def test_real_code_still_counts(prompt):
    assert segment_for(prompt) == "coding"


def test_missing_prompt_is_other():
    assert segment_for(None) == "other"
