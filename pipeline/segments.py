# step 2 of the pipeline: tag each prompt with a task type (coding, math, chit-chat, writing, q&a, other).
# reads the `prompts` table from step 1 and adds a `segment` column. the eval set is stratified on it.
import re

import duckdb

DB_PATH = "data/sommelier.duckdb"
HEAD_CHARS = 200

# c++ and c# sit outside the \b(...)\b group because \b needs a letter after the ++.
# rules are checked top to bottom and the first match wins, so order matters.
# coding goes first because "write a python function" is coding, not writing.
# math before writing so "write an equation" counts as math.
# chit-chat before q&a so "how are you?" isn't a question about something.
RULES = [
    # class/import/function only count with code punctuation around them,
    # otherwise "how to import a truck" and "my english class" end up as coding.
    ("coding", r"```|#include|\bc\+\+|\bc#|\bdef \w+\(|\bfunction\s*\w*\s*\(|\bclass \w+\s*(\{|\(|:\s*(\n|$))|"
               r"(^|\n)\s*(import [\w.]+|from [\w.]+ import)|<script|"
               r"\b(python|javascript|typescript|java|sql|html|css|vba|php|golang|rust|kotlin|swift|"
               r"bash|powershell|regex|api|json|docker|git|linux|node\.?js|django|flask|pandas|numpy)\b|"
               r"\b(code|coding|compile|debug|bug|syntax|script in|macro|flutter|error|exception|traceback)\b|"
               r"<[a-z]+[\s>]"),
    ("math", r"\b(equation|solve for|integral|derivative|algebra|calculus|probability|calculate|"
             r"math|matrix|theorem|geometry)\b|\d+\s*[\+\*\^=]\s*\d+"),
    ("chit-chat", r"^(hi|hello|hey|yo|thanks|thank you|good morning|good night|how are you|who are you|"
                  r"what is your name|what's up|lol)\b[\s\W]*$"),
    ("writing", r"\b(write|rewrite|story|essay|poem|song|lyrics|script|blog|article|paraphrase|summar\w*|"
                r"translat\w*|proofread|edit|improve|letter|email|scene|chapter|dialogue|speech|caption|"
                r"slogan|headline|outline|roleplay|react to|describe|create|dialog|reformulate|elaborate|brainstorm)\b"),
    ("q&a", r"^(what|why|how|when|where|who|which|is|are|can|could|does|do|did|should|would|explain|"
            r"tell me about|define|list|give me)\b|\?\s*$|\b(explain|differences? between)\b"),
]


def segment_for(prompt):
    # only look at the start. the ask is almost always up front, and long pasted docs
    # mention "calculate" or "error" somewhere and got mis-tagged when we read all of it.
    text = (prompt or "").lower().strip()[:HEAD_CHARS]
    for segment, pattern in RULES:
        if re.search(pattern, text):
            return segment
    return "other"


def add_segments(con):
    df = con.execute("select conversation_hash, prompt from prompts").df()
    df["segment"] = df["prompt"].apply(segment_for)
    # rebuild prompts with the new column instead of altering it, so a rerun just overwrites.
    # duckdb can read the pandas df by its variable name. conversation_hash is unique, so the join can't add rows.
    con.execute("""
        create or replace table prompts as
        select p.conversation_hash, p.hashed_ip, p.timestamp, p.prompt, df.segment
        from prompts p
        join df using (conversation_hash)
    """)


def main():
    con = duckdb.connect(DB_PATH)
    add_segments(con)
    print(con.execute("select segment, count(*) n from prompts group by 1 order by n desc").df())
    con.close()


if __name__ == "__main__":
    main()
