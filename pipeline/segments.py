# tags each prompt as coding, math, chit-chat, writing, q&a or other. the eval set is stratified on this.
import re

import duckdb

DB_PATH = "data/sommelier.duckdb"
HEAD_CHARS = 200

# first match wins, so order matters: coding first ("write a python function"), math before
# writing ("write an equation"), chit-chat before q&a ("how are you?").
# c++ and c# are outside the \b group because \b doesn't work after a "+".
RULES = [
    # class/import/function need code punctuation, otherwise "how to import a truck" is coding
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
    # first 200 chars only. the ask is usually up front, and long pasted docs kept getting
    # tagged math or coding over some random word deep inside
    text = (prompt or "").lower().strip()[:HEAD_CHARS]
    for segment, pattern in RULES:
        if re.search(pattern, text):
            return segment
    return "other"


def add_segments(con):
    df = con.execute("select conversation_hash, prompt from prompts").df()
    df["segment"] = df["prompt"].apply(segment_for)
    # rebuild the table so rerunning just overwrites it. duckdb reads `df` straight from python.
    # conversation_hash is unique so the join can't add rows.
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
