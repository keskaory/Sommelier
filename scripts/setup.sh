#!/bin/sh
# one-time setup: venv, deps, git, and the hook that strips ai attribution
set -e
[ -e .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
[ -d .git ] || git init
cp scripts/hooks/commit-msg .git/hooks/commit-msg
chmod +x .git/hooks/commit-msg
echo "done. activate with: source .venv/bin/activate"
