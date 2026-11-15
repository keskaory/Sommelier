#!/bin/sh
# one-time setup: venv, deps, git
set -e
[ -e .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
[ -d .git ] || git init
echo "done. activate with: source .venv/bin/activate"
