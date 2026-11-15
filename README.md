# sommelier

an llm router that sends each prompt to a cheap or expensive model, and a proper eval to prove it saves money without hurting quality.

*(work in progress. results coming.)*

## setup
```
sh scripts/setup.sh
source .venv/bin/activate
pytest
```

## data
prompts come from [wildchat-1m](https://huggingface.co/datasets/allenai/WildChat-1M) (allen ai, odc-by license). this public version already has toxic conversations removed. i also drop any conversation wildchat flagged as containing pii (`redacted`), even though it was anonymized.

to rebuild the prompt table, download shards `train-00000-of-00014.parquet` and `train-00007-of-00014.parquet` from the link above into `data/raw/`, then:
```
python -m pipeline.load_prompts
python -m pipeline.segments
python -m pipeline.split
python -m pipeline.eval_set
```
run them in this order. each step rebuilds the `prompts` table from the one before.
