"""Proje genelinde kullanilan dosya yollari."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

CSN_PARQUET_URL = (
    "https://huggingface.co/datasets/code-search-net/code_search_net/resolve/"
    "refs%2Fconvert%2Fparquet/python/test/0000.parquet"
)

HUMAN_JSONL = INTERIM / "human.jsonl"
AI_JSONL = INTERIM / "ai.jsonl"

TRAIN_JSONL = PROCESSED / "train.jsonl"
VAL_JSONL = PROCESSED / "val.jsonl"
TEST_JSONL = PROCESSED / "test.jsonl"

SEED = 42
