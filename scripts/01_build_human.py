"""CodeSearchNet (Python, test bolumu) uzerinden insan sinifi veri setini olusturur.

Her ornek icin imza + docstring saklanir; bu ikili daha sonra LLM'e ayni gorevi
yazdirmak icin istem olarak kullanilir (eslestirilmis veri seti).
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import requests
from tqdm import tqdm

from tez.normalize import function_header, normalize_code
from tez.paths import CSN_PARQUET_URL, HUMAN_JSONL, INTERIM, RAW

PARQUET = RAW / "csn_python_test.parquet"

MIN_LINES, MAX_LINES = 5, 60
MIN_DOC_WORDS = 5


def download() -> None:
    if PARQUET.exists():
        return
    RAW.mkdir(parents=True, exist_ok=True)
    with requests.get(CSN_PARQUET_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(PARQUET, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def build_record(row: pd.Series) -> dict | None:
    code = row["whole_func_string"]
    try:
        header = function_header(code)
    except (SyntaxError, ValueError):
        return None
    if header is None:
        return None
    signature, doc = header
    if len(doc.split()) < MIN_DOC_WORDS:
        return None

    norm = normalize_code(code)
    if norm is None:
        return None
    n_lines = len(norm.splitlines())
    if not MIN_LINES <= n_lines <= MAX_LINES:
        return None

    return {
        "id": hashlib.sha1(row["func_code_url"].encode()).hexdigest()[:12],
        "func_name": row["func_name"].split(".")[-1],
        "signature": signature,
        "docstring": doc,
        "code": norm,
        "repo": row["repository_name"],
        "url": row["func_code_url"],
        "label": 0,
    }


def main() -> None:
    download()
    df = pd.read_parquet(PARQUET, columns=[
        "repository_name", "func_name", "whole_func_string", "func_code_url",
    ])

    records, seen = [], set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="human"):
        rec = build_record(row)
        if rec is None or rec["code"] in seen:
            continue
        seen.add(rec["code"])
        records.append(rec)

    INTERIM.mkdir(parents=True, exist_ok=True)
    with open(HUMAN_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(df)} fonksiyondan {len(records)} tanesi tutuldu -> {HUMAN_JSONL}")


if __name__ == "__main__":
    main()
