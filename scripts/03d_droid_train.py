"""DroidCollection egitim bolumunden ek egitim verisi (buyuk modellerin kodu).

Kendi AI verimiz yalnizca deepseek-coder 6.7B'den geliyor; GPT-4o gibi buyuk
modellerin uzun ve duzenli kodunu insan saniyordu. Bu betik DroidCollection'in
(EMNLP 2025) YALNIZCA egitim bolumundeki Python fonksiyonlarini ayni temizlik ve
5-60 satir filtresiyle alir, insan/AI sayisini esitler ve bizim val/test ile
gercek dunya setinde gecen kodlari atar (sizinti olmasin). 03_build_dataset.py
bu dosyayi yalnizca EGITIM bolmesine ekler; val/test bizim veri olarak kalir.
DroidCollection'in test bolumu (10_eval_droid.py) bu nedenle ayri bir dis test olur.
"""
from __future__ import annotations

import hashlib
import json
import random

import pandas as pd

from tez.dataset import load_jsonl
from tez.normalize import extract_function, normalize_code
from tez.paths import DATA, INTERIM, SEED, TEST_JSONL, VAL_JSONL

DROID = DATA / "external" / "droid"
OUT = INTERIM / "external_train.jsonl"
LABELS = {"HUMAN_GENERATED": 0, "MACHINE_GENERATED": 1, "MACHINE_GENERATED_ADVERSARIAL": 1}
MAX_PER_CLASS = 8000


def main() -> None:
    parts = []
    for path in sorted(DROID.glob("train-*.parquet")):
        df = pd.read_parquet(path, columns=["Code", "Label", "Language", "Model_Family", "Source"])
        parts.append(df[(df["Language"] == "Python") & df["Label"].isin(LABELS)])
    df = pd.concat(parts, ignore_index=True)
    print(f"Python ham kayit: {len(df)}")

    blocked = {r["code"].strip() for p in (VAL_JSONL, TEST_JSONL, DATA / "real_world" / "human.jsonl",
                                          DATA / "real_world" / "ai.jsonl") for r in load_jsonl(p)}
    seen, rows = set(), []
    for code, label, fam, src in zip(df["Code"], df["Label"], df["Model_Family"], df["Source"]):
        fn = extract_function(code)
        norm = normalize_code(fn) if fn else None
        if not norm or not 5 <= len(norm.splitlines()) <= 60:
            continue
        key = norm.strip()
        if key in seen or key in blocked:
            continue
        seen.add(key)
        rows.append({"code": norm, "label": LABELS[label], "family": fam, "droid_label": label,
                     "droid_source": src})

    rng = random.Random(SEED)
    human = [r for r in rows if r["label"] == 0]
    ai = [r for r in rows if r["label"] == 1]
    n = min(len(human), len(ai), MAX_PER_CLASS)
    rng.shuffle(human)
    rng.shuffle(ai)
    out = human[:n] + ai[:n]
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            pid = "droid_" + hashlib.sha1(r["code"].encode()).hexdigest()[:12]
            f.write(json.dumps({"id": f"{pid}_{'h' if r['label'] == 0 else 'a'}", "pair_id": pid,
                                "code": r["code"], "label": r["label"], "repo": "droid",
                                "source": "human" if r["label"] == 0 else f"droid:{r['family']}"},
                               ensure_ascii=False) + "\n")
    fams = pd.Series([r["family"] for r in ai[:n]]).value_counts().head(8).to_dict()
    print(f"uygun fonksiyon: insan {len(human)}, AI {len(ai)}; dengeli {n}+{n} -> {OUT}")
    print("AI aileleri:", fams)


if __name__ == "__main__":
    main()
