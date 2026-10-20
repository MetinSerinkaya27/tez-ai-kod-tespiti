"""Insan ve AI kayitlarini eslestirip depo bazli train/val/test bolmelerini yazar.

Cikti alanlari: id, pair_id, code, label, repo, source. Bolme oranlari ornek
sayisina gore 70/15/15; bir depo yalnizca tek bir bolmede yer alir.
"""
from __future__ import annotations

import sys

from tez.audit import pair_similarity
from tez.dataset import (
    check_splits,
    drop_similar_pairs,
    load_jsonl,
    pair_records,
    split_by_repo,
    split_stats,
    write_jsonl,
)
from tez.paths import AI_JSONL, HUMAN_JSONL, SEED, TEST_JSONL, TRAIN_JSONL, VAL_JSONL

OUTPUTS = {"train": TRAIN_JSONL, "val": VAL_JSONL, "test": TEST_JSONL}
MAX_BODY_SIMILARITY = 0.8


def main() -> None:
    for path, step in ((HUMAN_JSONL, "01_build_human.py"), (AI_JSONL, "02_generate_ai.py")):
        if not path.exists():
            sys.exit(f"{path} bulunamadi; once scripts/{step} calistirilmali.")

    human, ai = load_jsonl(HUMAN_JSONL), load_jsonl(AI_JSONL)
    records, dropped = pair_records(human, ai)
    print(f"{len(human)} insan + {len(ai)} AI kaydindan {len(records)} ornek tutuldu")
    for reason, n in sorted(dropped.items()):
        if n:
            print(f"  atilan {reason}: {n}")
    if not records:
        sys.exit("Eslesen kayit yok; ai.jsonl id'leri human.jsonl ile uyusmuyor olabilir.")

    # Govdesi insan ciftine fazla benzeyen AI kodlari etiket gurultusudur (03b denetimi).
    records, n_similar = drop_similar_pairs(
        records, MAX_BODY_SIMILARITY,
        lambda h, a: pair_similarity(h, a)["seq_ratio_body"],
    )
    print(f"  atilan ai_insana_cok_benzer (seq_ratio_body > {MAX_BODY_SIMILARITY}): {n_similar}")

    splits = split_by_repo(records, seed=SEED)
    check_splits(splits)
    for name, recs in splits.items():
        write_jsonl(OUTPUTS[name], recs)

    print(f"\n{'bolme':<6} {'ornek':>7} {'oran':>6} {'depo':>5} {'cift':>6} {'insan':>6} {'ai':>6}")
    for row in split_stats(splits):
        print(
            f"{row['bolme']:<6} {row['ornek']:>7} {row['oran']:>6.1%} {row['depo']:>5} "
            f"{row['cift']:>6} {row['insan']:>6} {row['ai']:>6}  {row['kaynak']}"
        )
    print(f"\nYazildi -> {TRAIN_JSONL.parent}")


if __name__ == "__main__":
    main()
