"""Train/val/test icin dil modeli perplexity ozelliklerini hesaplar ve onbellekler.

Cikti: <cache-dir>/ppl_<bolme>_<hash>.parquet (04_train_xgb.py --with-ppl okur)
ve <cache-dir>/ppl_summary.json (sure, hiz, kesilen kod sayisi).
Model deepseek ailesinden secilmez (bkz. tez/perplexity.py basligi).

Kullanim: uv run python scripts/03c_perplexity.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tez.dataset import load_jsonl
from tez.paths import TRAIN_JSONL
from tez.perplexity import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    cached_ppl,
    ppl_cache_path,
    resolve_device,
)

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=TRAIN_JSONL.parent,
                    help="train/val/test.jsonl dizini (varsayilan data/processed)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="parquet onbellegi (varsayilan <data-dir>/features)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="nedensel LM (deepseek OLMAYAN)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if "deepseek" in args.model.lower():
        raise SystemExit("deepseek ailesi kullanilamaz: AI verisi deepseek ile uretildi.")
    cache_dir = args.cache_dir or args.data_dir / "features"
    device = resolve_device(args.device)
    print(f"Model {args.model}, cihaz {device}, yigin {args.batch_size}, "
          f"max_length {args.max_length}")
    summary = {"model": args.model, "device": device, "batch_size": args.batch_size,
               "max_length": args.max_length, "splits": {}}
    for s in SPLITS:
        path = args.data_dir / f"{s}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} bulunamadi; once scripts/03_build_dataset.py calistirilmali.")
        recs = load_jsonl(path)
        out = ppl_cache_path(recs, cache_dir, s, args.model, args.max_length)
        t0 = time.time()
        X = cached_ppl(recs, cache_dir, s, args.model, args.max_length, compute=True,
                       device=device, batch_size=args.batch_size, progress=True)
        dt = time.time() - t0
        n_trunc = X.attrs.get("n_truncated")
        if n_trunc is None:
            print(f"{s}: onbellekten okundu -> {out.name}")
            continue
        n_nan = int(X.isna().any(axis=1).sum())
        summary["splits"][s] = {"n": len(recs), "seconds": round(dt, 1),
                                "samples_per_sec": round(len(recs) / max(dt, 1e-9), 2),
                                "n_truncated": n_trunc, "n_nan_rows": n_nan, "file": out.name}
        print(f"{s}: {len(recs)} ornek, {dt:.1f} sn ({len(recs) / max(dt, 1e-9):.2f} ornek/sn), "
              f"{n_trunc} kod {args.max_length} tokende kesildi, {n_nan} NaN satir -> {out.name}")
    if summary["splits"]:
        with open(cache_dir / "ppl_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
