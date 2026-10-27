"""Dis veri seti testi: DroidCollection (EMNLP 2025) Python, fonksiyon duzeyi.

Egitimde hic gorulmemis buyuk modellerin (GPT-4o, GPT-4o-mini, Qwen, Llama,
DeepSeek, Phi, Gemma, ...) kodlarinda dedektorleri olcer. Yalnizca test bolumu ve
fonksiyon duzeyindeki kaynak (THEVAULT_FUNCTION) kullanilir; kodlar bizim veriyle
ayni normalizasyon ve 5-60 satir filtresinden gecer, egitim verimizde birebir
gecen kodlar cikarilir. Esikler modellerin val'de secilmis esikleridir.

Veri: data/external/droid/test.parquet
  (huggingface.co/datasets/project-droid/DroidCollection, data/test-00000-of-00001.parquet)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from tez.dataset import load_jsonl
from tez.normalize import extract_function, normalize_code
from tez.paths import DATA, MODELS, OUTPUTS, TRAIN_JSONL

LABELS = {"HUMAN_GENERATED": 0, "MACHINE_GENERATED": 1, "MACHINE_GENERATED_ADVERSARIAL": 1}


def prepare(path: Path, sources: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[(df["Language"] == "Python") & df["Source"].isin(sources) & df["Label"].isin(LABELS)].copy()
    codes = []
    for c in df["Code"]:
        fn = extract_function(c)
        norm = normalize_code(fn) if fn else None
        codes.append(norm if norm and 5 <= len(norm.splitlines()) <= 60 else None)
    df["code"] = codes
    df = df[df["code"].notna()].drop_duplicates("code")
    train = {r["code"].strip() for r in load_jsonl(TRAIN_JSONL)}
    df = df[~df["code"].str.strip().isin(train)]
    df["y"] = df["Label"].map(LABELS)
    return df


def report(df: pd.DataFrame, p: np.ndarray, thr: float) -> list[dict]:
    rows = []
    human = df["y"] == 0
    groups = [("tum (adversarial haric)", df["Label"] != "MACHINE_GENERATED_ADVERSARIAL"),
              ("adversarial AI + insan", human | (df["Label"] == "MACHINE_GENERATED_ADVERSARIAL"))]
    for fam in sorted(df.loc[~human, "Model_Family"].unique()):
        groups.append((f"aile={fam}", human | ((df["Model_Family"] == fam)
                                              & (df["Label"] == "MACHINE_GENERATED"))))
    for name, m in groups:
        y, q = df.loc[m, "y"].to_numpy(), p[m.to_numpy()]
        if y.sum() == 0:
            continue
        rows.append({"alt_kume": name, "n_ai": int(y.sum()), "n_insan": int((y == 0).sum()),
                     "auc": round(float(roc_auc_score(y, q)), 4),
                     "ai_yakalama": round(float((q[y == 1] >= thr).mean()), 3),
                     "insan_dogru": round(float((q[y == 0] < thr).mean()), 3),
                     "dogruluk": round(float(accuracy_score(y, (q >= thr).astype(int))), 3)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA / "external" / "droid" / "test.parquet")
    ap.add_argument("--sources", nargs="+", default=["THEVAULT_FUNCTION"])
    ap.add_argument("--models", nargs="+", default=["ensemble_final", "unixcoder", "codebert"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    df = prepare(args.data, args.sources)
    print(f"{len(df)} fonksiyon (insan {int((df.y == 0).sum())}, AI {int(df.y.sum())})")
    out = OUTPUTS / "droid"
    out.mkdir(parents=True, exist_ok=True)
    sonuc = {}
    for name in args.models:
        if name.startswith("ensemble"):
            from tez.ensemble import EnsembleDetector
            det = EnsembleDetector.load(MODELS / name, device=args.device)
        else:
            from tez.codebert import CodeBertDetector
            det = CodeBertDetector.from_dir(MODELS / name, device=args.device)
        p = np.asarray(det.predict_proba(df["code"].tolist()))
        sonuc[name] = report(df, p, float(det.threshold))
        print(f"\n== {name} (esik {det.threshold:.3f})")
        print(pd.DataFrame(sonuc[name]).to_string(index=False))
    (out / "sonuc.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
