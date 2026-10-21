"""Karakter + token n-gram modeli (tez.ngram_model): egitim, val'de secim, test.

Akis: kodlar normalize edilir (idempotent) -> her gorunumun TF-IDF sozlugu
YALNIZCA train'de kurulur -> gorunum kombinasyonu x siniflandirici x C izgarasi
val ROC AUC'ye (esitlikte val logloss) gore secilir -> esik best_threshold ile
val'de -> test en sonda bir kez, sabit esikle.
Ciktilar: model -> <models>/ngram (ngram.joblib + meta.json), tablolar ->
<outputs>/ngram (grid_search.csv, metrics.csv/json, per_source_metrics.csv,
top_ngrams.csv, predictions_test.csv, summary.md).

Kullanim:
    uv run python scripts/04b_train_ngram.py
    uv run python scripts/04b_train_ngram.py --data-dir <dir> --out-dir <dir> --quick
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tez.dataset import load_jsonl
from tez.metrics import compute_metrics, save_metrics
from tez.ngram_model import (
    CLF_GRID,
    VIEW_COMBOS,
    VIEWS,
    grid_points,
    peak_memory_mb,
    prep_code,
    select_and_fit,
)
from tez.paths import MODELS, OUTPUTS, SEED, TRAIN_JSONL
from tez.robustness import bootstrap_ci, leakage_report

SPLITS = ("train", "val", "test")
QUICK_CLF = {"lr": [1.0, 10.0]}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=TRAIN_JSONL.parent,
                    help="train/val/test.jsonl dizini (varsayilan data/processed)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="verilirse model -> <out>/models/ngram, cikti -> <out>/outputs/ngram")
    ap.add_argument("--views", nargs="+", choices=VIEWS, default=None,
                    help="yalnizca bu gorunumlerin kombinasyonlari denenir")
    ap.add_argument("--quick", action="store_true", help="kucuk izgara (yalnizca LR, C=1,10)")
    ap.add_argument("--min-df", type=int, default=None, help="tum gorunumler icin min_df")
    ap.add_argument("--max-features", type=int, default=None,
                    help="tum gorunumler icin max_features")
    ap.add_argument("--top", type=int, default=30, help="her yonde raporlanacak n-gram sayisi")
    ap.add_argument("--n-boot", type=int, default=1000)
    return ap.parse_args(argv)


def doc_repo_counts(det, codes: list[str], labels, repos) -> dict[str, dict]:
    """Her n-gram icin egitimde gectigi belge sayisi (sinif basina) ve depo sayisi.

    Sizinti belirtisi: ust siradaki bir n-gram yalnizca 1-2 depoda geciyorsa
    depo/proje ozel bir dize olabilir.
    """
    X = det.transform(codes, prepped=True).tocsc()
    names = det.feature_names()
    labels = np.asarray(labels)
    repos = np.asarray(repos)
    out = {}
    for j in range(X.shape[1]):
        rows = X.indices[X.indptr[j]:X.indptr[j + 1]]
        out[names[j]] = {"df_ai": int((labels[rows] == 1).sum()),
                         "df_human": int((labels[rows] == 0).sum()),
                         "n_repos": int(len(set(repos[rows])))}
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    model_dir = (args.out_dir / "models" / "ngram") if args.out_dir else MODELS / "ngram"
    out = (args.out_dir / "outputs" / "ngram") if args.out_dir else OUTPUTS / "ngram"
    out.mkdir(parents=True, exist_ok=True)

    recs = {}
    for s in SPLITS:
        path = args.data_dir / f"{s}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} bulunamadi; once scripts/03_build_dataset.py calistirilmali.")
        recs[s] = load_jsonl(path)
    leak = leakage_report(recs)
    if any(leak.values()):
        raise SystemExit(f"Bolmeler arasi sizinti: { {k: v for k, v in leak.items() if v} }")
    codes = {s: [prep_code(r["code"]) for r in recs[s]] for s in SPLITS}
    y = {s: np.array([r["label"] for r in recs[s]], dtype=int) for s in SPLITS}
    for s in SPLITS:
        print(f"{s}: {len(y[s])} ornek, AI orani {y[s].mean():.3f}")

    combos = VIEW_COMBOS if args.views is None else [
        c for c in VIEW_COMBOS if set(c) <= set(args.views)]
    grid = grid_points(combos, QUICK_CLF if args.quick else CLF_GRID)
    view_cfg = {v: {k: val for k, val in (("min_df", args.min_df),
                                           ("max_features", args.max_features)) if val is not None}
                for v in VIEWS}

    # 1) Secim: yalnizca train + val.
    t_fit = time.time()
    det, log, res = select_and_fit(codes["train"], y["train"], codes["val"], y["val"],
                                   grid, view_cfg)
    fit_sec = time.time() - t_fit
    pd.DataFrame(log).to_csv(out / "grid_search.csv", index=False)
    print(f"Secilen: gorunum {'+'.join(det.views)}, {det.meta['clf']} C={det.meta['C']}, "
          f"esik {det.threshold:.4f} ({len(grid)} aday, {fit_sec:.1f} sn)")

    # 2) Test: tek seferlik, sabit esik.
    p = {s: det.predict_proba(codes[s], prepped=True) for s in ("val", "test")}
    rows = [{"model": "ngram", "split": s, "views": "+".join(det.views),
             **compute_metrics(y[s], p[s], det.threshold)} for s in ("val", "test")]
    from tez.xgb_model import per_source_metrics  # salt okunur arayuz
    src_rows = per_source_metrics(y["test"], p["test"], [r["source"] for r in recs["test"]],
                                  det.threshold)
    repos_test = [r["repo"] for r in recs["test"]]
    ci = bootstrap_ci(y["test"], p["test"], det.threshold, repos_test, args.n_boot, SEED)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(src_rows).to_csv(out / "per_source_metrics.csv", index=False)
    pd.DataFrame({"id": [r["id"] for r in recs["test"]], "repo": repos_test,
                  "source": [r["source"] for r in recs["test"]], "label": y["test"],
                  "p": p["test"]}).to_csv(out / "predictions_test.csv", index=False)

    # 3) Yorumlanabilirlik: en ayirt edici n-gramlar (egitim belge/depo sayilari ile).
    top = det.top_ngrams(args.top)
    stats = doc_repo_counts(det, codes["train"], y["train"], [r["repo"] for r in recs["train"]])
    top_rows = []
    for side, items in top.items():
        for rank, (name, w) in enumerate(items, 1):
            view, ng = name.split(":", 1)
            top_rows.append({"yon": side, "sira": rank, "view": view, "ngram": repr(ng),
                             "weight": w, **stats[name]})
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(out / "top_ngrams.csv", index=False)
    suspicious = top_df[top_df["n_repos"] <= 2]

    det.meta.update({"test_metrics": rows[1], "n_train": int(len(y["train"])),
                     "vocab": res["vocab"]})
    det.save(model_dir)
    resources = {"fit_select_sec": round(fit_sec, 1), "vectorize_sec": res["vectorize_sec"],
                 "vocab": res["vocab"], "peak_memory_mb": round(peak_memory_mb(), 1),
                 "model_size_mb": round((model_dir / "ngram.joblib").stat().st_size / 2**20, 2),
                 "total_sec": round(time.time() - t0, 1)}
    summary = {"chosen": {k: det.meta[k] for k in ("views", "clf", "C")},
               "threshold": det.threshold, "metrics": rows, "ci_test_depo": ci,
               "per_source_test": src_rows, "resources": resources, "leakage": leak,
               "n_grid": len(grid), "seed": SEED, "n_suspicious_top": int(len(suspicious)),
               "model_dir": str(model_dir)}
    save_metrics(out / "metrics.json", summary)
    (out / "summary.md").write_text(summary_markdown(summary, log, top_df), encoding="utf-8")

    print(pd.DataFrame(rows)[["split", "threshold", "accuracy", "f1", "roc_auc", "pr_auc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"test AUC (depo GA) {ci['roc_auc']['point']:.4f} "
          f"[{ci['roc_auc']['lo']:.4f}, {ci['roc_auc']['hi']:.4f}]")
    for side in ("ai", "human"):
        head = top_df[top_df.yon == side].head(8)
        print(f"En ayirt edici ({side}): "
              + ", ".join(f"{t.view}:{t.ngram}" for t in head.itertuples()))
    if len(suspicious):
        print(f"UYARI: ust n-gramlardan {len(suspicious)} tanesi egitimde <=2 depoda geciyor "
              "(depo ozel dize olabilir; top_ngrams.csv).")
    print(f"Kaynak: {resources}")
    print(f"\nModel -> {model_dir}\nCiktilar -> {out}\nSure: {time.time() - t0:.1f} sn")


def summary_markdown(s: dict, log: list[dict], top_df: pd.DataFrame) -> str:
    ch, r = s["chosen"], s["resources"]
    L = ["# n-gram modeli ozeti\n",
         f"Secilen (val ROC AUC, esitlikte logloss): gorunum **{'+'.join(ch['views'])}**, "
         f"siniflandirici **{ch['clf']}**, C={ch['C']}; esik {s['threshold']:.4f} (val, F1). "
         f"Izgara: {s['n_grid']} aday. Test bir kez, sabit esikle.\n",
         "| bolme | AUC | PR-AUC | F1 | accuracy |", "|---|---:|---:|---:|---:|"]
    for m in s["metrics"]:
        L.append(f"| {m['split']} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['f1']:.4f} | "
                 f"{m['accuracy']:.4f} |")
    c = s["ci_test_depo"]["roc_auc"]
    L += ["", f"Test AUC depo bazli %95 GA: {c['point']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]\n",
          "## Izgara (val; ilk 10)", "| gorunum | clf | C | n_ozellik | val AUC | val logloss |",
          "|---|---|---:|---:|---:|---:|"]
    for g in sorted(log, key=lambda g: (-g["val_roc_auc"], g["val_logloss"]))[:10]:
        L.append(f"| {g['views']} | {g['clf']} | {g['C']} | {g['n_features']} | "
                 f"{g['val_roc_auc']:.4f} | {g['val_logloss']:.4f} |")
    L += ["", f"## Kaynaklar\nSecim+egitim {r['fit_select_sec']} sn, tepe bellek "
          f"{r['peak_memory_mb']} MB, model {r['model_size_mb']} MB, sozluk {r['vocab']}.\n",
          "## En ayirt edici n-gramlar (agirlik; egitim belge sayilari, depo sayisi)",
          "n_repos <= 2 olanlar depo ozel dize olabilir (sizinti belirtisi).\n"]
    for side, title in (("ai", "AI yonu (+)"), ("human", "Insan yonu (-)")):
        L += [f"### {title}", "| # | gorunum | n-gram | w | df_ai | df_insan | depo |",
              "|---:|---|---|---:|---:|---:|---:|"]
        for t in top_df[top_df.yon == side].itertuples():
            ng = t.ngram.replace("|", "\\|")
            L.append(f"| {t.sira} | {t.view} | `{ng}` | {t.weight:+.3f} | {t.df_ai} | "
                     f"{t.df_human} | {t.n_repos} |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
