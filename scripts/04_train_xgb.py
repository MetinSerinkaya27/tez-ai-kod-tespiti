"""AST stilometri ozellikleriyle XGBoost egitimi, ablasyon, taban cizgisi ve SHAP.

Akis: ozellikler (parquet onbellekli) -> her ablasyon icin val'de izgara +
erken durdurma ve esik secimi -> satir sayisi taban cizgisi -> test kumesi en
sonda bir kez, sabit esiklerle -> ana model (tum ozellikler) icin SHAP.
Ciktilar: model ve meta -> MODELS/xgb, tablolar ve grafikler -> OUTPUTS/xgb.

--with-ppl: 03c_perplexity.py'nin parquet'leri ozellik matrisine eklenir; mevcut
ablasyonlar aynen kalir, "tum+ppl", "yalniz_ppl", "donemsiz+ppl" eklenir. Ana model
"tum+ppl" olur ve ana xgb'nin ustune yazilmaz: MODELS/xgb_ppl, OUTPUTS/xgb_ppl.
Bayrak yokken davranis eskisiyle birebir aynidir.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from tez.dataset import load_jsonl
from tez.metrics import best_threshold, compute_metrics, save_metrics
from tez.paths import MODELS, OUTPUTS, SEED, TRAIN_JSONL
from tez.perplexity import DEFAULT_MAX_LENGTH, DEFAULT_MODEL, PPL_FEATURES, cached_ppl
from tez.xgb_model import (
    BASELINE_FEATURES,
    LENGTH_CORR_THRESHOLD,
    XGBDetector,
    ablation_sets,
    cached_features,
    fit_length_baseline,
    length_columns,
    per_source_metrics,
    select_and_fit,
)

# Ekransiz ortamda grafik (pencere acilmaz).
matplotlib.use("Agg")

SPLITS = ("train", "val", "test")
MAIN = "tum"
MAIN_PPL = "tum+ppl"
BASELINE = "taban_satir_sayisi"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=TRAIN_JSONL.parent,
                    help="train/val/test.jsonl dizini (varsayilan data/processed)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="ozellik parquet onbellegi (varsayilan <data-dir>/features)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="verilirse model -> <out>/models/xgb, cikti -> <out>/outputs/xgb")
    ap.add_argument("--n-jobs", type=int, default=None, help="XGBoost is parcacigi sayisi")
    ap.add_argument("--with-ppl", action="store_true",
                    help="perplexity ozelliklerini ekle (once 03c_perplexity.py)")
    ap.add_argument("--ppl-model", default=DEFAULT_MODEL)
    ap.add_argument("--ppl-max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return ap.parse_args()


def shap_report(det: XGBDetector, X_test: pd.DataFrame, out: Path) -> pd.DataFrame:
    """TreeExplainer ile test kumesinde global ozet grafikleri ve ilk 20 tablosu."""
    X = X_test[det.features]
    explainer = shap.TreeExplainer(det.model)
    sv = explainer(X)
    for kind, name in ((None, "shap_beeswarm.png"), ("bar", "shap_bar.png")):
        plt.figure()
        shap.summary_plot(sv.values, X, plot_type=kind, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(out / name, dpi=150)
        plt.close("all")
    top = (pd.DataFrame({"feature": det.features,
                         "mean_abs_shap": np.abs(sv.values).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False).head(20))
    top.to_csv(out / "shap_top20.csv", index=False)
    return top


def main() -> None:
    args = parse_args()
    t0 = time.time()
    tag = "xgb_ppl" if args.with_ppl else "xgb"
    main_name = MAIN_PPL if args.with_ppl else MAIN
    model_dir = (args.out_dir / "models" / tag) if args.out_dir else MODELS / tag
    out = (args.out_dir / "outputs" / tag) if args.out_dir else OUTPUTS / tag
    cache_dir = args.cache_dir or args.data_dir / "features"
    out.mkdir(parents=True, exist_ok=True)

    recs, X, y = {}, {}, {}
    for s in SPLITS:
        path = args.data_dir / f"{s}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} bulunamadi; once scripts/03_build_dataset.py calistirilmali.")
        recs[s] = load_jsonl(path)
        X[s] = cached_features(recs[s], cache_dir, s)
        if args.with_ppl:
            try:
                P = cached_ppl(recs[s], cache_dir, s, args.ppl_model, args.ppl_max_length,
                               compute=False)
            except FileNotFoundError as e:
                raise SystemExit(f"--with-ppl: {e}")
            X[s] = X[s].join(P, how="left")
        y[s] = np.array([r["label"] for r in recs[s]], dtype=int)
        print(f"{s}: {len(recs[s])} ornek, AI orani {y[s].mean():.3f}")

    # Uzunluk kolonlari yalnizca egitim kumesinden belirlenir.
    length_cols, max_r = length_columns(X["train"])
    sets = ablation_sets(length_cols, PPL_FEATURES if args.with_ppl else None)
    with open(out / "length_columns.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": LENGTH_CORR_THRESHOLD, "removed": length_cols,
                   "raw_count_max_abs_r": max_r}, f, indent=2)
    print(f"Uzunluk kolonlari ({len(length_cols)}): {length_cols}")

    # 1) Model secimi ve esik: yalnizca train + val.
    fitted: dict[str, tuple] = {}
    grid_logs = []
    for name, cols in sets.items():
        model, params, log = select_and_fit(
            X["train"][cols], y["train"], X["val"][cols], y["val"], n_jobs=args.n_jobs)
        meta = {"ablation": name, "params": params, "seed": SEED}
        if args.with_ppl:
            meta.update(ppl_model=args.ppl_model, ppl_max_length=args.ppl_max_length)
        det = XGBDetector(model, cols, 0.5, meta)
        p_val = det.predict_proba(X["val"])
        det.threshold = best_threshold(y["val"], p_val)
        fitted[name] = (det, p_val)
        grid_logs += [{"ablation": name, **row} for row in log]
        print(f"[{name}] {len(cols)} ozellik, secilen {params}, esik {det.threshold:.4f}")
    pd.DataFrame(grid_logs).to_csv(out / "grid_search.csv", index=False)

    lr = fit_length_baseline(X["train"], y["train"])
    p_val_lr = lr.predict_proba(X["val"][BASELINE_FEATURES])[:, 1]
    thr_lr = best_threshold(y["val"], p_val_lr)

    # 2) Test: tek seferlik, val'de secilmis sabit esiklerle.
    rows, src_rows = [], []
    sources = [r["source"] for r in recs["test"]]
    preds = {name: (det.predict_proba(X["test"]), det.threshold, p_val, len(det.features))
             for name, (det, p_val) in fitted.items()}
    preds[BASELINE] = (lr.predict_proba(X["test"][BASELINE_FEATURES])[:, 1], thr_lr,
                       p_val_lr, len(BASELINE_FEATURES))
    for name, (p_test, thr, p_val, n_feat) in preds.items():
        for split, yy, pp in (("val", y["val"], p_val), ("test", y["test"], p_test)):
            rows.append({"model": name, "split": split, "n_features": n_feat,
                         **compute_metrics(yy, pp, thr)})
        src_rows += [{"model": name, **r} for r in per_source_metrics(y["test"], p_test, sources, thr)]

    table = pd.DataFrame(rows)
    table.to_csv(out / "ablation_metrics.csv", index=False)
    save_metrics(out / "ablation_metrics.json", rows)
    pd.DataFrame(src_rows).to_csv(out / "per_source_metrics.csv", index=False)
    save_metrics(out / "per_source_metrics.json", src_rows)

    # 3) Kaydet: ana model + ablasyon modelleri.
    main_det = fitted[main_name][0]
    main_det.meta["test_metrics"] = next(
        r for r in rows if r["model"] == main_name and r["split"] == "test")
    main_det.save(model_dir)
    for name, (det, _) in fitted.items():
        if name != main_name:
            det.save(model_dir / "ablation" / name)

    # 4) SHAP (ana model, test kumesi).
    top = shap_report(main_det, X["test"], out)

    cols = ["model", "split", "n_features", "threshold", "accuracy", "f1", "roc_auc", "pr_auc"]
    print("\n" + table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nSHAP ilk 10:\n" + top.head(10).to_string(index=False))
    print(f"\nModel -> {model_dir}\nCiktilar -> {out}\nSure: {time.time() - t0:.1f} sn")


if __name__ == "__main__":
    main()
