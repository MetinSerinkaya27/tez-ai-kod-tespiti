"""Model-bagimsiz degerlendirme ve saglamlik raporu (tez.robustness).

Uretilen kanitlar (test kumesinde, val'de secilmis sabit esikle):
  - depo bazli cluster bootstrap %95 GA (karsilastirma: ornek bazli iid),
  - kalibrasyon (Brier, ECE, guvenilirlik diyagrami),
  - perturbasyon saglamligi (black, yeniden adlandirma, girinti) + eslesik GA,
  - kaynak basina metrik ve "gorulmeyen uretici" deneyi (yalniz xgb),
  - train+val uzerinde depo bazli 5 katli CV (yalniz xgb; test kullanilmaz),
  - en emin FP/FN ornekleri (xgb icin SHAP ile).
Ciktilar: OUTPUTS/eval/<model>/ (ya da <out-dir>/outputs/eval/<model>/).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tez.dataset import load_jsonl
from tez.metrics import compute_metrics, save_metrics
from tez.paths import MODELS, OUTPUTS, SEED, TRAIN_JSONL
from tez.robustness import (
    PERTURBATIONS,
    XGBCodeDetector,
    ai_sources,
    bootstrap_ci,
    calibration_summary,
    calibration_table,
    ci_rows,
    cross_source_eval,
    cv_summary,
    error_markdown,
    group_cv_xgb,
    leakage_report,
    make_xgb_fit_fn,
    plot_reliability,
    robustness_report,
)

SPLITS = ("train", "val", "test")
SMALL_GRID = {"max_depth": [3, 5], "learning_rate": [0.1], "min_child_weight": [1]}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("xgb", "codebert"), default="xgb")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="varsayilan: <out-dir>/models/<model> ya da models/<model>")
    ap.add_argument("--data-dir", type=Path, default=TRAIN_JSONL.parent,
                    help="train/val/test.jsonl dizini (varsayilan data/processed)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="verilirse model <out>/models/<model>, cikti <out>/outputs/eval/<model>")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--top-errors", type=int, default=20)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--skip-cross-source", action="store_true")
    ap.add_argument("--small-grid", action="store_true",
                    help="CV ve kaynak deneyinde kucuk izgara (hizli gelistirme)")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--device", default="auto", help="codebert icin cihaz")
    return ap.parse_args()


def load_detector(name: str, model_dir: Path, device: str):
    if name == "xgb":
        from tez.xgb_model import XGBDetector
        return XGBCodeDetector(XGBDetector.load(model_dir))
    from tez.codebert import CodeBertDetector
    return CodeBertDetector.from_dir(model_dir, device=device)


def fmt_ci(d: dict) -> str:
    return f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"


def main() -> None:
    args = parse_args()
    t0 = time.time()
    root_models = (args.out_dir / "models") if args.out_dir else MODELS
    root_out = (args.out_dir / "outputs") if args.out_dir else OUTPUTS
    model_dir = args.model_dir or root_models / args.model
    out = root_out / "eval" / args.model
    out.mkdir(parents=True, exist_ok=True)
    grid = SMALL_GRID if args.small_grid else None

    recs = {}
    for s in SPLITS:
        path = args.data_dir / f"{s}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} bulunamadi; once scripts/03_build_dataset.py calistirilmali.")
        recs[s] = load_jsonl(path)

    # Sizinti kontrolu: bolmeler arasi ortak depo/cift/kod olmamali.
    leak = leakage_report(recs)
    if any(leak.values()):
        raise SystemExit(f"Bolmeler arasi sizinti: { {k: v for k, v in leak.items() if v} }")

    det = load_detector(args.model, model_dir, args.device)
    thr = float(det.threshold)
    test = recs["test"]
    y = np.array([r["label"] for r in test], dtype=int)
    repos = [r["repo"] for r in test]
    pairs = [r["pair_id"] for r in test]
    codes = [r["code"] for r in test]
    p = np.asarray(det.predict_proba(codes), dtype=float)
    n_repo = len(set(repos))
    print(f"test: {len(y)} ornek, {n_repo} depo, esik {thr:.4f} ({model_dir})")

    summary: dict = {"model": args.model, "model_dir": str(model_dir), "threshold": thr,
                     "n_test": int(len(y)), "n_test_repos": n_repo,
                     "n_test_pairs": len(set(pairs)), "seed": SEED, "n_boot": args.n_boot,
                     "leakage": leak}
    pd.DataFrame({"id": [r["id"] for r in test], "repo": repos, "source": [r["source"] for r in test],
                  "label": y, "p": p}).to_csv(out / "predictions_test.csv", index=False)

    # 1) Guven araliklari.
    ci_depo = bootstrap_ci(y, p, thr, repos, args.n_boot, SEED)
    ci_cift = bootstrap_ci(y, p, thr, pairs, args.n_boot, SEED)
    ci_iid = bootstrap_ci(y, p, thr, None, args.n_boot, SEED)
    rows = (ci_rows(ci_depo, yontem="depo") + ci_rows(ci_cift, yontem="cift")
            + ci_rows(ci_iid, yontem="iid"))
    pd.DataFrame(rows).to_csv(out / "metrics_ci.csv", index=False)
    summary["test_metrics"] = compute_metrics(y, p, thr)
    summary["ci"] = {"depo": ci_depo, "cift": ci_cift, "iid": ci_iid}
    print(f"ROC AUC {fmt_ci(ci_depo['roc_auc'])}  F1 {fmt_ci(ci_depo['f1'])} (depo bootstrap)")

    # 3) Kalibrasyon.
    table = calibration_table(y, p)
    table.to_csv(out / "calibration_bins.csv", index=False)
    summary["calibration"] = calibration_summary(y, p)
    plot_reliability(table, out / "reliability.png", f"{args.model} - test")

    # 5a) Kaynak basina.
    from tez.xgb_model import per_source_metrics
    src_rows = per_source_metrics(y, p, [r["source"] for r in test], thr)
    pd.DataFrame(src_rows).to_csv(out / "per_source.csv", index=False)
    summary["per_source"] = src_rows

    # 4) Saglamlik.
    if not args.skip_robustness:
        rob, probs = robustness_report(det, codes, y, repos, PERTURBATIONS, p, args.n_boot, SEED)
        pd.DataFrame(rob).drop(columns=["ornek_hatalar"], errors="ignore").to_csv(
            out / "robustness.csv", index=False)
        summary["robustness"] = rob
        pd.DataFrame({"id": [r["id"] for r in test], "label": y, **{f"p_{k}": v for k, v in probs.items()}}
                     ).to_csv(out / "predictions_perturbed.csv", index=False)
        for r in rob:
            print(f"  [{r['perturbation']}] AUC {r['roc_auc']:.4f} F1 {r['f1']:.4f} "
                  f"degisen {r['n_changed']} basarisiz {r['n_failed']}")

    # 2) CV (yalnizca xgb; test kullanilmaz).
    if args.model == "xgb" and not args.skip_cv:
        from tez.features import features_frame
        fit_recs = recs["train"] + recs["val"]
        X = features_frame(r["code"] for r in fit_recs)
        cv_rows = group_cv_xgb(X, [r["label"] for r in fit_recs], [r["repo"] for r in fit_recs],
                               det.det.features, args.n_folds, grid=grid, n_jobs=args.n_jobs)
        pd.DataFrame(cv_rows).to_csv(out / "cv_folds.csv", index=False)
        summary["cv"] = {"folds": cv_rows, "summary": cv_summary(cv_rows)}
        s = summary["cv"]["summary"]
        print(f"CV ({s['n_folds_ok']} kat): AUC {s['roc_auc']['mean']:.4f} +/- {s['roc_auc']['std']:.4f}")

    # 5b) Gorulmeyen uretici (yalnizca xgb: yeniden egitim gerekir).
    if args.model == "xgb" and not args.skip_cross_source:
        srcs = ai_sources(recs["train"])
        fit_fn = make_xgb_fit_fn(det.det.features, grid=grid, n_jobs=args.n_jobs)
        cs = cross_source_eval(recs["train"], recs["val"], test, fit_fn)
        pd.DataFrame(cs).to_csv(out / "cross_source.csv", index=False)
        summary["cross_source"] = {"train_sources": srcs, "test_sources": ai_sources(test),
                                   "rows": cs,
                                   "not": ("tek kaynak: yalnizca gorulen uretici satiri var; "
                                           "ikinci LLM verisi gelince gorulmeyen satirlar dolar")
                                   if len(set(srcs) | set(ai_sources(test))) < 2 else ""}

    # 6) Hata analizi.
    md = error_markdown(test, p, thr, det, k=args.top_errors)
    (out / "errors.md").write_text(md, encoding="utf-8")

    save_metrics(out / "summary.json", summary)
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    print(f"Ciktilar -> {out}\nSure: {time.time() - t0:.1f} sn")


def summary_markdown(s: dict) -> str:
    L = [f"# Degerlendirme ozeti: {s['model']}\n",
         f"Model: `{s['model_dir']}`; esik {s['threshold']:.4f} (val'de secildi, testte sabit).",
         f"Test: {s['n_test']} ornek, {s['n_test_repos']} depo, {s['n_test_pairs']} cift. "
         f"Bootstrap: {s['n_boot']} tekrar, seed {s['seed']}.",
         "Bolmeler arasi ortak depo/cift/kod: 0 (kontrol edildi).\n"]
    if s["n_test_repos"] < 30:
        L.append(f"**UYARI:** test yalnizca {s['n_test_repos']} depo iceriyor; depo bazli GA "
                 "guvenilir degildir, sonuclar yalnizca hattin calistigini gosterir.\n")
    L += ["## Test metrikleri ve %95 GA",
          "Ornekler bagimsiz degil: ayni depodaki fonksiyonlar ve ayni ciftin insan/AI uyeleri "
          "iliskili. Ana GA depo bazli cluster bootstrap'tir (depolar yerine koyarak cekilir, "
          "secilen deponun tum ornekleri alinir; cift uyeleri ayni depoda oldugundan cift "
          "bagimliligi da kapsanir). iid GA yalnizca karsilastirma icindir.\n",
          "| metrik | nokta | GA (depo) | GA (cift) | GA (iid) |", "|---|---:|---:|---:|---:|"]
    for m, d in s["ci"]["depo"].items():
        c, i = s["ci"]["cift"][m], s["ci"]["iid"][m]
        L.append(f"| {m} | {d['point']:.4f} | [{d['lo']:.4f}, {d['hi']:.4f}] | "
                 f"[{c['lo']:.4f}, {c['hi']:.4f}] | [{i['lo']:.4f}, {i['hi']:.4f}] |")
    c = s["calibration"]
    L += ["", "## Kalibrasyon",
          f"Brier {c['brier']:.4f} (sabit taban orani tahmini: {c['base_rate_brier']:.4f}), "
          f"ECE ({c['n_bins']} kova) {c['ece']:.4f}. Grafik: `reliability.png`.", ""]
    if "robustness" in s:
        L += ["## Saglamlik (perturbasyon)",
              "Ayni donusum her iki sinifa uygulandi, esik degismedi. Fark GA'si eslesik depo "
              "bootstrap'tir. Basarisiz donusumlerde orijinal kod kaldi.\n",
              "| donusum | AUC | F1 | dAUC [GA] | dF1 [GA] | degisen | basarisiz | tahmini donen |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in s["robustness"]:
            if r["perturbation"] == "orijinal":
                L.append(f"| orijinal | {r['roc_auc']:.4f} | {r['f1']:.4f} | - | - | - | - | - |")
                continue
            L.append(f"| {r['perturbation']} | {r['roc_auc']:.4f} | {r['f1']:.4f} | "
                     f"{r['d_roc_auc']:+.4f} [{r['d_roc_auc_lo']:+.4f}, {r['d_roc_auc_hi']:+.4f}] | "
                     f"{r['d_f1']:+.4f} [{r['d_f1_lo']:+.4f}, {r['d_f1_hi']:+.4f}] | "
                     f"{r['n_changed']} | {r['n_failed']} | {r['flip_frac']:.3f} |")
        L.append("")
    L += ["## Kaynak basina (test)", "| kaynak | n | AUC | F1 | accuracy |", "|---|---:|---:|---:|---:|"]
    for r in s["per_source"]:
        L.append(f"| {r['source']} | {r['n']} | {r['roc_auc']:.4f} | {r['f1']:.4f} | {r['accuracy']:.4f} |")
    L.append("")
    if "cv" in s:
        cs = s["cv"]["summary"]
        L += ["## Depo bazli capraz dogrulama (train+val, test kullanilmadi)",
              f"{cs['n_folds_ok']} kat; her katta ic dogrulama (depo ayrik) ile izgara ve esik.\n",
              "| metrik | ortalama | std | min | max |", "|---|---:|---:|---:|---:|"]
        for m, v in cs.items():
            if isinstance(v, dict):
                L.append(f"| {m} | {v['mean']:.4f} | {v['std']:.4f} | {v['min']:.4f} | {v['max']:.4f} |")
        L.append("")
    if "cross_source" in s:
        x = s["cross_source"]
        L += ["## Gorulmeyen uretici", f"Egitim kaynaklari: {x['train_sources']}; test: {x['test_sources']}. {x['not']}\n",
              "| egitim | test | gorulen | AUC | F1 |", "|---|---|---|---:|---:|"]
        for r in x["rows"]:
            if r.get("durum") == "tamam":
                L.append(f"| {r['train_source']} | {r['test_source']} | {r['gorulen']} | "
                         f"{r['roc_auc']:.4f} | {r['f1']:.4f} |")
            else:
                L.append(f"| {r['train_source']} | - | - | {r['durum']} | |")
        L.append("")
    L += ["## Hata analizi", "En emin FP/FN ornekleri: `errors.md`."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
