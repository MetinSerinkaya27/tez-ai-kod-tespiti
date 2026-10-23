"""N uyeli topluluk modeli (tez.ensemble): xgb, codebert, ngram.

Adimlar:
  1. Uyeler yuklenir (--members; varsayilan: models/ altinda mevcut olan
     xgb/codebert/ngram'in hepsi; eksik uye uyariyla atlanir, en az 2 uye
     gerekir). CodeBERT icin --device.
  2. val ve test tahminleri hesaplanir; alt model tahminleri
     outputs/ensemble/preds_<bolme>.parquet'te id ile onbelleklenir (model
     dizini degisirse ilgili sutun yeniden hesaplanir).
  3. Birlestiriciler (ortalama, logit uzerinde L2 LR) ve esikleri YALNIZCA val'de
     ogrenilir; models/ensemble/ensemble.json'a kaydedilir.
  4. Test (sabit esik): tek modeller vs ortalama vs lojistik, depo bazli bootstrap
     GA, uye ciftleri arasi korelasyon/uyusmazlik; val'de 5 katli depo bazli
     cross-fitting ile N agirligin kararliligi.

Iyimserlik: alt modellerin esik/erken durdurmasi ve birlestirici val'de secildi;
val metrikleri iyimserdir, test temizdir.

Kullanim:
    uv run python scripts/09_ensemble.py --device auto
    uv run python scripts/09_ensemble.py --members xgb ngram
    # prova (eski arayuz, iki uye): ikinci uye olarak baska bir XGB
    uv run python scripts/09_ensemble.py --second-kind xgb --second-name xgb_donemsiz \
        --second-dir models/xgb/ablation/donemsiz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tez.dataset import load_jsonl
from tez.ensemble import (
    DEFAULT_C,
    MEMBER_KINDS,
    METHODS,
    OPTIMISM_NOTE,
    EnsembleDetector,
    pairwise_agreement,
    cross_fit,
    load_member,
    member_exists,
)
from tez.metrics import compute_metrics, save_metrics
from tez.paths import MODELS, OUTPUTS, SEED, TRAIN_JSONL
from tez.robustness import bootstrap_ci, bootstrap_delta_ci, leakage_report

SPLITS = ("train", "val", "test")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=TRAIN_JSONL.parent,
                    help="train/val/test.jsonl dizini (varsayilan data/processed)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="verilirse modeller <out>/models/*, cikti <out>/outputs/ensemble")
    ap.add_argument("--device", default="auto", help="codebert icin cihaz (auto|cpu|cuda)")
    ap.add_argument("--members", nargs="+", choices=MEMBER_KINDS, default=None,
                    help="uyeler (varsayilan: mevcut olan xgb/codebert/ngram'in hepsi)")
    ap.add_argument("--xgb-dir", type=Path, default=None, help="varsayilan <models>/xgb")
    ap.add_argument("--codebert-dir", type=Path, default=None, help="varsayilan <models>/codebert")
    ap.add_argument("--unixcoder-dir", type=Path, default=None, help="varsayilan <models>/unixcoder")
    ap.add_argument("--ngram-dir", type=Path, default=None, help="varsayilan <models>/ngram")
    ap.add_argument("--second-kind", choices=MEMBER_KINDS, default=None,
                    help="eski arayuz: xgb + bu tur (verilirse --members yok sayilir)")
    ap.add_argument("--second-dir", type=Path, default=None, help="varsayilan <models>/<kind>")
    ap.add_argument("--second-name", default=None, help="varsayilan --second-kind")
    ap.add_argument("--name", default="ensemble",
                    help="model/cikti dizini adi (orn. ensemble_unix)")
    ap.add_argument("--method", choices=METHODS, default="logistic",
                    help="kaydedilen dedektorun birlestiricisi")
    ap.add_argument("--C", type=float, default=DEFAULT_C, help="LR L2 ters guclu (sabit)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--no-cache", action="store_true", help="tahmin onbellegini yok say")
    return ap.parse_args(argv)


def resolve_specs(args: argparse.Namespace, root_models: Path) -> dict[str, dict]:
    """Uye adlari -> {"kind", "dir"}; eksik uyeler uyariyla atlanir (en az 2 gerekir)."""
    dirs = {"xgb": args.xgb_dir, "codebert": args.codebert_dir,
            "unixcoder": args.unixcoder_dir, "ngram": args.ngram_dir}
    if args.second_kind:  # eski iki uyeli arayuz: eksik uye hatadir
        name = args.second_name or args.second_kind
        if name == "xgb":
            raise SystemExit("--second-name 'xgb' olamaz (ilk uye adi)")
        specs = {"xgb": {"kind": "xgb", "dir": str(args.xgb_dir or root_models / "xgb")},
                 name: {"kind": args.second_kind,
                        "dir": str(args.second_dir or dirs[args.second_kind]
                                   or root_models / args.second_kind)}}
        missing = [f"{n} ({s['kind']}): {s['dir']}" for n, s in specs.items()
                   if not member_exists(s["kind"], s["dir"])]
        if missing:
            raise SystemExit("Alt model bulunamadi:\n  " + "\n  ".join(missing))
        return specs
    wanted = list(dict.fromkeys(args.members or MEMBER_KINDS))
    specs = {}
    for k in wanted:
        d = str(dirs[k] or root_models / k)
        if member_exists(k, d):
            specs[k] = {"kind": k, "dir": d}
        else:
            print(f"UYARI: uye '{k}' bulunamadi ({d}); atlandi.")
    if len(specs) < 2:
        raise SystemExit(f"En az 2 uye gerekli; mevcut: {list(specs) or 'yok'}. Once "
                         "04_train_xgb.py / 04b_train_ngram.py / 05_train_codebert.py.")
    print(f"Uyeler: {', '.join(specs)}")
    return specs


def dir_fingerprint(d: Path) -> str:
    """Model dizininin ust duzey dosyalarinin (ad, boyut, mtime) hash'i."""
    h = hashlib.sha256(str(d.resolve()).encode())
    for f in sorted(p for p in d.iterdir() if p.is_file()):
        st = f.stat()
        h.update(f"{f.name}\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
    return h.hexdigest()[:16]


def cached_member_preds(split: str, recs: list[dict], ens: EnsembleDetector, out: Path,
                        use_cache: bool) -> np.ndarray:
    """(n, k) alt model olasiliklari; id+kod ve model parmak izi uyusan sutunlar onbellekten."""
    ids = [r["id"] for r in recs]
    path, meta_path = out / f"preds_{split}.parquet", out / f"preds_{split}.json"
    df = pd.DataFrame({"id": ids})
    meta: dict = {}
    codes = [r["code"] for r in recs]
    data_fp = hashlib.sha256("\0".join(i + "\0" + c for i, c in zip(ids, codes)).encode()).hexdigest()[:16]
    if use_cache and path.exists() and meta_path.exists():
        old = pd.read_parquet(path)
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if list(old["id"]) == ids and old_meta.get("_veri") == data_fp:
            df, meta = old, old_meta
    meta["_veri"] = data_fp
    changed = False
    for n in ens.names:
        fp = dir_fingerprint(Path(ens.specs[n]["dir"]))
        col = f"p_{n}"
        if col in df and meta.get(n) == fp:
            print(f"  {split}/{n}: onbellekten")
            continue
        t = time.time()
        df[col] = np.asarray(ens.members[n].predict_proba(codes), dtype=float)
        meta[n] = fp
        changed = True
        print(f"  {split}/{n}: hesaplandi ({time.time() - t:.1f} sn)")
    if changed:
        df.to_parquet(path, index=False)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df[[f"p_{n}" for n in ens.names]].to_numpy(dtype=float)


def fmt_ci(d: dict) -> str:
    return f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"


def main(argv=None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    root_models = (args.out_dir / "models") if args.out_dir else MODELS
    root_out = (args.out_dir / "outputs") if args.out_dir else OUTPUTS
    out = root_out / args.name
    ens_dir = root_models / args.name
    specs = resolve_specs(args, root_models)

    recs = {}
    for s in SPLITS:
        path = args.data_dir / f"{s}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} bulunamadi; once scripts/03_build_dataset.py calistirilmali.")
        recs[s] = load_jsonl(path)
    leak = leakage_report(recs)
    if any(leak.values()):
        raise SystemExit(f"Bolmeler arasi sizinti: { {k: v for k, v in leak.items() if v} }")

    members = {n: load_member(s["kind"], s["dir"], args.device) for n, s in specs.items()}
    names = list(specs)
    out.mkdir(parents=True, exist_ok=True)
    tmp = EnsembleDetector(members, specs, {}, args.method)
    P = {s: cached_member_preds(s, recs[s], tmp, out, not args.no_cache) for s in ("val", "test")}
    y = {s: np.array([r["label"] for r in recs[s]], dtype=int) for s in ("val", "test")}
    repos = {s: [r["repo"] for r in recs[s]] for s in ("val", "test")}

    # Birlestirici YALNIZCA val'den; test burada kullanilmaz.
    ens = EnsembleDetector.fit(members, specs, P["val"], y["val"], args.method, args.C,
                               meta={"C": args.C, "seed": SEED, "data_dir": str(args.data_dir)})
    ens.save(ens_dir)
    cf = cross_fit(P["val"], y["val"], repos["val"], args.n_folds, args.C)

    # Sonuc tablolari: her yontemin kendi val esigi testte sabit.
    thr = {n: float(members[n].threshold) for n in names}
    thr.update({m: float(ens.combiners[m]["threshold"]) for m in METHODS})
    probs = {s: {**{n: P[s][:, i] for i, n in enumerate(names)},
                 **{m: ens.combine(P[s], m) for m in METHODS}} for s in ("val", "test")}
    rows, cis = [], {}
    for split in ("val", "test"):
        for k, p in probs[split].items():
            rows.append({"split": split, "model": k, **compute_metrics(y[split], p, thr[k])})
    for k, p in probs["test"].items():
        cis[k] = bootstrap_ci(y["test"], p, thr[k], repos["test"], args.n_boot, SEED)
    # Esiksiz metriklerde eslesik fark: topluluk - en iyi tek model (val AUC'ye gore secildi).
    best_single = max(names, key=lambda n: compute_metrics(y["val"], probs["val"][n], thr[n])["roc_auc"])
    deltas = {}
    for m in METHODS:
        d = bootstrap_delta_ci(y["test"], probs["test"][best_single], probs["test"][m], 0.5,
                               repos["test"], args.n_boot, SEED)
        deltas[m] = {k: d[k] for k in ("roc_auc", "pr_auc", "brier")}
    agree = {s: pairwise_agreement(P[s], names, [thr[n] for n in names])
             for s in ("val", "test")}

    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame([{"model": k, "metric": m, **v} for k, c in cis.items() for m, v in c.items()]
                 ).to_csv(out / "metrics_ci_test.csv", index=False)
    pd.DataFrame({"id": [r["id"] for r in recs["test"]], "repo": repos["test"],
                  "source": [r["source"] for r in recs["test"]], "label": y["test"],
                  **{f"p_{k}": v for k, v in probs["test"].items()}}
                 ).to_csv(out / "predictions_test.csv", index=False)
    summary = {"members": specs, "method": args.method, "C": args.C, "thresholds": thr,
               "combiners": ens.combiners, "note": OPTIMISM_NOTE, "leakage": leak,
               "n_val": int(len(y["val"])), "n_test": int(len(y["test"])),
               "n_test_repos": len(set(repos["test"])), "n_val_repos": len(set(repos["val"])),
               "n_boot": args.n_boot, "seed": SEED, "metrics": rows, "ci_test_depo": cis,
               "best_single": best_single, "delta_vs_best_single": deltas,
               "agreement": agree, "cross_fit": cf, "ensemble_dir": str(ens_dir)}
    save_metrics(out / "summary.json", summary)
    (out / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")

    for k in probs["test"]:
        print(f"  test {k:>14}: AUC {fmt_ci(cis[k]['roc_auc'])}  F1 {fmt_ci(cis[k]['f1'])}")
    lg = ens.combiners["logistic"]
    w = ", ".join(f"{n}={c:.4f}" for n, c in zip(names, lg["coef"]))
    print(f"LR agirliklari (val): {w}, "
          f"kesisim {lg['intercept']:.4f}; esik {ens.threshold:.4f} ({args.method})")
    print(f"NOT: {OPTIMISM_NOTE}")
    print(f"Model -> {ens_dir}\nCiktilar -> {out}\nSure: {time.time() - t0:.1f} sn")


def summary_markdown(s: dict) -> str:
    names = list(s["members"])
    L = ["# Topluluk (ensemble) ozeti\n",
         "Uyeler: " + ", ".join(f"`{n}` ({v['kind']}, `{v['dir']}`)" for n, v in s["members"].items()),
         f"Kaydedilen birlestirici: **{s['method']}** (C={s['C']}). "
         f"Val: {s['n_val']} ornek / {s['n_val_repos']} depo; test: {s['n_test']} ornek / "
         f"{s['n_test_repos']} depo. Bolmeler arasi ortak depo/cift/kod: 0.\n",
         f"**Iyimserlik notu:** {s['note']}\n"]
    if s["n_test_repos"] < 30:
        L.append(f"**UYARI:** test yalnizca {s['n_test_repos']} depo iceriyor; depo bazli GA "
                 "guvenilir degildir.\n")
    L += ["## Birlestirici (val'de ogrenildi)",
          "Lojistik: p = sigmoid(b + sum_i w_i * logit(p_i)), olasiliklar [1e-6, 1-1e-6] "
          "araligina kirpildi. Ortalama: p = mean_i p_i.\n",
          "| uye | agirlik w (tum val) | cross-fit ort. +/- std |", "|---|---:|---:|"]
    lg, cf = s["combiners"]["logistic"], s["cross_fit"]
    for i, n in enumerate(names):
        cfs = (f"{cf['coef_mean'][i]:.4f} +/- {cf['coef_std'][i]:.4f}"
               if cf.get("n_folds") else cf.get("durum", "-"))
        L.append(f"| {n} | {lg['coef'][i]:.4f} | {cfs} |")
    ics = (f"{cf['intercept_mean']:.4f} +/- {cf['intercept_std']:.4f}"
           if cf.get("n_folds") else "-")
    L.append(f"| kesisim | {lg['intercept']:.4f} | {ics} |")
    if cf.get("n_folds"):
        L += ["", f"Cross-fitting: val uzerinde {cf['n_folds']} katli depo bazli GroupKFold; "
              "OOF (disarida kalan) val metrikleri, esik her katin egitim kisminda secildi "
              "(birlestirici icin iyimser olmayan val tahmini; alt modeller yine val'de ayarli):\n",
              "| birlestirici | OOF AUC | OOF PR-AUC | OOF F1 | OOF accuracy |",
              "|---|---:|---:|---:|---:|"]
        for m, v in cf["oof"].items():
            L.append(f"| {m} | {v['roc_auc']:.4f} | {v['pr_auc']:.4f} | {v['f1']:.4f} | "
                     f"{v['accuracy']:.4f} |")
    L += ["", "## Test metrikleri (val esigi sabit) ve depo bazli %95 GA",
          "| model | esik | AUC [GA] | PR-AUC [GA] | F1 [GA] | accuracy [GA] | Brier |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for k, c in s["ci_test_depo"].items():
        L.append(f"| {k} | {s['thresholds'][k]:.4f} | {_ci(c['roc_auc'])} | {_ci(c['pr_auc'])} | "
                 f"{_ci(c['f1'])} | {_ci(c['accuracy'])} | {c['brier']['point']:.4f} |")
    L += ["", f"Eslesik depo bootstrap farki (topluluk - `{s['best_single']}`; tek model val "
          "AUC'sine gore secildi). Yalnizca esiksiz metrikler:\n",
          "| birlestirici | dAUC [GA] | dPR-AUC [GA] | dBrier [GA] |", "|---|---:|---:|---:|"]
    for m, d in s["delta_vs_best_single"].items():
        L.append(f"| {m} | {_dci(d['roc_auc'])} | {_dci(d['pr_auc'])} | {_dci(d['brier'])} |")
    L += ["", "## Val metrikleri (IYIMSER: birlestirici ve esikler burada secildi)",
          "| model | AUC | F1 | accuracy |", "|---|---:|---:|---:|"]
    for r in s["metrics"]:
        if r["split"] == "val":
            L.append(f"| {r['model']} | {r['roc_auc']:.4f} | {r['f1']:.4f} | {r['accuracy']:.4f} |")
    L += ["", "## Uyeler arasi uyum (tum ciftler)",
          "Uyusmazlik: her uyenin kendi val esigiyle 0/1 tahminlerinin farkli oldugu oran.\n",
          "| cift | bolme | Pearson (p) | Pearson (logit) | Spearman | uyusmazlik |",
          "|---|---|---:|---:|---:|---:|"]
    for sp, pairs in s["agreement"].items():
        for pair, a in pairs.items():
            L.append(f"| {pair} | {sp} | {a['pearson_p']:.4f} | {a['pearson_logit']:.4f} | "
                     f"{a['spearman']:.4f} | {a['disagreement']:.4f} |")
    return "\n".join(L) + "\n"


def _ci(d: dict) -> str:
    return f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"


def _dci(d: dict) -> str:
    return f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


if __name__ == "__main__":
    main()
