"""Gercek dunya test seti degerlendirmesi (data/real_world, tez.real_world_eval).

Egitilmis dedektor (xgb ya da codebert) 06_evaluate.py'deki yukleme koduyla
yuklenir; esik modelin dogrulama kumesinde secilmis kayitli esigidir ve bu
sette AYARLANMAZ. GA'lar proje bazli cluster bootstrap'tir.

Girdi: <data-dir>/human.jsonl ve <data-dir>/ai.jsonl (07_real_world.py).
Cikti: OUTPUTS/real_world/<model>/ (ya da <out-dir>/outputs/real_world/<model>/):
  summary.md, summary.json, metrics.csv (tum alt kumeler), per_project.csv,
  model_x_prompt.csv, memorization.csv, predictions.csv.

Kullanim:
    uv run python scripts/08_eval_real_world.py --model xgb
    uv run python scripts/08_eval_real_world.py --model codebert --device cuda
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tez.metrics import save_metrics
from tez.paths import DATA, MODELS, OUTPUTS, ROOT, SEED
from tez.real_world_eval import MEMO_THRESHOLD, all_subset_rows, build_rows, evaluate


def load_eval_module():
    """06_evaluate.py'yi modul olarak yukler (load_detector kopyalanmaz)."""
    spec = importlib.util.spec_from_file_location("eval06", ROOT / "scripts" / "06_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("xgb", "codebert"), default="xgb")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="varsayilan: <out-dir>/models/<model> ya da models/<model>")
    ap.add_argument("--data-dir", type=Path, default=DATA / "real_world",
                    help="human.jsonl + ai.jsonl dizini (varsayilan data/real_world)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="verilirse model <out>/models/<model>, cikti <out>/outputs/real_world/<model>")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--memo-threshold", type=float, default=MEMO_THRESHOLD)
    ap.add_argument("--length-bin", type=float, default=1, help="uzunluk kovasi (satir)")
    ap.add_argument("--device", default="auto", help="codebert icin cihaz")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    root_models = (args.out_dir / "models") if args.out_dir else MODELS
    root_out = (args.out_dir / "outputs") if args.out_dir else OUTPUTS
    model_dir = args.model_dir or root_models / args.model
    out = root_out / "real_world" / args.model

    h_path, a_path = args.data_dir / "human.jsonl", args.data_dir / "ai.jsonl"
    human = read_jsonl(h_path)
    if not human:
        raise SystemExit(f"{h_path} yok ya da bos; once: uv run python scripts/07_real_world.py human")
    ai = read_jsonl(a_path)
    if not ai:
        raise SystemExit(f"{a_path} yok ya da bos; once AI kodlari uretilmeli: "
                         "uv run python scripts/07_real_world.py ai")
    if not model_dir.exists():
        raise SystemExit(f"Model dizini yok: {model_dir} (once 04/05 egitim betikleri)")

    det = load_eval_module().load_detector(args.model, model_dir, args.device)
    thr = float(det.threshold)  # val'de secilmis; burada degistirilmez
    rows = build_rows(human, ai)
    p = np.asarray(det.predict_proba([r["code"] for r in rows]), dtype=float)
    print(f"{len(rows)} kod ({sum(r['label'] for r in rows)} AI), "
          f"{len({r['project'] for r in rows})} proje, esik {thr:.4f} ({model_dir})")

    res = evaluate(rows, p, thr, args.n_boot, SEED, args.memo_threshold, args.length_bin)
    assert res["threshold"] == thr
    res.update({"model": args.model, "model_dir": str(model_dir),
                "data_dir": str(args.data_dir)})

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: r[k] for k in ("uid", "func_id", "project", "model", "prompt_type",
                                     "label", "fstring", "sim_body")} | {"p": pi}
                  for r, pi in zip(rows, p)]).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(all_subset_rows(res)).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(res["per_project"]).to_csv(out / "per_project.csv", index=False)
    pd.DataFrame(res["model_x_prompt"]).to_csv(out / "model_x_prompt.csv", index=False)
    pd.DataFrame([{"uid": r["uid"], "model": r["model"], "prompt_type": r["prompt_type"],
                   "project": r["project"], "sim_body": r["sim_body"], "p": pi}
                  for r, pi in zip(rows, p) if r["label"] == 1]
                 ).to_csv(out / "memorization.csv", index=False)
    save_metrics(out / "summary.json", res)
    (out / "summary.md").write_text(summary_markdown(res), encoding="utf-8")
    o = res["overall"]
    print(f"AUC {o['roc_auc']:.4f} [{o['roc_auc_lo']:.4f}, {o['roc_auc_hi']:.4f}]  "
          f"F1 {o['f1']:.4f} [{o['f1_lo']:.4f}, {o['f1_hi']:.4f}] (proje bootstrap)")
    print(f"Ciktilar -> {out}\nSure: {time.time() - t0:.1f} sn")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _ci(r: dict, k: str) -> str:
    v = r.get(k)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"


HDR = ["| alt kume | n (AI/insan) | proje | AUC | F1 | recall | TNR | accuracy |",
       "|---|---:|---:|---:|---:|---:|---:|---:|"]


def _line(r: dict) -> str:
    if r.get("durum") != "tamam":
        return f"| {r['subset']} | 0 | 0 | bos | | | | |"
    return (f"| {r['subset']} | {r['n']} ({r['n_ai']}/{r['n_human']}) | {r['n_projects']} | "
            f"{_ci(r, 'roc_auc')} | {_ci(r, 'f1')} | {_ci(r, 'recall')} | {_ci(r, 'tnr')} | "
            f"{_ci(r, 'accuracy')} |")


def summary_markdown(s: dict) -> str:
    L = [f"# Gercek dunya degerlendirmesi: {s['model']}\n",
         f"Model: `{s['model_dir']}`; esik {s['threshold']:.4f} (val'de secildi, bu sette sabit).",
         f"Veri: {s['n']} kod ({s['n_ai']} AI, {s['n_human']} insan), {s['n_projects']} proje; "
         f"ureticiler {s['models']}, istem tipleri {s['prompt_types']}.",
         f"GA: proje bazli cluster bootstrap, {s['n_boot']} tekrar, seed {s['seed']} (%95 yuzdelik). "
         "Uretici/istem alt kumeleri: o grubun AI kodlari + TUM insan kodlari.\n"]
    if s["n_projects"] < 30:
        L.append(f"**UYARI:** yalnizca {s['n_projects']} proje var; proje bazli GA kaba ve "
                 "muhtemelen dar/kararsizdir, nokta tahminleri temkinli yorumlanmali.\n")
    L += ["## Genel ve uretici / istem tipi", *HDR, _line(s["overall"])]
    L += [_line(r) for r in s["per_model"]] + [_line(r) for r in s["per_prompt"]]
    L += ["", "## Uretici x istem tipi", *HDR] + [_line(r) for r in s["model_x_prompt"]]
    L += ["", "## Proje basina insan dogru-negatif orani",
          "TNR GA'si Wilson (proje ici fonksiyonlar bagimsiz varsayilir; iyimser).\n",
          "| proje | insan n | TNR [GA] | ort. p (insan) | AI n | AI recall |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in s["per_project"]:
        L.append(f"| {r['project']} | {r['n_human']} | {r['tnr']:.3f} [{r['tnr_lo']:.3f}, "
                 f"{r['tnr_hi']:.3f}] | {r['mean_p_human']:.3f} | {r['n_ai']} | "
                 f"{r['ai_recall']:.3f} |")
    m = s["memorization"]
    d = m["dist"]
    L += ["", "## Ezber kontrolu",
          f"AI kodu ile eslestigi insan kodu arasindaki govde benzerligi (seq_ratio_body). "
          f"Eslesen AI: {m['n_ai_matched']}, eslesmeyen: {m['n_ai_unmatched']}. "
          f"> {m['memo_threshold']}: {m['n_high']} ({m['frac_high']:.1%}).",
          f"Dagilim: p5 {d['p5']:.3f}, p25 {d['p25']:.3f}, medyan {d['p50']:.3f}, "
          f"p75 {d['p75']:.3f}, p95 {d['p95']:.3f}, ort {d['mean']:.3f}.\n", *HDR,
          _line(m["dahil"]), _line(m["haric"]), _line(m["yalniz_yuksek"])]
    f = s["fstring"]
    L += ["", "## Donem kontrolu (f-string)",
          "Egitim verisinde f-string AI'yi isaret ediyordu; buradaki insan kodu 2020-2022 "
          "surumlerinden oldugu icin f-string insan kodunda da yaygin olabilir. Iki alt kume "
          "her iki sinifi da icerir.",
          f"f-string orani: insan {f['frac_fstring_human']:.1%}, AI {f['frac_fstring_ai']:.1%}; "
          f"ayristirilamayan {f['n_parse_fail']}.\n", *HDR, _line(f["with"]), _line(f["without"])]
    L += ["", "## Uzunluk eslestirmeli alt orneklem",
          "Her satir-sayisi kovasinda iki siniftan esit sayida kod (audit.length_matched_sample).\n",
          *HDR, _line(s["length_matched"])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
