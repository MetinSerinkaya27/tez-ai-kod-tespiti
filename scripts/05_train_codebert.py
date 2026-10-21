"""CodeBERT'i insan/AI siniflandirmasi icin ince ayarlar, degerlendirir ve aciklar.

Akis: train ile egitim -> her epoch sonunda val ROC-AUC (esitlikte F1) ile en
iyi checkpoint -> en iyi modelle val'de esik (best_threshold) -> test bir kez,
ayni esikle. Istege bagli: LIME aciklamalari ve ONNX (fp32 + int8) disa aktarimi.

Ornek (6 GB GPU):
  uv run python scripts/05_train_codebert.py --device cuda --amp bf16 \
      --batch-size 8 --grad-accum 4 --epochs 3 --lr 2e-5 --lime-n 3 --onnx --quantize
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from tez.codebert import (
    BASE_MODEL,
    DEFAULT_HEAD_FRAC,
    DEFAULT_MAX_LENGTH,
    TRUNC_STRATEGIES,
    CodeBertDetector,
    EncodedDataset,
    autocast_ctx,
    compare_onnx,
    encode_codes,
    explanation_html,
    export_onnx,
    is_better,
    make_collate,
    predict_loader,
    quantize_onnx,
    save_bundle,
    select_device,
    set_seed,
)
from tez.dataset import load_jsonl
from tez.metrics import best_threshold, compute_metrics, save_metrics
from tez.paths import MODELS, OUTPUTS, PROCESSED, SEED


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=PROCESSED, help="train/val/test.jsonl dizini")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Verilirse model <out>/models/codebert, ciktilar <out>/outputs/codebert altina")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--name", default="codebert",
                   help="model/cikti dizini adi (orn. unixcoder ile 'unixcoder')")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1, help="Isinma orani (toplam adimin)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="256 veya 512")
    p.add_argument("--trunc", choices=TRUNC_STRATEGIES, default="head_tail")
    p.add_argument("--head-frac", type=float, default=DEFAULT_HEAD_FRAC)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--amp", choices=("auto", "none", "fp16", "bf16"), default="auto")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--select-metric", choices=("roc_auc", "f1"), default="roc_auc")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--limit", type=int, default=0,
                   help="Yalnizca deneme: her bolmeden en fazla N ornek (seed ile)")
    p.add_argument("--lime-n", type=int, default=0, help="Aciklanacak test ornegi sayisi")
    p.add_argument("--lime-samples", type=int, default=500)
    p.add_argument("--lime-features", type=int, default=10)
    p.add_argument("--onnx", action="store_true", help="En iyi modeli ONNX'e aktar ve dogrula")
    p.add_argument("--quantize", action="store_true", help="ONNX'e ek olarak dinamik int8")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def load_split(data_dir: Path, name: str, limit: int, seed: int) -> list[dict]:
    path = data_dir / f"{name}.jsonl"
    if not path.exists():
        sys.exit(f"{path} bulunamadi; once scripts/03_build_dataset.py calistirilmali.")
    recs = load_jsonl(path)
    if limit and len(recs) > limit:
        recs = random.Random(seed).sample(recs, limit)
    return recs


def plot_curves(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["step"], history["train_loss"], lw=0.8, label="train (adim)")
    ep_steps = [e["step"] for e in history["epochs"]]
    axes[0].plot(ep_steps, [e["val_loss"] for e in history["epochs"]], "o-", label="val (epoch)")
    axes[0].set_xlabel("optimizer adimi"); axes[0].set_ylabel("kayip"); axes[0].legend()
    eps = [e["epoch"] for e in history["epochs"]]
    for key in ("roc_auc", "f1"):
        axes[1].plot(eps, [e["val"][key] for e in history["epochs"]], "o-", label=f"val {key}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1.02); axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def per_source_metrics(recs: list[dict], probs: np.ndarray, thr: float) -> dict:
    """Kaynak basina metrikler; AI kaynaklari icin ayrica 'insan + o kaynak' alt kumesi."""
    y = np.array([r["label"] for r in recs])
    src = np.array([r["source"] for r in recs])
    out = {}
    for s in sorted(set(src)):
        m = src == s
        out[s] = compute_metrics(y[m], probs[m], thr)
        if s != "human" and (src == "human").any():
            mm = m | (src == "human")
            out[f"human_vs_{s}"] = compute_metrics(y[mm], probs[mm], thr)
    return out


def main() -> None:
    args = parse_args()
    sys.stdout.reconfigure(errors="replace")
    model_dir = (args.out_dir / "models" / args.name) if args.out_dir else MODELS / args.name
    out_dir = (args.out_dir / "outputs" / args.name) if args.out_dir else OUTPUTS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    amp = args.amp
    if amp == "auto":
        amp = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "none"
    print(f"cihaz={device} amp={amp} max_length={args.max_length} kesme={args.trunc}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    gen = set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train = load_split(args.data_dir, "train", args.limit, args.seed)
    val = load_split(args.data_dir, "val", args.limit, args.seed)

    def encode(recs, name):
        seqs, n_cut = encode_codes(tokenizer, [r["code"] for r in recs], args.max_length, args.trunc, args.head_frac)
        print(f"{name}: {len(recs)} ornek, {n_cut} kesildi ({n_cut / max(len(recs), 1):.1%}), "
              f"sinif {dict(Counter(r['label'] for r in recs))}")
        return seqs, n_cut

    tr_seqs, tr_cut = encode(train, "train")
    va_seqs, va_cut = encode(val, "val")
    collate = make_collate(tokenizer.pad_token_id)
    tr_loader = DataLoader(EncodedDataset(tr_seqs, [r["label"] for r in train]), batch_size=args.batch_size,
                           shuffle=True, generator=gen, collate_fn=collate, num_workers=0)
    va_loader = DataLoader(EncodedDataset(va_seqs, [r["label"] for r in val]),
                           batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    y_val = np.array([r["label"] for r in val])

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2).to(device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(tr_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and device.type == "cuda"))

    cfg = {
        "base_model": args.base_model, "max_length": args.max_length, "trunc": args.trunc,
        "head_frac": args.head_frac, "seed": args.seed, "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "amp": amp,
        "select_metric": args.select_metric, "threshold": 0.5,
    }
    history = {"step": [], "train_loss": [], "epochs": []}
    best, best_epoch, step = None, -1, 0
    t_train = time.perf_counter()
    n_seen = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        run_loss, run_n = 0.0, 0
        for i, batch in enumerate(tr_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx(device, amp):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), batch["labels"])
            scaler.scale(loss / args.grad_accum).backward()
            run_loss += loss.item() * len(logits); run_n += len(logits); n_seen += len(logits)
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(tr_loader):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update(); sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                history["step"].append(step); history["train_loss"].append(run_loss / run_n)
                run_loss, run_n = 0.0, 0
        p_val, val_loss = predict_loader(model, va_loader, device, amp)
        thr = best_threshold(y_val, p_val)
        m = compute_metrics(y_val, p_val, thr)
        history["epochs"].append({"epoch": epoch, "step": step, "val_loss": val_loss, "val": m})
        improved = is_better(m, best, args.select_metric)
        print(f"epoch {epoch}: val_loss={val_loss:.4f} roc_auc={m['roc_auc']:.4f} f1={m['f1']:.4f} "
              f"(esik {thr:.3f}){'  * en iyi' if improved else ''}")
        if improved:
            best, best_epoch = m, epoch
            save_bundle(model, tokenizer, model_dir, {**cfg, "best_epoch": epoch})
    train_sec = time.perf_counter() - t_train
    print(f"egitim {train_sec:.1f} sn, {n_seen / train_sec:.2f} ornek/sn (epoch ici degerlendirme dahil)")

    # En iyi checkpoint'i yukle; esigi val'de sec, teste aynen uygula.
    del model, optim
    if device.type == "cuda":
        torch.cuda.empty_cache()
    det = CodeBertDetector.from_dir(model_dir, device=str(device))
    p_val = det.predict_proba([r["code"] for r in val], args.eval_batch_size)
    thr = best_threshold(y_val, p_val)
    val_metrics = compute_metrics(y_val, p_val, thr)
    cfg.update({"threshold": thr, "best_epoch": best_epoch, "val_metrics": val_metrics})
    with open(model_dir / "codebert_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    det.threshold = thr

    test = load_split(args.data_dir, "test", args.limit, args.seed)
    _, te_cut = encode(test, "test")
    y_test = np.array([r["label"] for r in test])
    t0 = time.perf_counter()
    p_test = det.predict_proba([r["code"] for r in test], args.eval_batch_size)
    infer_sec = time.perf_counter() - t0
    test_metrics = compute_metrics(y_test, p_test, thr)
    print(f"val  (esik {thr:.3f}): roc_auc={val_metrics['roc_auc']:.4f} f1={val_metrics['f1']:.4f}")
    print(f"test (esik {thr:.3f}): roc_auc={test_metrics['roc_auc']:.4f} f1={test_metrics['f1']:.4f} "
          f"acc={test_metrics['accuracy']:.4f}  [{len(test) / infer_sec:.1f} ornek/sn cikarim]")

    results = {
        "config": cfg, "val": val_metrics, "test": test_metrics,
        "test_per_source": per_source_metrics(test, p_test, thr),
        "truncated": {"train": tr_cut, "val": va_cut, "test": te_cut},
        "n": {"train": len(train), "val": len(val), "test": len(test)},
        "timing": {"train_sec": train_sec, "train_samples_per_sec": n_seen / train_sec,
                   "test_infer_samples_per_sec": len(test) / infer_sec, "device": str(device)},
        "history": history,
    }
    with open(out_dir / "test_predictions.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(test, p_test):
            f.write(json.dumps({"id": r["id"], "label": r["label"], "source": r["source"], "prob": float(p)}) + "\n")
    plot_curves(history, out_dir / "training_curve.png")

    if args.lime_n:
        lime_out = []
        for r in test[: args.lime_n]:
            exp = det.explain_one(r["code"], args.lime_features, args.lime_samples, seed=args.seed,
                                  batch_size=args.eval_batch_size)
            print(f"LIME {r['id']} (etiket {r['label']}, p={exp['prob']:.3f}): {exp['seconds']:.1f} sn, "
                  f"{exp['n_features']} ozellik, R2={exp['score']:.3f}")
            for name, w in exp["features"][:5]:
                print(f"    {name!r:<28} {w:+.4f}")
            (out_dir / f"lime_{r['id']}.html").write_text(
                explanation_html(exp, f"{r['id']} | etiket={r['label']} | {r['source']}"), encoding="utf-8")
            lime_out.append({"id": r["id"], "label": r["label"], "prob": exp["prob"], "features": exp["features"],
                             "score": exp["score"], "seconds": exp["seconds"], "n_features": exp["n_features"]})
        results["lime"] = lime_out

    if args.onnx:
        onnx_dir = model_dir / "onnx"
        fp32 = export_onnx(model_dir, onnx_dir / "model.onnx", opset=args.opset)
        cpu_det = CodeBertDetector.from_dir(model_dir, device="cpu")
        sample = [r["code"] for r in test[:16]]
        results["onnx"] = {"fp32": compare_onnx(cpu_det, fp32, sample)}
        if args.quantize:
            int8 = quantize_onnx(fp32, onnx_dir / "model.int8.onnx")
            results["onnx"]["int8"] = compare_onnx(cpu_det, int8, sample)
        for k, v in results["onnx"].items():
            print(f"ONNX {k}: {v['size_mb']:.1f} MB, max|dlogit|={v['max_abs_logit_diff']:.2e}, "
                  f"max|dp|={v['max_abs_prob_diff']:.2e}, etiket uyumu={v['label_agreement']:.3f}")

    save_metrics(out_dir / "metrics.json", results)
    print(f"Yazildi -> {model_dir} , {out_dir}")


if __name__ == "__main__":
    main()
