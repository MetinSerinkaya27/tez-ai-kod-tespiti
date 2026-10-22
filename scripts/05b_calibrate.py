"""CodeBERT/UniXcoder icin sicaklik olcekleme (temperature scaling).

Sicaklik YALNIZCA val kumesinde ogrenilir ve modelin codebert_config.json
dosyasina yazilir; esik ayni kararlari verecek sekilde yeni olcege tasinir.
Donusum monoton oldugu icin AUC ve 0/1 kararlar degismez; yalnizca olasiliklar
gercek isabet oranina yaklasir. Test kumesine dokunulmaz.

Kullanim: uv run python scripts/05b_calibrate.py --model-dir models/unixcoder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss

from tez.codebert import CONFIG_NAME, CodeBertDetector, fit_platt, rescale_threshold
from tez.dataset import load_jsonl
from tez.paths import VAL_JSONL


def ece(y, p, bins: int = 10) -> float:
    """Esit genislikte kovalarla beklenen kalibrasyon hatasi."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=VAL_JSONL)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    det = CodeBertDetector.from_dir(args.model_dir, device=args.device)
    cfg_path = args.model_dir / CONFIG_NAME
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    # Her zaman kalibrasyonsuz olasiliklardan baslanir (tekrar calistirmaya dayanikli).
    det.temperature, det.bias = 1.0, 0.0
    thr0 = float(cfg.get("threshold_uncalibrated", cfg["threshold"]))

    recs = load_jsonl(args.val)
    y = np.array([r["label"] for r in recs])
    p0 = det.predict_proba([r["code"] for r in recs])
    t, b = fit_platt(p0, y)
    d0 = np.log(np.clip(p0, 1e-7, 1 - 1e-7) / np.clip(1 - p0, 1e-7, 1))
    p1 = 1 / (1 + np.exp(-(d0 / t + b)))
    thr1 = rescale_threshold(thr0, t, b)
    assert ((p0 >= thr0) == (p1 >= thr1)).mean() > 0.999, "esik tasima kararlari degistirdi"

    cfg.update({
        "temperature": t, "calib_bias": b, "threshold_uncalibrated": thr0, "threshold": thr1,
        "calibration": {"set": "val", "n": int(len(y)),
                        "ece_before": ece(y, p0), "ece_after": ece(y, p1),
                        "brier_before": float(brier_score_loss(y, p0)),
                        "brier_after": float(brier_score_loss(y, p1))},
    })
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    c = cfg["calibration"]
    print(f"{args.model_dir}: T={t:.3f}, b={b:+.3f}, esik {thr0:.4f} -> {thr1:.4f}; "
          f"val ECE {c['ece_before']:.4f} -> {c['ece_after']:.4f}, "
          f"Brier {c['brier_before']:.4f} -> {c['brier_after']:.4f}")


if __name__ == "__main__":
    main()
