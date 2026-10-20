"""Gercek dunya test seti (data/real_world) degerlendirmesi icin saf fonksiyonlar.

Ilke: esik modelin dogrulama kumesinde secilmis kayitli esigidir; bu sette
hicbir sekilde ayarlanmaz. Guven araliklari PROJE bazli cluster bootstrap'tir
(ayni projedeki fonksiyonlar ve ayni insan fonksiyonundan uretilen AI kodlari
bagimsiz degildir; AI kaydi eslestigi insan koduyla ayni projededir).

Alt kumeler:
- tum set,
- uretici (model) basina: o ureticinin AI kodlari + TUM insan kodlari,
- istem tipi basina (skeleton/natural): o tipin AI kodlari + tum insanlar,
- uretici x istem tipi: ayni mantik,
- proje basina insan dogru-negatif orani (TNR; proje ici Wilson GA),
- ezber kontrolu: AI <-> eslesik insan govde benzerligi (seq_ratio_body);
  esigi asanlar dahil/haric,
- donem kontrolu: f-string iceren / icermeyen kodlar (her iki sinif),
- uzunluk eslestirmeli alt orneklem (audit.length_matched_sample).
"""
from __future__ import annotations

import ast
import math
from collections.abc import Sequence

import numpy as np

from tez.audit import length_matched_sample, pair_similarity, percentiles
from tez.metrics import compute_metrics
from tez.paths import SEED
from tez.robustness import _summarize, bootstrap_samples, fast_metrics

MEMO_THRESHOLD = 0.8
CI_KEYS = ("roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall", "tnr", "brier")


# ---------------------------------------------------------------------------
# Kayitlarin hazirlanmasi
# ---------------------------------------------------------------------------

def has_fstring(code: str) -> int | None:
    """Kod f-string iceriyorsa 1, icermiyorsa 0; ayristirilamazsa None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return int(any(isinstance(n, ast.JoinedStr) for n in ast.walk(tree)))


def build_rows(human: Sequence[dict], ai: Sequence[dict]) -> list[dict]:
    """Insan + AI kayitlarini tek degerlendirme tablosuna cevirir.

    Her satir: uid (benzersiz), func_id (insan id'si), label, project, model,
    prompt_type, code, fstring, sim_body (yalniz AI; eslesik insan yoksa None).
    """
    by_id = {h["id"]: h for h in human}
    rows: list[dict] = []
    for h in human:
        rows.append({"uid": f"h:{h['id']}", "func_id": h["id"], "label": 0,
                     "project": h["project"], "model": "insan", "prompt_type": "insan",
                     "code": h["code"], "fstring": has_fstring(h["code"]), "sim_body": None})
    seen: set[str] = set()
    for a in ai:
        uid = f"a:{a['id']}:{a['model']}:{a['prompt_type']}"
        if uid in seen:
            raise ValueError(f"tekrarlanan AI kaydi: {uid}")
        seen.add(uid)
        h = by_id.get(a["id"])
        sim = pair_similarity(h["code"], a["code"])["seq_ratio_body"] if h else None
        # Proje insan kaydindan alinir (AI kaydindaki ile ayni olmali).
        project = h["project"] if h else a.get("project")
        rows.append({"uid": uid, "func_id": a["id"], "label": int(a.get("label", 1)),
                     "project": project, "model": a["model"], "prompt_type": a["prompt_type"],
                     "code": a["code"], "fstring": has_fstring(a["code"]), "sim_body": sim})
    return rows


# ---------------------------------------------------------------------------
# Metrikler + proje bazli cluster bootstrap
# ---------------------------------------------------------------------------

def _metrics_tnr(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    m = fast_metrics(y, p, thr)
    neg = y == 0
    m["tnr"] = float(np.mean(p[neg] < thr)) if neg.any() else float("nan")
    return m


def cluster_ci(y, p, threshold: float, groups: Sequence, n_boot: int = 1000,
               seed: int = SEED, alpha: float = 0.05) -> dict[str, dict]:
    """Proje bazli cluster bootstrap yuzdelik GA (esik sabit). CI_KEYS metrikleri."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    point = _metrics_tnr(y, p, threshold)
    draws = [_metrics_tnr(y[i], p[i], threshold)
             for i in bootstrap_samples(list(groups), len(y), n_boot, seed)]
    return _summarize(point, draws, alpha)


def subset_metrics(rows: Sequence[dict], p, threshold: float, mask, name: str,
                   n_boot: int = 1000, seed: int = SEED, **extra) -> dict:
    """Maskelenmis alt kume icin compute_metrics + proje bootstrap GA'lari (duz satir)."""
    mask = np.asarray(mask, dtype=bool)
    p = np.asarray(p, dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)[mask]
    ps = p[mask]
    groups = [r["project"] for r, m in zip(rows, mask) if m]
    out = {"subset": name, **extra, "n": int(mask.sum()), "n_ai": int((y == 1).sum()),
           "n_human": int((y == 0).sum()), "n_projects": len(set(groups))}
    if not mask.any():
        out["durum"] = "bos"
        return out
    cm = compute_metrics(y, ps, threshold)
    out.update({k: cm[k] for k in ("threshold", "accuracy", "precision", "recall", "f1",
                                   "roc_auc", "pr_auc", "tn", "fp", "fn", "tp")})
    ci = cluster_ci(y, ps, threshold, groups, n_boot, seed)
    for k in CI_KEYS:
        out[k] = ci[k]["point"]
        out[f"{k}_lo"] = ci[k]["lo"]
        out[f"{k}_hi"] = ci[k]["hi"]
    out["durum"] = "tamam"
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Binom oran icin Wilson %95 araligi."""
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def per_project_table(rows: Sequence[dict], p, threshold: float) -> list[dict]:
    """Proje basina insan TNR (Wilson GA, proje ici iid varsayimi) ve AI recall."""
    p = np.asarray(p, dtype=float)
    out = []
    for proj in sorted({r["project"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["project"] == proj]
        hum = [i for i in idx if rows[i]["label"] == 0]
        ai = [i for i in idx if rows[i]["label"] == 1]
        tn = int(sum(p[i] < threshold for i in hum))
        tp = int(sum(p[i] >= threshold for i in ai))
        lo, hi = wilson(tn, len(hum))
        out.append({"project": proj, "n_human": len(hum), "tn": tn,
                    "tnr": tn / len(hum) if hum else float("nan"), "tnr_lo": lo, "tnr_hi": hi,
                    "n_ai": len(ai), "tp": tp,
                    "ai_recall": tp / len(ai) if ai else float("nan"),
                    "mean_p_human": float(np.mean(p[hum])) if hum else float("nan")})
    return out


# ---------------------------------------------------------------------------
# Tum alt kumeler
# ---------------------------------------------------------------------------

def evaluate(rows: Sequence[dict], p, threshold: float, n_boot: int = 1000, seed: int = SEED,
             memo_threshold: float = MEMO_THRESHOLD, length_bin: float = 1) -> dict:
    """Tum raporu hesaplar. Esik disaridan gelir ve degistirilmez."""
    p = np.asarray(p, dtype=float)
    if len(p) != len(rows):
        raise ValueError("tahmin sayisi kayit sayisina esit degil")
    lab = np.array([r["label"] for r in rows], dtype=int)
    is_h, is_a = lab == 0, lab == 1
    models = sorted({r["model"] for r in rows if r["label"] == 1})
    ptypes = sorted({r["prompt_type"] for r in rows if r["label"] == 1})

    def sm(mask, name, **extra):
        return subset_metrics(rows, p, threshold, mask, name, n_boot, seed, **extra)

    def ai_where(**kv):
        return np.array([r["label"] == 1 and all(r[k] == v for k, v in kv.items())
                         for r in rows])

    res: dict = {"threshold": float(threshold), "n_boot": n_boot, "seed": seed,
                 "n": len(rows), "n_ai": int(is_a.sum()), "n_human": int(is_h.sum()),
                 "n_projects": len({r["project"] for r in rows}),
                 "models": models, "prompt_types": ptypes}
    res["overall"] = sm(np.ones(len(rows), bool), "tum")
    res["per_model"] = [sm(is_h | ai_where(model=m), f"uretici={m}", model=m) for m in models]
    res["per_prompt"] = [sm(is_h | ai_where(prompt_type=t), f"istem={t}", prompt_type=t)
                         for t in ptypes]
    res["model_x_prompt"] = [sm(is_h | ai_where(model=m, prompt_type=t), f"{m} x {t}",
                                model=m, prompt_type=t) for m in models for t in ptypes]
    res["per_project"] = per_project_table(rows, p, threshold)

    # Ezber kontrolu.
    sims = np.array([np.nan if r["sim_body"] is None else r["sim_body"] for r in rows], float)
    valid = is_a & ~np.isnan(sims)
    high = valid & (sims > memo_threshold)
    res["memorization"] = {
        "memo_threshold": memo_threshold,
        "n_ai_matched": int(valid.sum()),
        "n_ai_unmatched": int((is_a & np.isnan(sims)).sum()),
        "n_high": int(high.sum()),
        "frac_high": float(high.sum() / valid.sum()) if valid.any() else float("nan"),
        "dist": percentiles(sims[valid].tolist()),
        "per_model": {m: percentiles(sims[valid & ai_where(model=m)].tolist()) for m in models},
        "dahil": sm(np.ones(len(rows), bool), "ezber dahil"),
        "haric": sm(~high, f"ezber haric (sim>{memo_threshold} AI cikarildi)"),
        "yalniz_yuksek": sm(is_h | high, f"yalniz sim>{memo_threshold} AI + insanlar"),
    }

    # Donem kontrolu (f-string).
    fs = [r["fstring"] for r in rows]
    res["fstring"] = {
        "n_parse_fail": int(sum(f is None for f in fs)),
        "frac_fstring_human": _frac([f for f, r in zip(fs, rows) if r["label"] == 0]),
        "frac_fstring_ai": _frac([f for f, r in zip(fs, rows) if r["label"] == 1]),
        "with": sm(np.array([f == 1 for f in fs]), "f-string var"),
        "without": sm(np.array([f == 0 for f in fs]), "f-string yok"),
    }

    # Uzunluk eslestirmeli alt orneklem (id = uid; deterministik).
    keyed = [{"id": r["uid"], "label": r["label"], "code": r["code"]} for r in rows]
    keep = {k["id"] for k in length_matched_sample(keyed, bin_width=length_bin, seed=seed)}
    res["length_matched"] = sm(np.array([r["uid"] in keep for r in rows]),
                               f"uzunluk eslesik (kova {length_bin} satir)")
    return res


def _frac(vals: Sequence) -> float:
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else float("nan")


def all_subset_rows(res: dict) -> list[dict]:
    """CSV icin tum alt kume satirlari (duz)."""
    rows = [res["overall"], *res["per_model"], *res["per_prompt"], *res["model_x_prompt"]]
    m = res["memorization"]
    rows += [m["dahil"], m["haric"], m["yalniz_yuksek"]]
    rows += [res["fstring"]["with"], res["fstring"]["without"], res["length_matched"]]
    return rows
