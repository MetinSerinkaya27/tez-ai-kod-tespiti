"""N uyeli topluluk (ensemble) dedektoru (xgb, codebert, ngram, ...).

Alt dedektorler (orn. xgb, codebert, ngram) ayri ayri egitilmis ve kaydedilmistir;
burada yalnizca BIRLESTIRICI ogrenilir. Iki birlestirici vardir:
  - "mean": olasiliklarin basit ortalamasi (parametresiz),
  - "logistic": N uyenin logit(p_i) ozellikleri uzerinde L2'li LogisticRegression
    (C sabit; olasiliklar [EPS, 1-EPS] araligina kirpilir).
Birlestirici agirliklari VE esik (best_threshold, F1) YALNIZCA dogrulama (val)
kumesinde secilir; test ve gercek dunya setinde sabit kalir.

Iyimserlik notu: alt modellerin esikleri ve erken durdurmasi da val'de
secildi. Birlestiriciyi ayni val'de ogrenmek val metriklerini iyimser yapar
(val artik "gorulmus" veridir); test kumesi hicbir asamada kullanilmadigi icin
test metrikleri temiz kalir. Agirliklarin kararliligi icin val uzerinde depo
bazli K katli cross-fitting yapilir (cross_fit).

Kayit (models/ensemble/ensemble.json): yontem, esikler, LR agirliklari ve alt
model yollari (models kokune gore goreli, mumkunse). Bicim uye sayisindan
bagimsizdir; eski iki uyeli kayitlar da aynen yuklenir.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from tez.metrics import best_threshold, compute_metrics
from tez.paths import SEED

EPS = 1e-6
DEFAULT_C = 1.0
METHODS = ("mean", "logistic")
MEMBER_KINDS = ("xgb", "codebert", "ngram")
CONFIG_NAME = "ensemble.json"
OPTIMISM_NOTE = (
    "Birlestirici agirliklari ve esik val'de ogrenildi; alt modellerin esik/erken "
    "durdurmasi da val'de secildigi icin val metrikleri iyimserdir. Test kumesi hicbir "
    "secimde kullanilmadi; test metrikleri temizdir."
)


# ---------------------------------------------------------------------------
# Saf fonksiyonlar
# ---------------------------------------------------------------------------

def clip_proba(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p) -> np.ndarray:
    q = clip_proba(p)
    return np.log(q) - np.log1p(-q)


def as_matrix(P) -> np.ndarray:
    """(n, k) olasilik matrisi; tek boyutlu girdi (n, 1) olur."""
    P = np.asarray(P, dtype=float)
    return P[:, None] if P.ndim == 1 else P


def combine_mean(P) -> np.ndarray:
    """Olasiliklarin basit ortalamasi."""
    return as_matrix(P).mean(axis=1)


def fit_logistic(P, y, C: float = DEFAULT_C) -> dict:
    """logit(P) uzerinde L2 LR (sklearn varsayilan cezasi); {coef, intercept, C} dondurur."""
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                            random_state=SEED)
    lr.fit(logit(as_matrix(P)), np.asarray(y, dtype=int))
    return {"coef": [float(c) for c in lr.coef_[0]], "intercept": float(lr.intercept_[0]),
            "C": float(C)}


def combine_logistic(P, params: dict) -> np.ndarray:
    z = logit(as_matrix(P)) @ np.asarray(params["coef"], dtype=float) + params["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def combine(P, method: str, params: dict | None = None) -> np.ndarray:
    if method == "mean":
        return combine_mean(P)
    if method == "logistic":
        return combine_logistic(P, params or {})
    raise ValueError(f"Bilinmeyen birlestirici: {method}")


def fit_combiners(P_val, y_val, C: float = DEFAULT_C) -> dict[str, dict]:
    """YALNIZCA val tahmin/etiketleriyle her iki birlestiriciyi ve esiklerini kurar.

    Imzada test verisi yoktur: test etiketleri birlestiriciye ulasamaz.
    """
    P_val = as_matrix(P_val)
    y_val = np.asarray(y_val, dtype=int)
    out: dict[str, dict] = {"mean": {}}
    out["logistic"] = fit_logistic(P_val, y_val, C)
    for m in METHODS:
        out[m]["threshold"] = best_threshold(y_val, combine(P_val, m, out[m]))
    return out


def cross_fit(P_val, y_val, groups: Sequence, n_folds: int = 5,
              C: float = DEFAULT_C) -> dict:
    """Val uzerinde depo bazli K katli cross-fitting.

    Her katta LR diger katlarda ogrenilir; agirliklarin kat arasi ortalama/std'si
    ve disarida-kalan (OOF) tahminlerle val metrikleri (esik her katin egitim
    kisminda secilir) raporlanir. Test kullanilmaz.
    """
    P_val = as_matrix(P_val)
    y_val = np.asarray(y_val, dtype=int)
    groups = np.asarray(groups)
    k = min(n_folds, len(np.unique(groups)))
    if k < 2:
        return {"n_folds": 0, "durum": "yetersiz depo"}
    oof = {m: np.full(len(y_val), np.nan) for m in METHODS}
    pred_oof = {m: np.zeros(len(y_val), dtype=int) for m in METHODS}
    folds = []
    for tr, te in GroupKFold(n_splits=k).split(P_val, y_val, groups):
        if len(np.unique(y_val[tr])) < 2:
            continue
        comb = fit_combiners(P_val[tr], y_val[tr], C)
        folds.append({"coef": comb["logistic"]["coef"],
                      "intercept": comb["logistic"]["intercept"],
                      "thr_logistic": comb["logistic"]["threshold"],
                      "thr_mean": comb["mean"]["threshold"], "n_test": int(len(te))})
        for m in METHODS:
            p = combine(P_val[te], m, comb[m])
            oof[m][te] = p
            pred_oof[m][te] = (p >= comb[m]["threshold"]).astype(int)
    if not folds:
        return {"n_folds": 0, "durum": "tek sinifli katlar"}
    coefs = np.array([f["coef"] for f in folds])
    inter = np.array([f["intercept"] for f in folds])
    res = {"n_folds": len(folds), "folds": folds,
           "coef_mean": coefs.mean(axis=0).tolist(),
           "coef_std": coefs.std(axis=0, ddof=1).tolist() if len(folds) > 1 else [float("nan")] * coefs.shape[1],
           "intercept_mean": float(inter.mean()),
           "intercept_std": float(inter.std(ddof=1)) if len(folds) > 1 else float("nan"),
           "oof": {}}
    ok = ~np.isnan(oof["mean"])
    for m in METHODS:
        # AUC esiksizdir; F1/accuracy kat-ici esiklerle elde edilen OOF 0/1 tahminden.
        met = compute_metrics(y_val[ok], oof[m][ok], 0.5)
        pred = pred_oof[m][ok]
        yt = y_val[ok]
        tp = int(np.sum((pred == 1) & (yt == 1)))
        fp = int(np.sum((pred == 1) & (yt == 0)))
        fn = int(np.sum((pred == 0) & (yt == 1)))
        res["oof"][m] = {"roc_auc": met["roc_auc"], "pr_auc": met["pr_auc"],
                         "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
                         "accuracy": float(np.mean(pred == yt)), "n": int(ok.sum())}
    return res


def agreement_stats(p_a, p_b, thr_a: float, thr_b: float) -> dict:
    """Iki modelin tahmin korelasyonu (olasilik ve logit) ve uyusmazlik orani."""
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    from scipy.stats import spearmanr

    def pear(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")

    return {"pearson_p": pear(p_a, p_b), "pearson_logit": pear(logit(p_a), logit(p_b)),
            "spearman": float(spearmanr(p_a, p_b).statistic) if len(p_a) > 1 else float("nan"),
            "disagreement": float(np.mean((p_a >= thr_a) != (p_b >= thr_b))),
            "n": int(len(p_a))}


def pairwise_agreement(P, names: Sequence[str], thresholds: Sequence[float]) -> dict[str, dict]:
    """Tum uye ciftleri icin agreement_stats; anahtar "a vs b"."""
    P = as_matrix(P)
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out[f"{names[i]} vs {names[j]}"] = agreement_stats(P[:, i], P[:, j],
                                                               thresholds[i], thresholds[j])
    return out


# ---------------------------------------------------------------------------
# Alt dedektor yukleme
# ---------------------------------------------------------------------------

def load_member(kind: str, model_dir: str | Path, device: str = "cpu"):
    """Alt dedektoru `predict_proba(codes)` + `threshold` arayuzuyle yukler."""
    model_dir = Path(model_dir)
    if kind == "xgb":
        from tez.robustness import XGBCodeDetector
        from tez.xgb_model import XGBDetector
        return XGBCodeDetector(XGBDetector.load(model_dir))
    if kind == "codebert":
        from tez.codebert import CodeBertDetector  # torch/transformers burada yuklenir
        return CodeBertDetector.from_dir(model_dir, device=device)
    if kind == "ngram":
        from tez.ngram_model import NgramDetector
        return NgramDetector.load(model_dir)
    raise ValueError(f"Bilinmeyen alt model turu: {kind}")


def member_exists(kind: str, model_dir: str | Path) -> bool:
    d = Path(model_dir)
    if kind == "xgb":
        return (d / "meta.json").exists() and (d / "model.json").exists()
    if kind == "codebert":
        return (d / "config.json").exists()
    if kind == "ngram":
        return (d / "ngram.joblib").exists()
    return False


def _rel(path: Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve())


# ---------------------------------------------------------------------------
# Topluluk dedektoru
# ---------------------------------------------------------------------------

@dataclass
class EnsembleDetector:
    """Alt dedektorler + val'de ogrenilmis birlestirici; arayuz 06/08 ile ayni."""

    members: dict[str, object]           # ad -> dedektor (predict_proba, threshold)
    specs: dict[str, dict]               # ad -> {"kind", "dir"}
    combiners: dict[str, dict]           # fit_combiners ciktisi
    method: str = "logistic"
    meta: dict = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.specs)

    @property
    def threshold(self) -> float:
        return float(self.combiners[self.method]["threshold"])

    def member_proba(self, codes: Sequence[str]) -> np.ndarray:
        """(n, k) alt model olasiliklari (sutun sirasi = names)."""
        codes = list(codes)
        if not codes:
            return np.zeros((0, len(self.names)))
        return np.column_stack([np.asarray(self.members[n].predict_proba(codes), dtype=float)
                                for n in self.names])

    def combine(self, P, method: str | None = None) -> np.ndarray:
        m = method or self.method
        return combine(P, m, self.combiners[m])

    def predict_proba(self, codes: Sequence[str]) -> np.ndarray:
        if len(codes) == 0:
            return np.zeros(0)
        return self.combine(self.member_proba(codes))

    def predict(self, codes: Sequence[str]) -> np.ndarray:
        return (self.predict_proba(codes) >= self.threshold).astype(int)

    @classmethod
    def fit(cls, members: dict[str, object], specs: dict[str, dict], P_val, y_val,
            method: str = "logistic", C: float = DEFAULT_C, meta: dict | None = None):
        """Onceden hesaplanmis val tahminleriyle (sutun sirasi = specs) kurar."""
        if method not in METHODS:
            raise ValueError(f"Bilinmeyen birlestirici: {method}")
        P_val = as_matrix(P_val)
        if P_val.shape[1] != len(specs):
            raise ValueError("P_val sutun sayisi alt model sayisina esit olmali")
        return cls(members, dict(specs), fit_combiners(P_val, y_val, C), method,
                   {**(meta or {}), "fit_split": "val", "not": OPTIMISM_NOTE})

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir.parent
        cfg = {"method": self.method, "threshold": self.threshold,
               "members": [{"name": n, "kind": s["kind"], "dir": _rel(Path(s["dir"]), base)}
                           for n, s in self.specs.items()],
               "combiners": self.combiners, "eps": EPS, "meta": self.meta}
        with open(out_dir / CONFIG_NAME, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cpu",
             loader: Callable[[str, Path, str], object] | None = None,
             method: str | None = None) -> EnsembleDetector:
        """Kayittan yukler; goreli alt model yollari ensemble dizininin ustune gore."""
        model_dir = Path(model_dir)
        path = model_dir / CONFIG_NAME
        if not path.exists():
            raise FileNotFoundError(f"{path} yok; once scripts/09_ensemble.py calistirilmali")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        loader = loader or load_member
        specs, members = {}, {}
        for m in cfg["members"]:
            d = Path(m["dir"])
            d = d if d.is_absolute() else model_dir.parent / d
            specs[m["name"]] = {"kind": m["kind"], "dir": str(d)}
            members[m["name"]] = loader(m["kind"], d, device)
        return cls(members, specs, cfg["combiners"], method or cfg["method"], cfg.get("meta", {}))
