"""Karakter + token n-gram stilometrisi (yazar tanimada klasik taban model).

Girdi normalize edilmis koddur (tez.normalize.normalize_code; zaten
normalize edilmis kodda idempotenttir). Uc gorunum vardir:
  - "char":   karakter n-gram (analyzer='char', 2-5), TF-IDF, sublinear_tf;
              bosluk, girinti ve satir sonlari OLDUGU GIBI korunur.
  - "tok":    Python tokenize lekseleri (tanimlayici, anahtar kelime, operator,
              sayi, string) uzerinde 1-3 gram; girinti/satir sonu ozel jetonlar.
  - "struct": ayni lekseler, ancak tanimlayicilar NAME, sayilar NUM, stringler
              STR (onek + tirnak bicimi korunur, orn. STR:f") olarak soyutlanir;
              anahtar kelimeler ve operatorler kalir (1-4 gram). Tanimlayici
              iceriginden bagimsiz yapisal uslup.
Her gorunumun vektorleyicisi YALNIZCA egitim kumesinde kurulur (sozluk + IDF);
gorunumler satir bazli L2 normalize edilip yan yana eklenir.

Siniflandirici: L2 LogisticRegression (liblinear) ya da LinearSVC + sigmoid
kalibrasyon (CalibratedClassifierCV, egitim icinde 3 kat). Gorunum kombinasyonu,
siniflandirici ve C kucuk bir izgarada YALNIZCA val'de secilir (val ROC AUC,
esitlikte val logloss); esik best_threshold ile val'de. Son model yalnizca
train ile kurulur (esik o modelin val tahminlerinden). Test en sonda bir kez.
"""
from __future__ import annotations

import io
import itertools
import keyword
import re
import time
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.svm import LinearSVC

from tez.metrics import best_threshold
from tez.normalize import normalize_code
from tez.paths import MODELS, SEED

VIEWS = ("char", "tok", "struct")
MODEL_FILE = "ngram.joblib"
# Gorunum ayarlari: (ngram_range, min_df, max_features).
VIEW_CONFIG: dict[str, dict] = {
    "char": {"ngram_range": (2, 5), "min_df": 3, "max_features": 200_000},
    "tok": {"ngram_range": (1, 3), "min_df": 3, "max_features": 100_000},
    "struct": {"ngram_range": (1, 4), "min_df": 3, "max_features": 50_000},
}
# Varsayilan izgara: gorunum kombinasyonlari x siniflandirici x C.
VIEW_COMBOS: list[tuple[str, ...]] = [
    c for k in range(1, len(VIEWS) + 1) for c in itertools.combinations(VIEWS, k)]
CLF_GRID: dict[str, list[float]] = {"lr": [0.3, 1.0, 3.0, 10.0], "svm": [0.03, 0.1, 0.3, 1.0]}
MAX_STR = 40  # tok gorunumunde string jetonlari bu uzunlukta kesilir

_KW = set(keyword.kwlist) | set(getattr(keyword, "softkwlist", []))
_STR_PREFIX = re.compile(r"^([A-Za-z]*)('''|\"\"\"|'|\")")
_FALLBACK = re.compile(r"\w+|[^\w\s]")


# ---------------------------------------------------------------------------
# On isleme ve jetonlama (modul duzeyi: joblib ile picklelanabilir olmali)
# ---------------------------------------------------------------------------

def prep_code(code: str) -> str:
    """Egitimdeki normalizasyon; parse edilemezse kod oldugu gibi kullanilir."""
    norm = normalize_code(code)
    return norm if norm is not None else code


def _lexemes(code: str) -> list[tuple[int, str]]:
    """(tokenize turu, metin) listesi; jetonlanamazsa basit regex yedegi."""
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type in (tokenize.ENDMARKER, tokenize.COMMENT, tokenize.ENCODING):
                continue
            out.append((t.type, t.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [(tokenize.NAME if w[0].isalnum() or w[0] == "_" else tokenize.OP, w)
                for w in _FALLBACK.findall(code)]
    return out


_SPECIAL = {tokenize.NEWLINE: "<NL>", tokenize.NL: "<nl>", tokenize.INDENT: "<IND>",
            tokenize.DEDENT: "<DED>"}


def _str_shape(s: str) -> str:
    m = _STR_PREFIX.match(s)
    return f"STR:{m.group(1).lower()}{m.group(2)}" if m else "STR"


def tok_tokens(code: str) -> list[str]:
    """Somut lekseler: tanimlayicilar ve anahtar kelimeler oldugu gibi."""
    out = []
    for typ, s in _lexemes(code):
        if typ in _SPECIAL:
            out.append(_SPECIAL[typ])
        elif s.strip():
            # String/f-string parcalari kesilir; bosluklar n-gram ayiricisiyla karismasin.
            out.append(s[:MAX_STR].replace(" ", "\\x20").replace("\n", "\\n"))
    return out


def struct_tokens(code: str) -> list[str]:
    """Soyut lekseler: NAME / NUM / STR:<onek><tirnak>; anahtar kelime ve operator kalir."""
    out = []
    for typ, s in _lexemes(code):
        if typ in _SPECIAL:
            out.append(_SPECIAL[typ])
        elif typ == tokenize.NAME:
            out.append(s if s in _KW else "NAME")
        elif typ == tokenize.NUMBER:
            out.append("NUM")
        elif typ == tokenize.STRING:
            out.append(_str_shape(s))
        elif typ in (getattr(tokenize, "FSTRING_START", -1),):
            out.append(_str_shape(s) + "{")
        elif typ in (getattr(tokenize, "FSTRING_MIDDLE", -2),):
            continue
        elif typ in (getattr(tokenize, "FSTRING_END", -3),):
            out.append("}STR")
        elif s.strip():
            out.append(s)
    return out


def _identity(x):
    return x


def make_vectorizer(view: str, cfg: dict | None = None) -> TfidfVectorizer:
    c = {**VIEW_CONFIG[view], **(cfg or {})}
    common = dict(sublinear_tf=True, lowercase=False, dtype=np.float32,
                  ngram_range=tuple(c["ngram_range"]), min_df=c["min_df"],
                  max_features=c["max_features"])
    if view == "char":
        return TfidfVectorizer(analyzer="char", strip_accents=None, **common)
    tokfn = tok_tokens if view == "tok" else struct_tokens
    return TfidfVectorizer(analyzer="word", tokenizer=tokfn, preprocessor=_identity,
                           token_pattern=None, **common)


# ---------------------------------------------------------------------------
# Siniflandirici
# ---------------------------------------------------------------------------

def make_clf(kind: str, C: float):
    if kind == "lr":
        return LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=SEED)
    if kind == "svm":
        return CalibratedClassifierCV(LinearSVC(C=C, max_iter=5000, random_state=SEED),
                                      cv=3, method="sigmoid")
    raise ValueError(f"Bilinmeyen siniflandirici: {kind}")


def linear_weights(clf) -> tuple[np.ndarray, float]:
    """Karar fonksiyonunun (yaklasik) dogrusal agirliklari ve kesisimi.

    SVM+kalibrasyonda katlarin LinearSVC agirliklari ortalanir (sigmoid monoton
    oldugu icin isaret ve siralama yorumu gecerlidir; olcek yaklasiktir).
    """
    if isinstance(clf, CalibratedClassifierCV):
        ests = [c.estimator for c in clf.calibrated_classifiers_]
        return (np.mean([e.coef_[0] for e in ests], axis=0),
                float(np.mean([e.intercept_[0] for e in ests])))
    return np.asarray(clf.coef_[0]), float(clf.intercept_[0])


def grid_points(views_combos=None, clf_grid=None) -> list[dict]:
    views_combos = VIEW_COMBOS if views_combos is None else views_combos
    clf_grid = CLF_GRID if clf_grid is None else clf_grid
    return [{"views": tuple(v), "clf": k, "C": float(C)}
            for v in views_combos for k, Cs in clf_grid.items() for C in Cs]


def fit_views(train_codes: Sequence[str], views: Sequence[str], view_cfg: dict | None = None
              ) -> dict[str, TfidfVectorizer]:
    """Her gorunumun vektorleyicisini YALNIZCA egitim kodlariyla kurar."""
    view_cfg = view_cfg or {}
    out = {}
    for v in views:
        vec = make_vectorizer(v, view_cfg.get(v))
        vec.fit(train_codes)
        out[v] = vec
    return out


def stack(mats: dict[str, sp.spmatrix], views: Sequence[str]) -> sp.csr_matrix:
    return sp.hstack([mats[v] for v in views], format="csr")


def select_and_fit(train_codes: Sequence[str], y_tr, val_codes: Sequence[str], y_val,
                   grid: list[dict] | None = None, view_cfg: dict | None = None,
                   log_fn=print):
    """Izgarayi val'de tarar; en iyiyi (val AUC, esitlikte logloss) secip dondurur.

    Imzada test verisi YOKTUR. Kodlar prep_code'dan gecmis olmalidir.
    Donus: (NgramDetector, izgara kaydi, kaynak olcumleri).
    """
    grid = grid or grid_points()
    y_tr = np.asarray(y_tr, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    need = sorted({v for g in grid for v in g["views"]}, key=VIEWS.index)
    res: dict = {"vectorize_sec": {}, "vocab": {}}
    vecs = {}
    X_tr, X_val = {}, {}
    for v in need:
        t = time.time()
        vecs[v] = fit_views(train_codes, [v], view_cfg)[v]
        X_tr[v] = vecs[v].transform(train_codes)
        X_val[v] = vecs[v].transform(val_codes)
        res["vectorize_sec"][v] = round(time.time() - t, 2)
        res["vocab"][v] = len(vecs[v].vocabulary_)
        log_fn(f"  gorunum {v}: {res['vocab'][v]} n-gram, {res['vectorize_sec'][v]} sn")
    log, best = [], None
    for g in grid:
        t = time.time()
        clf = make_clf(g["clf"], g["C"])
        clf.fit(stack(X_tr, g["views"]), y_tr)
        p = clf.predict_proba(stack(X_val, g["views"]))[:, 1]
        auc = float(roc_auc_score(y_val, p)) if len(np.unique(y_val)) == 2 else float("nan")
        row = {"views": "+".join(g["views"]), "clf": g["clf"], "C": g["C"],
               "n_features": int(sum(X_tr[v].shape[1] for v in g["views"])),
               "val_roc_auc": auc, "val_logloss": float(log_loss(y_val, p, labels=[0, 1])),
               "fit_sec": round(time.time() - t, 2)}
        log.append(row)
        key = (-np.nan_to_num(auc, nan=-1.0), row["val_logloss"])
        if best is None or key < best[0]:
            best = (key, g, clf, p)
    _, g, clf, p_val = best
    thr = best_threshold(y_val, p_val)
    det = NgramDetector({v: vecs[v] for v in g["views"]}, clf, list(g["views"]), thr,
                        {"clf": g["clf"], "C": g["C"], "views": list(g["views"]),
                         "view_config": {v: {**VIEW_CONFIG[v], **(view_cfg or {}).get(v, {})}
                                         for v in g["views"]},
                         "seed": SEED, "selection": "val_roc_auc, esitlikte val_logloss"})
    return det, log, res


# ---------------------------------------------------------------------------
# Dedektor
# ---------------------------------------------------------------------------

@dataclass
class NgramDetector:
    """Egitilmis vektorleyiciler + dogrusal siniflandirici + val'de secilmis esik."""

    vectorizers: dict[str, TfidfVectorizer]
    clf: object
    views: list[str]
    threshold: float
    meta: dict = field(default_factory=dict)
    name: str = "ngram"

    def transform(self, codes: Sequence[str], prepped: bool = False) -> sp.csr_matrix:
        codes = list(codes) if prepped else [prep_code(c) for c in codes]
        return stack({v: self.vectorizers[v].transform(codes) for v in self.views}, self.views)

    def predict_proba(self, codes: Sequence[str], prepped: bool = False) -> np.ndarray:
        """Ham ya da normalize kod listesi -> P(AI)."""
        if len(codes) == 0:
            return np.zeros(0)
        return self.clf.predict_proba(self.transform(codes, prepped))[:, 1]

    def predict(self, codes: Sequence[str]) -> np.ndarray:
        return (self.predict_proba(codes) >= self.threshold).astype(int)

    def feature_names(self) -> list[str]:
        return [f"{v}:{n}" for v in self.views
                for n in self.vectorizers[v].get_feature_names_out()]

    def top_ngrams(self, k: int = 30) -> dict[str, list[tuple[str, float]]]:
        """Global en ayirt edici n-gramlar: agirliga gore AI (+) ve insan (-) yonu."""
        w, _ = linear_weights(self.clf)
        names = self.feature_names()
        order = np.argsort(w)
        return {"ai": [(names[i], float(w[i])) for i in order[::-1][:k]],
                "human": [(names[i], float(w[i])) for i in order[:k]]}

    def explain_one(self, code: str, top: int | None = 10) -> list[tuple[str, float, float]]:
        """Tek kod icin (n-gram, tf-idf, katki = agirlik x tf-idf); |katki|'ya gore azalan.

        Pozitif katki AI yonune, negatif insan yonune iter (log-odds olcegi; SVM'de
        yaklasik). Bicim XGBDetector.explain_one ile ayni: (ad, deger, katki).
        """
        x = self.transform([code]).tocsr()
        w, _ = linear_weights(self.clf)
        idx, val = x.indices, x.data
        contrib = w[idx] * val
        order = np.argsort(-np.abs(contrib))
        if top:
            order = order[:top]
        names = self.feature_names()
        return [(names[idx[j]], float(val[j]), float(contrib[j])) for j in order]

    def save(self, out_dir: str | Path) -> None:
        import json

        import joblib
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizers": self.vectorizers, "clf": self.clf, "views": self.views,
                     "threshold": float(self.threshold), "meta": self.meta},
                    out_dir / MODEL_FILE, compress=3)
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({**self.meta, "views": self.views, "threshold": float(self.threshold)},
                      f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load(cls, model_dir: str | Path) -> NgramDetector:
        import joblib
        path = Path(model_dir) / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} yok; once scripts/04b_train_ngram.py calistirilmali")
        d = joblib.load(path)
        return cls(d["vectorizers"], d["clf"], list(d["views"]), float(d["threshold"]),
                   d.get("meta", {}))


def explain_one(code: str, model_dir: str | Path | None = None,
                top: int | None = 10) -> list[tuple[str, float, float]]:
    """Kaydedilmis n-gram modeliyle tek kod aciklamasi (API/mobil icin)."""
    return _load_cached(str(model_dir or MODELS / "ngram")).explain_one(code, top)


_CACHE: dict[str, NgramDetector] = {}


def _load_cached(model_dir: str) -> NgramDetector:
    if model_dir not in _CACHE:
        _CACHE[model_dir] = NgramDetector.load(model_dir)
    return _CACHE[model_dir]


def peak_memory_mb() -> float:
    """Surecin tepe bellek kullanimi (MB); olculemezse nan."""
    try:
        import resource  # Unix
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float("nan")
        return pmc.PeakWorkingSetSize / 2**20
    except Exception:  # noqa: BLE001
        return float("nan")
