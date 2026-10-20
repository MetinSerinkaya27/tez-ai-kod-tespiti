"""AST stilometri ozellikleri uzerinde XGBoost siniflandirici, ablasyon ve SHAP.

Kurallar: hiperparametre (kucuk izgara + erken durdurma) ve esik yalnizca
dogrulama kumesinde secilir; test kumesi en sonda bir kez, sabit esikle
degerlendirilir. Ozellik matrisleri bolme basina parquet olarak onbelleklenir;
ozellik kodu ya da veri degisirse hash degisir ve yeniden hesaplanir.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tez import features as _features_mod
from tez.features import _IDENT, _STRUCT, FEATURE_NAMES, features_frame
from tez.metrics import best_threshold, compute_metrics
from tez.normalize import normalize_code
from tez.paths import MODELS, SEED

# Uzunlugun dogrudan olcusu olan kolonlar (uzunluksuz ablasyonda hep cikarilir).
LENGTH_BASE = ["num_nodes", "num_statements", "fmt_num_lines", "cyclomatic_complexity"]
# Uzunluga gore normalize edilmemis ham sayim/boyut kolonlari; bunlardan egitim
# kumesinde LENGTH_BASE ile |r| > esik olanlar da cikarilir.
RAW_COUNTS = [
    "max_depth", "max_block_nesting", "nested_function_count", "return_count",
    "early_return_count", "param_count", "local_name_count", "local_name_max_len",
    "fmt_line_len_max",
]
LENGTH_CORR_THRESHOLD = 0.8

# Ozellik gruplari (features.py'deki dort grup; AST yapisi = node_* + _STRUCT).
GROUPS: dict[str, list[str]] = {
    "ast_yapi": [c for c in FEATURE_NAMES if c.startswith("node_")] + list(_STRUCT),
    "tanimlayici": list(_IDENT),
    "deyim": [c for c in FEATURE_NAMES if c.startswith("idiom_")],
    "bicim": [c for c in FEATURE_NAMES if c.startswith("fmt_")],
}

BASELINE_FEATURES = ["fmt_num_lines"]

FIXED_PARAMS = {
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}
GRID: dict[str, list] = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [1, 5],
}


# --- Ozellik onbellegi ---

def feature_hash(records: list[dict]) -> str:
    """Ozellik kodu, ozellik listesi ve (id, kod) iceriginin hash'i."""
    h = hashlib.sha256()
    h.update(Path(_features_mod.__file__).read_bytes())
    h.update(json.dumps(FEATURE_NAMES).encode())
    for r in records:
        h.update(r["id"].encode() + b"\0" + r["code"].encode() + b"\0")
    return h.hexdigest()[:16]


def cached_features(records: list[dict], cache_dir: str | Path, name: str) -> pd.DataFrame:
    """Kayitlarin ozellik matrisi (index = id); hash uyusursa parquet'ten okunur."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{name}_{feature_hash(records)}.parquet"
    ids = [r["id"] for r in records]
    if path.exists():
        X = pd.read_parquet(path)
        if list(X.index) == ids and list(X.columns) == FEATURE_NAMES:
            return X
    X = features_frame(r["code"] for r in records)
    X.index = pd.Index(ids, name="id")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob(f"{name}_*.parquet"):
        old.unlink()
    X.to_parquet(path)
    return X


# --- Ablasyon kolonlari ---

def length_columns(X_train: pd.DataFrame, threshold: float = LENGTH_CORR_THRESHOLD) -> tuple[list[str], dict[str, float]]:
    """Uzunluk kolonlari: LENGTH_BASE + egitimde onlarla |r| > threshold ham sayimlar.

    Ikinci deger her aday ham sayim icin LENGTH_BASE ile en buyuk |r|.
    Sabit kolonlarda (r tanimsiz) korelasyon 0 kabul edilir.
    """
    corr = X_train[LENGTH_BASE + RAW_COUNTS].corr(method="pearson").abs().fillna(0.0)
    max_r = {c: float(corr.loc[c, LENGTH_BASE].max()) for c in RAW_COUNTS}
    extra = [c for c in RAW_COUNTS if max_r[c] > threshold]
    return LENGTH_BASE + extra, max_r


def ablation_sets(length_cols: list[str]) -> dict[str, list[str]]:
    """Ablasyon adi -> kolon listesi (FEATURE_NAMES sirasinda)."""
    drop = set(length_cols)
    fmt = set(GROUPS["bicim"])
    return {
        "tum": list(FEATURE_NAMES),
        "uzunluksuz": [c for c in FEATURE_NAMES if c not in drop],
        "yalniz_bicim": [c for c in FEATURE_NAMES if c in fmt],
        "yalniz_ast": [c for c in FEATURE_NAMES if c not in fmt],
    }


# --- Egitim ---

def grid_points(grid: dict[str, list] | None = None) -> list[dict]:
    grid = GRID if grid is None else grid
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _make_model(params: dict, n_jobs: int | None, **override) -> xgb.XGBClassifier:
    p = {**FIXED_PARAMS, **params, **override}
    return xgb.XGBClassifier(**p, random_state=SEED, n_jobs=n_jobs)


def select_and_fit(
    X_tr: pd.DataFrame, y_tr, X_val: pd.DataFrame, y_val,
    grid: dict[str, list] | None = None, n_jobs: int | None = None,
) -> tuple[xgb.XGBClassifier, dict, list[dict]]:
    """Izgarayi val logloss'una (erken durdurma ile) gore tarar, en iyiyi yeniden kurar.

    Son model en iyi iterasyon sayisiyla erken durdurmasiz egitilir (ayni seed ile
    ayni agaclar); boylece kaydedilen modelde kullanilmayan agac kalmaz.
    Donus: (model, secilen parametreler + n_estimators, izgara kaydi).
    """
    log = []
    best = None
    for params in grid_points(grid):
        m = _make_model(params, n_jobs)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        row = {**params, "best_iteration": int(m.best_iteration), "val_logloss": float(m.best_score)}
        log.append(row)
        if best is None or row["val_logloss"] < best["val_logloss"]:
            best = row
    chosen = {k: best[k] for k in (grid or GRID)}
    chosen["n_estimators"] = best["best_iteration"] + 1
    final = _make_model(chosen, n_jobs, early_stopping_rounds=None)
    final.fit(X_tr, y_tr, verbose=False)
    return final, chosen, log


def fit_length_baseline(X_tr: pd.DataFrame, y_tr):
    """Yalnizca satir sayisiyla lojistik regresyon (taban cizgisi)."""
    lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED))
    lr.fit(X_tr[BASELINE_FEATURES], y_tr)
    return lr


# --- Model sarmalayici ---

@dataclass
class XGBDetector:
    """Egitilmis model + kolon listesi + val'de secilmis sabit esik."""

    model: xgb.XGBClassifier
    features: list[str]
    threshold: float
    meta: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def code_frame(self, code: str) -> pd.DataFrame:
        """Ham kodu egitimdeki gibi normalize edip tek satirlik ozellik tablosu yapar."""
        norm = normalize_code(code)
        if norm is None:
            raise ValueError("Kod parse edilemedi")
        return features_frame([norm])[self.features]

    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """TreeSHAP katkilari (log-odds) ve taban degerleri: (n, f), (n,)."""
        dm = xgb.DMatrix(X[self.features])
        contrib = self.model.get_booster().predict(dm, pred_contribs=True)
        return contrib[:, :-1], contrib[:, -1]

    def explain_one(self, code: str, top: int | None = None) -> list[tuple[str, float, float]]:
        """Tek kod icin (ozellik, deger, shap) listesi; |shap|'e gore azalan."""
        X = self.code_frame(code)
        sv, _ = self.shap_values(X)
        items = [(f, float(X.iloc[0][f]), float(s)) for f, s in zip(self.features, sv[0])]
        items.sort(key=lambda t: -abs(t[2]))
        return items[:top] if top else items

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(out_dir / "model.json")
        meta = {**self.meta, "features": self.features, "threshold": self.threshold}
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, model_dir: str | Path) -> XGBDetector:
        model_dir = Path(model_dir)
        with open(model_dir / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        model = xgb.XGBClassifier()
        model.load_model(model_dir / "model.json")
        features = meta.pop("features")
        threshold = float(meta.pop("threshold"))
        return cls(model, features, threshold, meta)


@lru_cache(maxsize=4)
def _load_cached(model_dir: str) -> XGBDetector:
    return XGBDetector.load(model_dir)


def explain_one(code: str, model_dir: str | Path | None = None,
                top: int | None = None) -> list[tuple[str, float, float]]:
    """Kaydedilmis ana modelle tek kod aciklamasi (API/mobil icin)."""
    return _load_cached(str(model_dir or MODELS / "xgb")).explain_one(code, top)


# --- Raporlama ---

def per_source_metrics(y, prob, sources, threshold: float) -> list[dict]:
    """Kaynak basina metrikler: her AI kaynagi tum insan ornekleriyle birlikte.

    "human" satiri yalnizca insan orneklerini icerir (accuracy = dogru negatif orani).
    """
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    sources = np.asarray(sources)
    human = sources == "human"
    rows = [{"source": "tum", **compute_metrics(y, prob, threshold)}]
    for s in sorted(set(sources) - {"human"}):
        mask = human | (sources == s)
        rows.append({"source": s, **compute_metrics(y[mask], prob[mask], threshold)})
    if human.any():
        rows.append({"source": "human", **compute_metrics(y[human], prob[human], threshold)})
    return rows

