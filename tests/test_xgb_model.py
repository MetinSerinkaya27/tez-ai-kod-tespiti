import random

import numpy as np
import pandas as pd
import pytest

from tez.features import FEATURE_NAMES, features_frame
from tez.metrics import best_threshold
from tez.xgb_model import (
    ERA_COLUMNS,
    GROUPS,
    LENGTH_BASE,
    RAW_COUNTS,
    XGBDetector,
    ablation_sets,
    cached_features,
    length_columns,
    per_source_metrics,
    select_and_fit,
)

KUCUK_IZGARA = {"max_depth": [2], "learning_rate": [0.3], "min_child_weight": [1]}


def _kod(label: int, i: int, rng: random.Random) -> str:
    """Iki uslupta sentetik fonksiyon: 0 tek tirnak/sikisik, 1 cift tirnak/bosluklu."""
    n = rng.randint(1, 4)
    if label == 0:
        body = "".join(f"    x{k}=a+{k}*'s'\n" for k in range(n))
        return f"def f{i}(a):\n{body}    return x0\n"
    body = "".join(f"    value_{k} = a + {k} * \"s\"\n" for k in range(n))
    return f"def f{i}(a):\n{body}    return value_0\n"


def _veri(n: int, seed: int) -> tuple[list[dict], pd.DataFrame, np.ndarray]:
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        label = i % 2
        recs.append({"id": f"{seed}_{i}", "code": _kod(label, i, rng), "label": label,
                     "source": "model-a" if label else "human"})
    X = features_frame(r["code"] for r in recs)
    y = np.array([r["label"] for r in recs])
    return recs, X, y


@pytest.fixture(scope="module")
def egitilmis():
    _, X_tr, y_tr = _veri(40, 1)
    _, X_val, y_val = _veri(20, 2)
    cols = list(FEATURE_NAMES)
    model, params, log = select_and_fit(X_tr[cols], y_tr, X_val[cols], y_val,
                                        grid=KUCUK_IZGARA, n_jobs=1)
    det = XGBDetector(model, cols, 0.5)
    det.threshold = best_threshold(y_val, det.predict_proba(X_val))
    return det, params, log


def test_egitim_tahmin_kaydet_yukle(egitilmis, tmp_path):
    det, params, log = egitilmis
    assert len(log) == 1 and params["n_estimators"] == log[0]["best_iteration"] + 1
    _, X_te, y_te = _veri(10, 3)
    p = det.predict_proba(X_te)
    assert p.shape == (10,) and ((p >= 0) & (p <= 1)).all()
    # Uslup farki belirgin oldugu icin sentetik testte ayirabilmeli.
    assert (det.predict(X_te) == y_te).mean() >= 0.9

    det.save(tmp_path)
    yuklu = XGBDetector.load(tmp_path)
    assert yuklu.features == det.features
    assert yuklu.threshold == det.threshold
    np.testing.assert_allclose(yuklu.predict_proba(X_te), p, rtol=1e-6)


def test_esik_testte_degismez(egitilmis):
    det, _, _ = egitilmis
    esik = det.threshold
    recs, X_te, y_te = _veri(10, 4)
    p = det.predict_proba(X_te)
    rows = per_source_metrics(y_te, p, [r["source"] for r in recs], esik)
    assert det.threshold == esik
    assert {r["threshold"] for r in rows} == {esik}
    assert [r["source"] for r in rows] == ["tum", "model-a", "human"]
    assert rows[-1]["n"] == 5 and rows[-1]["tp"] + rows[-1]["fn"] == 0


def test_ablasyon_kolonlari():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    X["fmt_num_lines"] = np.arange(50)
    X["local_name_count"] = np.arange(50) * 2 + rng.random(50)  # |r| ~ 1
    X["param_count"] = 3.0  # sabit -> r tanimsiz -> tutulur
    cols, max_r = length_columns(X)
    assert cols[: len(LENGTH_BASE)] == LENGTH_BASE
    assert "local_name_count" in cols and "param_count" not in cols
    assert set(max_r) == set(RAW_COUNTS) and max_r["param_count"] == 0.0

    sets = ablation_sets(cols)
    assert sets["tum"] == FEATURE_NAMES
    assert not set(sets["uzunluksuz"]) & set(cols)
    assert set(sets["uzunluksuz"]) | set(cols) == set(FEATURE_NAMES)
    assert all(c.startswith("fmt_") for c in sets["yalniz_bicim"])
    assert sets["yalniz_bicim"] == [c for c in FEATURE_NAMES if c.startswith("fmt_")]
    assert not any(c.startswith("fmt_") for c in sets["yalniz_ast"])
    assert sorted(sets["yalniz_bicim"] + sets["yalniz_ast"]) == sorted(FEATURE_NAMES)
    assert set(ERA_COLUMNS) <= set(FEATURE_NAMES)
    assert not set(sets["donemsiz"]) & set(ERA_COLUMNS)
    assert set(sets["donemsiz"]) | set(ERA_COLUMNS) == set(FEATURE_NAMES)
    assert not set(sets["uzunluksuz_donemsiz"]) & (set(ERA_COLUMNS) | set(cols))
    # Gruplar FEATURE_NAMES'i ortusmeden boler.
    grup_kolonlari = [c for g in GROUPS.values() for c in g]
    assert sorted(grup_kolonlari) == sorted(FEATURE_NAMES)


def test_explain_one_sekli(egitilmis):
    det, _, _ = egitilmis
    kod = 'def g(a):\n    """Belge."""\n    value_0 = a + 1 * "s"  # yorum\n    return value_0\n'
    out = det.explain_one(kod)
    assert len(out) == len(det.features)
    assert all(isinstance(f, str) and isinstance(v, float) and isinstance(s, float)
               for f, v, s in out)
    assert [abs(s) for _, _, s in out] == sorted((abs(s) for _, _, s in out), reverse=True)
    assert len(det.explain_one(kod, top=5)) == 5

    # Katkilar + taban = modelin log-odds ciktisi (TreeSHAP yerel dogrulugu).
    X = det.code_frame(kod)
    sv, base = det.shap_values(X)
    p = det.predict_proba(X)[0]
    assert sv.sum() + base[0] == pytest.approx(np.log(p / (1 - p)), abs=1e-4)


def test_explain_one_shap_kutuphanesiyle_ayni(egitilmis):
    shap = pytest.importorskip("shap")
    det, _, _ = egitilmis
    _, X, _ = _veri(6, 5)
    ours, _ = det.shap_values(X)
    theirs = shap.TreeExplainer(det.model)(X[det.features]).values
    np.testing.assert_allclose(ours, theirs, atol=1e-4)


def test_parse_edilemeyen_kod(egitilmis):
    with pytest.raises(ValueError):
        egitilmis[0].explain_one("def f(:\n")


def test_ozellik_onbellegi(tmp_path):
    recs, X, _ = _veri(6, 6)
    a = cached_features(recs, tmp_path, "train")
    assert list(a.index) == [r["id"] for r in recs]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), X)
    assert len(list(tmp_path.glob("train_*.parquet"))) == 1
    ilk = next(tmp_path.glob("train_*.parquet"))

    recs[0] = {**recs[0], "code": "def h():\n    return 1\n"}
    b = cached_features(recs, tmp_path, "train")
    dosyalar = list(tmp_path.glob("train_*.parquet"))
    assert len(dosyalar) == 1 and dosyalar[0] != ilk
    assert b.iloc[0]["num_statements"] == 1


def test_ablasyon_ppl_istege_bagli():
    from tez.perplexity import PPL_FEATURES

    cols = list(LENGTH_BASE)
    eski = ablation_sets(cols)
    yeni = ablation_sets(cols, PPL_FEATURES)
    # ppl verilmezse ek satir yok; verilirse eski ablasyonlar aynen ve ayni sirada.
    assert not any("ppl" in k for k in eski)
    assert list(yeni)[: len(eski)] == list(eski)
    assert all(yeni[k] == eski[k] for k in eski)
    assert yeni["tum+ppl"] == list(FEATURE_NAMES) + PPL_FEATURES
    assert yeni["yalniz_ppl"] == PPL_FEATURES
    assert yeni["donemsiz+ppl"] == eski["donemsiz"] + PPL_FEATURES
    assert not set(yeni["donemsiz+ppl"]) & set(ERA_COLUMNS)


def test_ppl_olmayan_modelde_lm_yuklenmez(egitilmis, monkeypatch):
    from tez import perplexity

    def yasak(*a, **k):
        raise AssertionError("ppl ozelligi olmayan modelde LM yuklenmemeli")

    monkeypatch.setattr(perplexity, "load_lm", yasak)
    det = egitilmis[0]
    assert len(det.explain_one("def g(a):\n    return a\n")) == len(det.features)
