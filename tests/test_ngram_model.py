import random

import numpy as np
import pytest

from tez.metrics import best_threshold
from tez.ngram_model import (
    NgramDetector,
    grid_points,
    prep_code,
    select_and_fit,
    struct_tokens,
    tok_tokens,
)

# Hizli izgara: iki gorunum kombinasyonu x LR iki C.
KUCUK = grid_points([("char",), ("tok", "struct")], {"lr": [1.0, 10.0]})
KUCUK_CFG = {v: {"min_df": 1, "max_features": 5000} for v in ("char", "tok", "struct")}


def _kod(label: int, i: int, rng: random.Random) -> str:
    """Iki uslup: 0 tek tirnak/sikisik/%-bicim, 1 cift tirnak/bosluklu/f-string."""
    n = rng.randint(1, 3)
    if label == 0:
        body = "".join(f"    x{k}=a+{k}*'s%d'%b\n" for k in range(n))
        return f"def f{i}(a,b):\n{body}    return x0\n"
    body = "".join(f"    value_{k} = a + {k} * f\"s{{b}}\"\n" for k in range(n))
    return f"def g{i}(a, b):\n{body}\n    return value_0\n"


def _veri(n=60, seed=0):
    rng = random.Random(seed)
    y = [i % 2 for i in range(n)]
    return [prep_code(_kod(lab, i, rng)) for i, lab in enumerate(y)], np.array(y)


@pytest.fixture(scope="module")
def egitilmis():
    tr, ytr = _veri(60, 0)
    va, yva = _veri(30, 1)
    det, log, res = select_and_fit(tr, ytr, va, yva, KUCUK, KUCUK_CFG, log_fn=lambda *_: None)
    return det, log, res, (tr, ytr, va, yva)


def test_jetonlar_soyut_ve_somut():
    c = "def f(a):\n    return a + 1 * 'x y'\n"
    t, s = tok_tokens(c), struct_tokens(c)
    assert "f" in t and "return" in t and "'x\\x20y'" in t
    assert "NAME" in s and "NUM" in s and "STR:'" in s and "return" in s
    assert "f" not in s and "a" not in s
    assert all(" " not in x for x in t + s)


def test_prep_idempotent():
    ham = "def f(a):\n    '''doc'''\n    # yorum\n    return a\n"
    n1 = prep_code(ham)
    assert "doc" not in n1 and "yorum" not in n1
    assert prep_code(n1) == n1
    assert prep_code("def (:") == "def (:"  # parse edilemezse oldugu gibi


def test_egitim_tahmin_ve_ayrim(egitilmis):
    det, log, res, (_, _, va, yva) = egitilmis
    assert len(log) == len(KUCUK)
    p = det.predict_proba(va, prepped=True)
    assert p.shape == (len(va),) and np.all((p >= 0) & (p <= 1))
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(yva, p) > 0.9
    # Esik val'de best_threshold ile secilir.
    assert det.threshold == best_threshold(yva, p)
    assert det.predict_proba([]).shape == (0,)


def test_ham_kod_normalize_edilir(egitilmis):
    det = egitilmis[0]
    ham = "def g1(a, b):\n    '''aciklama'''\n    value_0 = a + 0 * f\"s{b}\"  # not\n\n    return value_0\n"
    np.testing.assert_allclose(det.predict_proba([ham]), det.predict_proba([prep_code(ham)]))


def test_kaydet_yukle(egitilmis, tmp_path):
    det, _, _, (_, _, va, _) = egitilmis
    det.save(tmp_path / "ngram")
    det2 = NgramDetector.load(tmp_path / "ngram")
    assert det2.threshold == det.threshold and det2.views == det.views
    np.testing.assert_allclose(det2.predict_proba(va), det.predict_proba(va))
    with pytest.raises(FileNotFoundError, match="04b_train_ngram"):
        NgramDetector.load(tmp_path / "yok")


def test_esik_testte_sabit(egitilmis):
    det = egitilmis[0]
    t = det.threshold
    te, yte = _veri(20, 7)
    det.predict(te)
    from tez.metrics import compute_metrics
    p = det.predict_proba(te)
    assert compute_metrics(yte, p, det.threshold)["threshold"] == t
    assert compute_metrics(1 - yte, p, det.threshold)["threshold"] == t
    assert det.threshold == t


def test_secim_test_etiketlerinden_bagimsiz():
    tr, ytr = _veri(60, 0)
    va, yva = _veri(30, 1)
    sus = dict(log_fn=lambda *_: None)
    d1, l1, _ = select_and_fit(tr, ytr, va, yva, KUCUK, KUCUK_CFG, **sus)
    # Test kumesi secime hic girmez: ayni train/val -> ayni secim ve esik.
    d2, l2, _ = select_and_fit(tr, ytr, va, yva, KUCUK, KUCUK_CFG, **sus)
    def puan(log):
        return [{k: v for k, v in r.items() if k != "fit_sec"} for r in log]

    assert d1.meta == d2.meta and d1.threshold == d2.threshold and puan(l1) == puan(l2)
    # Val etiketleri bozulursa izgara puanlari degisir (secim gercekten val'den).
    rng = np.random.default_rng(0)
    d3, l3, _ = select_and_fit(tr, ytr, va, rng.permutation(yva), KUCUK, KUCUK_CFG, **sus)
    assert [r["val_roc_auc"] for r in l3] != [r["val_roc_auc"] for r in l1]
    import inspect
    assert not any("test" in p for p in inspect.signature(select_and_fit).parameters)


def test_explain_one_sekli(egitilmis):
    det = egitilmis[0]
    kod = _kod(1, 99, random.Random(3))
    ex = det.explain_one(kod, top=5)
    assert 0 < len(ex) <= 5
    assert all(isinstance(n, str) and ":" in n and isinstance(v, float) and isinstance(c, float)
               for n, v, c in ex)
    assert [abs(c) for _, _, c in ex] == sorted((abs(c) for _, _, c in ex), reverse=True)
    assert ex[0][0].split(":", 1)[0] in det.views
    tam = det.explain_one(kod, top=None)
    assert len(tam) >= len(ex)
    top = det.top_ngrams(3)
    assert len(top["ai"]) == 3 and len(top["human"]) == 3
    assert top["ai"][0][1] >= top["ai"][-1][1] and top["human"][0][1] <= top["human"][-1][1]


def test_svm_kalibre():
    tr, ytr = _veri(60, 0)
    va, yva = _veri(30, 1)
    det, _, _ = select_and_fit(tr, ytr, va, yva, grid_points([("struct",)], {"svm": [0.1]}),
                               KUCUK_CFG, log_fn=lambda *_: None)
    p = det.predict_proba(va, prepped=True)
    assert np.all((p >= 0) & (p <= 1))
    assert det.explain_one(va[0], top=3)
