import json

import numpy as np
import pytest

from tez.ensemble import (
    EPS,
    EnsembleDetector,
    agreement_stats,
    clip_proba,
    combine_mean,
    cross_fit,
    fit_combiners,
    logit,
)


class SahteDedektor:
    """Sabit kural: kod uzunluguna (karakter) bagli olasilik."""

    def __init__(self, olcek: float, esik: float = 0.5):
        self.olcek = olcek
        self.threshold = esik

    def predict_proba(self, codes):
        n = np.array([len(c) for c in codes], dtype=float)
        return 1.0 / (1.0 + np.exp(-(n - 20.0) / self.olcek))


class SabitDedektor:
    """Her kod icin onceden verilmis olasilik (kod = indeks metni)."""

    def __init__(self, p, esik=0.5):
        self.p = np.asarray(p, dtype=float)
        self.threshold = esik

    def predict_proba(self, codes):
        return self.p[[int(c) for c in codes]]


def veri(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    a = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, n), 0.01, 0.99)
    b = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.25, n), 0.01, 0.99)
    groups = [f"depo{i % 20}" for i in range(n)]
    return np.column_stack([a, b]), y, groups


SPECS = {"xgb": {"kind": "xgb", "dir": "x"}, "codebert": {"kind": "codebert", "dir": "c"}}


def test_kirpma_ve_logit_sonlu():
    p = np.array([0.0, 1.0, 0.5, -0.1, 1.1])
    q = clip_proba(p)
    assert q.min() == EPS and q.max() == 1 - EPS
    z = logit(p)
    assert np.all(np.isfinite(z))
    assert z[2] == pytest.approx(0.0)
    assert z[0] == pytest.approx(np.log(EPS / (1 - EPS)))


def test_ortalama_dogru():
    P = np.array([[0.2, 0.6], [1.0, 0.0], [0.3, 0.3]])
    np.testing.assert_allclose(combine_mean(P), [0.4, 0.5, 0.3])


def test_birlestirici_yalniz_valden_test_etiketi_etkisiz():
    P, y, _ = veri()
    val, test = slice(0, 120), slice(120, None)
    c1 = fit_combiners(P[val], y[val])
    # Test etiketleri degisir (tersine cevrilir): val'den ogrenilen hicbir sey degismez.
    y_test_a, y_test_b = y[test], 1 - y[test]
    c2 = fit_combiners(P[val], y[val])
    assert c1 == c2
    ens = EnsembleDetector(
        {"xgb": SabitDedektor(P[:, 0]), "codebert": SabitDedektor(P[:, 1])}, SPECS, c1)
    thr0 = ens.threshold
    p_test = ens.predict_proba([str(i) for i in range(120, 200)])
    from tez.metrics import compute_metrics
    m_a = compute_metrics(y_test_a, p_test, ens.threshold)
    m_b = compute_metrics(y_test_b, p_test, ens.threshold)
    assert m_a["threshold"] == m_b["threshold"] == thr0
    assert ens.combiners == c1
    # Val etiketleri degisirse ise agirliklar degisir (gercekten val'den ogreniyor).
    c3 = fit_combiners(P[val], 1 - y[val])
    assert c3["logistic"]["coef"] != c1["logistic"]["coef"]


def test_esik_testte_sabit_ve_valde_secilir():
    P, y, _ = veri()
    ens = EnsembleDetector.fit(
        {"xgb": SabitDedektor(P[:, 0]), "codebert": SabitDedektor(P[:, 1])},
        SPECS, P[:120], y[:120])
    from tez.metrics import best_threshold
    assert ens.threshold == best_threshold(y[:120], ens.combine(P[:120]))
    t = ens.threshold
    ens.predict([str(i) for i in range(120, 200)])
    assert ens.threshold == t
    assert ens.meta["fit_split"] == "val"


def test_lojistik_agirliklar_pozitif_ve_guclu_uye_baskin():
    P, y, _ = veri(n=600, seed=1)
    c = fit_combiners(P, y)["logistic"]
    assert c["coef"][0] > c["coef"][1] > 0


def test_save_load_turu(tmp_path):
    P, y, _ = veri()
    members = {"xgb": SahteDedektor(5.0), "codebert": SahteDedektor(10.0)}
    codes = ["x" * k for k in range(5, 45)]
    Pv = np.column_stack([members[n].predict_proba(codes) for n in ("xgb", "codebert")])
    yv = np.array([int(len(c) > 22) for c in codes])
    specs = {"xgb": {"kind": "xgb", "dir": str(tmp_path / "models" / "xgb")},
             "codebert": {"kind": "codebert", "dir": "D:/baska/codebert"}}
    ens = EnsembleDetector.fit(members, specs, Pv, yv)
    ens.save(tmp_path / "models" / "ensemble")
    cfg = json.loads((tmp_path / "models" / "ensemble" / "ensemble.json").read_text())
    assert cfg["members"][0]["dir"] == "xgb"  # models kokune goreli
    yuklenen = []

    def loader(kind, d, device):
        yuklenen.append((kind, str(d), device))
        return {"xgb": SahteDedektor(5.0), "codebert": SahteDedektor(10.0)}[kind]

    ens2 = EnsembleDetector.load(tmp_path / "models" / "ensemble", device="cpu", loader=loader)
    assert [k for k, _, _ in yuklenen] == ["xgb", "codebert"]
    assert yuklenen[0][1] == str(tmp_path / "models" / "xgb")
    assert ens2.threshold == ens.threshold
    assert ens2.combiners == ens.combiners
    np.testing.assert_allclose(ens2.predict_proba(codes), ens.predict_proba(codes))
    ens3 = EnsembleDetector.load(tmp_path / "models" / "ensemble", loader=loader, method="mean")
    np.testing.assert_allclose(ens3.predict_proba(codes), Pv.mean(axis=1))
    assert ens3.threshold == ens.combiners["mean"]["threshold"]


def test_load_yoksa_anlasilir_hata(tmp_path):
    with pytest.raises(FileNotFoundError, match="09_ensemble"):
        EnsembleDetector.load(tmp_path)


def test_cross_fit_kararlilik():
    P, y, g = veri(n=300)
    cf = cross_fit(P, y, g, n_folds=5)
    assert cf["n_folds"] == 5
    assert len(cf["coef_mean"]) == 2 and all(s >= 0 for s in cf["coef_std"])
    assert set(cf["oof"]) == {"mean", "logistic"}
    assert cf["oof"]["logistic"]["n"] == 300
    # Kat test depolari egitim depolariyla kesismez (GroupKFold): her depo tek katta.
    assert sum(f["n_test"] for f in cf["folds"]) == 300


def test_uyum_istatistikleri():
    a = np.array([0.1, 0.9, 0.8, 0.2])
    s = agreement_stats(a, a, 0.5, 0.5)
    assert s["pearson_p"] == pytest.approx(1.0) and s["disagreement"] == 0.0
    s2 = agreement_stats(a, 1 - a, 0.5, 0.5)
    assert s2["disagreement"] == 1.0 and s2["pearson_p"] == pytest.approx(-1.0)


def test_bos_girdi_ve_hatali_yontem():
    ens = EnsembleDetector({"xgb": SahteDedektor(5.0), "codebert": SahteDedektor(5.0)}, SPECS,
                           {"mean": {"threshold": 0.5}, "logistic": {"coef": [1, 1], "intercept": 0,
                                                                     "threshold": 0.5}})
    assert ens.predict_proba([]).shape == (0,)
    with pytest.raises(ValueError):
        EnsembleDetector.fit({}, SPECS, np.zeros((4, 2)), [0, 1, 0, 1], method="medyan")
