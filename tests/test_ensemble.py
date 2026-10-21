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


def test_n_uyeli_topluluk_uc_agirlik(tmp_path):
    rng = np.random.default_rng(3)
    n = 300
    y = rng.integers(0, 2, n)
    P = np.column_stack([np.clip(0.5 + s * (y - 0.5) + rng.normal(0, 0.2, n), 0.01, 0.99)
                         for s in (0.4, 0.2, 0.05)])
    specs = {k: {"kind": k, "dir": str(tmp_path / "models" / k)}
             for k in ("xgb", "codebert", "ngram")}
    members = {k: SabitDedektor(P[:, i]) for i, k in enumerate(specs)}
    ens = EnsembleDetector.fit(members, specs, P[:200], y[:200])
    lg = ens.combiners["logistic"]
    assert len(lg["coef"]) == 3 and lg["coef"][0] > lg["coef"][2]
    codes = [str(i) for i in range(n)]
    np.testing.assert_allclose(ens.member_proba(codes), P)
    np.testing.assert_allclose(ens.combine(P, "mean"), P.mean(axis=1))
    cf = cross_fit(P[:200], y[:200], [f"d{i % 10}" for i in range(200)], n_folds=5)
    assert len(cf["coef_mean"]) == 3 and len(cf["coef_std"]) == 3
    ens.save(tmp_path / "models" / "ensemble")
    ens2 = EnsembleDetector.load(tmp_path / "models" / "ensemble",
                                 loader=lambda k, d, dev: members[k])
    assert ens2.names == ["xgb", "codebert", "ngram"]
    np.testing.assert_allclose(ens2.predict_proba(codes), ens.predict_proba(codes))
    with pytest.raises(ValueError):
        EnsembleDetector.fit(members, specs, P[:200, :2], y[:200])


def test_eski_iki_uyeli_json_yuklenir(tmp_path):
    """Onceki surumun ensemble.json bicimi (iki uye) aynen yuklenir."""
    d = tmp_path / "models" / "ensemble"
    d.mkdir(parents=True)
    eski = {"method": "logistic", "threshold": 0.29,
            "members": [{"name": "xgb", "kind": "xgb", "dir": "xgb"},
                        {"name": "xgb_donemsiz", "kind": "xgb", "dir": "xgb/ablation/donemsiz"}],
            "combiners": {"mean": {"threshold": 0.30},
                          "logistic": {"coef": [0.84, 0.16], "intercept": -0.14, "C": 1.0,
                                       "threshold": 0.29}},
            "eps": 1e-6, "meta": {"fit_split": "val"}}
    (d / "ensemble.json").write_text(json.dumps(eski))
    gorulen = []

    def loader(kind, path, device):
        gorulen.append(str(path))
        return SahteDedektor(5.0)

    ens = EnsembleDetector.load(d, loader=loader)
    assert ens.names == ["xgb", "xgb_donemsiz"] and ens.threshold == 0.29
    assert gorulen[1] == str(tmp_path / "models" / "xgb" / "ablation" / "donemsiz")
    assert ens.predict_proba(["x" * 30]).shape == (1,)


def test_ikili_uyum_tum_ciftler():
    from tez.ensemble import pairwise_agreement
    P = np.array([[0.1, 0.2, 0.9], [0.9, 0.8, 0.1], [0.7, 0.6, 0.3], [0.2, 0.3, 0.8]])
    a = pairwise_agreement(P, ["xgb", "codebert", "ngram"], [0.5, 0.5, 0.5])
    assert list(a) == ["xgb vs codebert", "xgb vs ngram", "codebert vs ngram"]
    assert a["xgb vs codebert"]["disagreement"] == 0.0
    assert a["xgb vs ngram"]["disagreement"] == 1.0


def test_ngram_uye_turu_taninir(tmp_path):
    from tez.ensemble import MEMBER_KINDS, member_exists
    assert "ngram" in MEMBER_KINDS
    assert not member_exists("ngram", tmp_path)
    (tmp_path / "ngram.joblib").write_bytes(b"")
    assert member_exists("ngram", tmp_path)


def test_09_eksik_uye_atlanir_en_az_iki(tmp_path, capsys):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "ens09", Path(__file__).resolve().parents[1] / "scripts" / "09_ensemble.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = tmp_path / "models"
    (root / "xgb").mkdir(parents=True)
    (root / "xgb" / "meta.json").write_text("{}")
    (root / "xgb" / "model.json").write_text("{}")
    with pytest.raises(SystemExit, match="En az 2"):
        mod.resolve_specs(mod.parse_args([]), root)
    (root / "ngram").mkdir()
    (root / "ngram" / "ngram.joblib").write_bytes(b"")
    specs = mod.resolve_specs(mod.parse_args([]), root)
    assert list(specs) == ["xgb", "ngram"]
    assert "codebert" in capsys.readouterr().out
    specs = mod.resolve_specs(mod.parse_args(["--members", "ngram", "xgb"]), root)
    assert list(specs) == ["ngram", "xgb"]
    eski = mod.resolve_specs(mod.parse_args(["--second-kind", "ngram"]), root)
    assert list(eski) == ["xgb", "ngram"]
