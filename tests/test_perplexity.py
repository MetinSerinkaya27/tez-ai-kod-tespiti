import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tez import perplexity as ppl
from tez.perplexity import PPL_FEATURES, cached_ppl, sequence_stats, token_stats

V = 7


class SahteTokenizer:
    """Karakter -> id (1..V-1); 0 dolgu."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + ord(c) % (V - 1) for c in text]}


class SahteLM(torch.nn.Module):
    """Bigram LM: t konumundaki logitler yalnizca input_ids[t]'ye bagli."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Embedding(V, V)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(V, V, generator=g) * 2)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.emb(input_ids))


def _elle(ids: list[int], W: np.ndarray) -> dict[str, float]:
    """Numpy ile elle: her t>=1 icin softmax(W[ids[t-1]])."""
    lps, ents, ranks = [], [], []
    for t in range(1, len(ids)):
        z = W[ids[t - 1]].astype(np.float64)
        logp = z - np.log(np.exp(z - z.max()).sum()) - z.max()
        lps.append(logp[ids[t]])
        ents.append(-(np.exp(logp) * logp).sum())
        ranks.append(int((z > z[ids[t]]).sum()) + 1)
    lps, ents, ranks = map(np.array, (lps, ents, ranks))
    return {"ppl_mean_logprob": lps.mean(), "ppl_std_logprob": lps.std(),
            "ppl_median_logprob": np.median(lps), "ppl_p10_logprob": np.percentile(lps, 10),
            "ppl_p90_logprob": np.percentile(lps, 90), "ppl_mean_entropy": ents.mean(),
            "ppl_mean_rank_log": np.log1p(ranks).mean(), "ppl_top1_frac": (ranks == 1).mean(),
            "ppl_top10_frac": (ranks <= 10).mean(),
            "ppl_logprob_entropy_ratio": -lps.mean() / ents.mean()}


@pytest.fixture()
def sahte():
    return SahteTokenizer(), SahteLM().eval()


def test_ozellikler_elle_hesapla_ayni(sahte):
    tok, lm = sahte
    kodlar = ["def f(a):\n    return a\n", "x = 1\n", "ab"]
    df = token_stats(kodlar, tokenizer=tok, model=lm, batch_size=3)
    assert list(df.columns) == PPL_FEATURES and len(df) == 3
    assert not any("num_tokens" in c for c in df.columns)  # uzunluk sizmasin
    W = lm.emb.weight.detach().numpy()
    for i, kod in enumerate(kodlar):
        beklenen = _elle(tok(kod)["input_ids"], W)
        for k, v in beklenen.items():
            assert df.iloc[i][k] == pytest.approx(v, rel=1e-5, abs=1e-6), k


def test_dolgu_ve_ilk_token_dislanir(sahte):
    tok, lm = sahte
    kodlar = ["a", "kisa", "cok daha uzun bir satir = 1\n"]
    tek = pd.concat([token_stats([k], tokenizer=tok, model=lm) for k in kodlar],
                    ignore_index=True)
    yigin = token_stats(kodlar, tokenizer=tok, model=lm, batch_size=3)
    # Tek tokenli kodda tahmin edilecek konum yok -> NaN (ilk token sayilmaz).
    assert yigin.iloc[0].isna().all()
    pd.testing.assert_frame_equal(tek, yigin, rtol=1e-6)

    # Dogrudan: dolgu konumlarinin logitleri ne olursa olsun sonuc degismez.
    ids = torch.tensor([[1, 2, 3, 0, 0], [0, 0, 1, 2, 3]])
    att = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])
    lg = lm(ids).logits.clone()
    lg[0, 2:] = 100 * torch.randn(3, V)  # son gercek token ve dolgu: kullanilmaz
    lg[1, :2] = 100 * torch.randn(2, V)  # sol dolgu
    a, b = sequence_stats(lg, ids, att)
    W = lm.emb.weight.detach().numpy()
    beklenen = _elle([1, 2, 3], W)
    for k, v in beklenen.items():
        assert a[k] == pytest.approx(v, rel=1e-5) and b[k] == pytest.approx(v, rel=1e-5)


def test_kesme_sayilir(sahte):
    tok, lm = sahte
    df = token_stats(["x" * 50, "y" * 5], tokenizer=tok, model=lm, max_length=10)
    assert df.attrs["n_truncated"] == 1
    kesik = token_stats(["x" * 10], tokenizer=tok, model=lm)
    pd.testing.assert_series_equal(df.iloc[0], kesik.iloc[0], check_names=False)


def _kayitlar(n=4):
    return [{"id": f"r{i}", "code": f"def f{i}(a):\n    return a + {i}\n", "label": i % 2}
            for i in range(n)]


def test_onbellek_gecersizlestirme(sahte, tmp_path, monkeypatch):
    tok, lm = sahte
    sayac = {"n": 0}
    gercek = ppl.token_stats

    def sayan(codes, **kw):
        sayac["n"] += 1
        return gercek(codes, tokenizer=tok, model=lm, **{k: v for k, v in kw.items()
                                                        if k == "max_length"})

    monkeypatch.setattr(ppl, "token_stats", sayan)
    recs = _kayitlar()
    with pytest.raises(FileNotFoundError):
        cached_ppl(recs, tmp_path, "train", compute=False)
    a = cached_ppl(recs, tmp_path, "train")
    assert list(a.index) == [r["id"] for r in recs] and sayac["n"] == 1
    b = cached_ppl(recs, tmp_path, "train", compute=False)
    assert sayac["n"] == 1
    pd.testing.assert_frame_equal(a, b)
    ilk = next(tmp_path.glob("ppl_train_*.parquet"))

    recs[0] = {**recs[0], "code": "x = 2\n"}  # kod degisti -> yeniden hesap
    cached_ppl(recs, tmp_path, "train")
    dosyalar = list(tmp_path.glob("ppl_train_*.parquet"))
    assert sayac["n"] == 2 and len(dosyalar) == 1 and dosyalar[0] != ilk
    # Model adi ya da kesme uzunlugu degisirse hash degisir.
    assert ppl.ppl_hash(recs, "baska/model") != ppl.ppl_hash(recs)
    assert ppl.ppl_hash(recs, max_length=256) != ppl.ppl_hash(recs)
    # 04'un AST onbellegi (train_*.parquet) ile cakismaz.
    assert not list(tmp_path.glob("train_*.parquet"))


def test_explain_one_ppl_modelinde_calisir(sahte, monkeypatch):
    from tez.features import FEATURE_NAMES, features_frame
    from tez.xgb_model import XGBDetector, select_and_fit

    tok, lm = sahte
    monkeypatch.setattr(ppl, "load_lm", lambda *a, **k: (tok, lm))
    kodlar = [f"def f{i}(a):\n    return a + {i}\n" if i % 2 else
              f"def g{i}(b):\n    x = b * {i}\n    return x\n" for i in range(30)]
    y = np.array([i % 2 for i in range(30)])
    X = pd.concat([features_frame(kodlar), token_stats(kodlar, tokenizer=tok, model=lm)], axis=1)
    cols = list(FEATURE_NAMES) + PPL_FEATURES
    m, _, _ = select_and_fit(X[cols], y, X[cols], y, n_jobs=1,
                             grid={"max_depth": [2], "learning_rate": [0.3],
                                   "min_child_weight": [1]})
    det = XGBDetector(m, cols, 0.5, {"ppl_model": "sahte"})
    F = det.code_frame(kodlar[1])
    assert list(F.columns) == cols
    np.testing.assert_allclose(F[PPL_FEATURES].to_numpy(dtype=float),
                               X[PPL_FEATURES].iloc[[1]].to_numpy(dtype=float), rtol=1e-6)
    # Toplu arayuz (06/08 degerlendiricileri icin) ayni degerleri verir.
    pd.testing.assert_frame_equal(det.frame(kodlar[:5]).reset_index(drop=True),
                                  X[cols].iloc[:5].reset_index(drop=True), rtol=1e-6)
    out = det.explain_one(kodlar[1])
    assert len(out) == len(cols) and any(f.startswith("ppl_") for f, _, _ in out)


# --- 04_train_xgb.py: --with-ppl yokken davranis aynen korunur ---

def _script04():
    spec = importlib.util.spec_from_file_location(
        "train_xgb_04", Path(__file__).resolve().parents[1] / "scripts" / "04_train_xgb.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _veri_dizini(d: Path, sahte) -> None:
    import random

    rng = random.Random(0)
    d.mkdir()
    for s, n in (("train", 40), ("val", 20), ("test", 20)):
        recs = []
        for i in range(n):
            lab = i % 2
            k = rng.randint(1, 3)
            body = ("".join(f"    x{j}=a+{j}*'s'\n" for j in range(k)) if lab == 0 else
                    "".join(f"    value_{j} = a + {j} * \"s\"\n" for j in range(k)))
            recs.append({"id": f"{s}{i}", "code": f"def f{i}(a):\n{body}    return a\n",
                         "label": lab, "source": "m" if lab else "human"})
        with open(d / f"{s}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)


def _calistir(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["04"] + argv)
    mod.main()


def test_04_with_ppl_yokken_ayni_davranis(sahte, tmp_path, monkeypatch):
    from tez.dataset import load_jsonl

    tok, lm = sahte
    mod = _script04()
    # Hiz icin tek noktali izgara (izgara secimi test_xgb_model'de sinaniyor).
    monkeypatch.setattr("tez.xgb_model.GRID", {"max_depth": [2], "learning_rate": [0.3],
                                               "min_child_weight": [1]})
    # SHAP grafikleri yavas ve karsilastirmayla ilgisiz.
    monkeypatch.setattr(mod, "shap_report", lambda det, X, out: pd.DataFrame(
        {"feature": det.features[:1], "mean_abs_shap": [0.0]}))
    data = tmp_path / "veri"
    _veri_dizini(data, sahte)
    eski = mod.ablation_sets(["fmt_num_lines"])
    assert list(eski) == ["tum", "uzunluksuz", "donemsiz", "uzunluksuz_donemsiz",
                          "yalniz_bicim", "yalniz_ast"]
    assert mod.ablation_sets(["fmt_num_lines"], None) == eski

    _calistir(mod, ["--data-dir", str(data), "--out-dir", str(tmp_path / "a"),
                    "--n-jobs", "1"], monkeypatch)
    assert (tmp_path / "a" / "models" / "xgb" / "model.json").exists()
    assert not (tmp_path / "a" / "models" / "xgb_ppl").exists()
    a = pd.read_csv(tmp_path / "a" / "outputs" / "xgb" / "ablation_metrics.csv")
    assert not a["model"].str.contains("ppl").any()

    # PPL onbellegi yoksa net hata.
    with pytest.raises(SystemExit, match="03c_perplexity"):
        _calistir(mod, ["--data-dir", str(data), "--out-dir", str(tmp_path / "b"),
                        "--n-jobs", "1", "--with-ppl"], monkeypatch)
    for s in ("train", "val", "test"):
        P = token_stats([r["code"] for r in load_jsonl(data / f"{s}.jsonl")],
                        tokenizer=tok, model=lm)
        recs = load_jsonl(data / f"{s}.jsonl")
        P.index = pd.Index([r["id"] for r in recs], name="id")
        P.to_parquet(ppl.ppl_cache_path(recs, data / "features", s))
    _calistir(mod, ["--data-dir", str(data), "--out-dir", str(tmp_path / "b"),
                    "--n-jobs", "1", "--with-ppl"], monkeypatch)
    b = pd.read_csv(tmp_path / "b" / "outputs" / "xgb_ppl" / "ablation_metrics.csv")
    assert {"tum+ppl", "yalniz_ppl", "donemsiz+ppl"} <= set(b["model"])
    # Ortak satirlar (eski ablasyonlar + taban) birebir ayni.
    ortak = b[b["model"].isin(a["model"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, ortak)
    meta = json.loads((tmp_path / "b" / "models" / "xgb_ppl" / "meta.json").read_text())
    assert meta["ablation"] == "tum+ppl" and set(PPL_FEATURES) <= set(meta["features"])


@pytest.mark.slow
def test_gercek_lm_deterministik():
    kodlar = ["def f(a):\n    return a + 1\n", "def g(x, y):\n    return [i * y for i in x]\n"]
    a = token_stats(kodlar, device="cpu", batch_size=2)
    b = token_stats(kodlar[::-1], device="cpu", batch_size=1)[::-1].reset_index(drop=True)
    assert np.isfinite(a.to_numpy()).all()
    assert (a["ppl_mean_logprob"] < 0).all() and (a["ppl_top1_frac"] <= 1).all()
    pd.testing.assert_frame_equal(a, b, atol=1e-4)
