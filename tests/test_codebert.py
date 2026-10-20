import math
from pathlib import Path

import numpy as np
import pytest

from tez.codebert import (
    apply_threshold,
    build_features,
    explain_code,
    is_better,
    pad_batch,
    render_masked,
    segment_code,
    truncate_ids,
)
from tez.metrics import best_threshold, compute_metrics

KOD = (
    "def topla(a, b=2):\n"
    "    if a  >= b:\n"
    "        return f'{a}' + \"\"\"x\n"
    "y\"\"\"\n"
    "    sonuc = [a ** 2, b // 3, 1.5e-3]\n"
    "\n"
    "    return sonuc  # yorum\n"
)


def _hf_onbellekte(model="microsoft/codebert-base"):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model, "config.json"), str)


# --- kesme ---

def test_kisa_dizi_kesilmez():
    assert truncate_ids([1, 2, 3], 5) == [1, 2, 3]


def test_bas_son_kesme():
    ids = list(range(100))
    out = truncate_ids(ids, 8, "head_tail", head_frac=0.25)
    assert out == [0, 1, 94, 95, 96, 97, 98, 99]


def test_bas_son_kesme_uzunluk_ve_uclar():
    ids = list(range(1000))
    out = truncate_ids(ids, 254, "head_tail", head_frac=0.25)
    assert len(out) == 254
    assert out[:64] == list(range(64)) and out[-1] == 999


def test_yalniz_bas_kesme():
    assert truncate_ids(list(range(10)), 4, "head") == [0, 1, 2, 3]


def test_head_frac_bir_ise_yalniz_bas():
    assert truncate_ids(list(range(10)), 4, "head_tail", head_frac=1.0) == [0, 1, 2, 3]


def test_gecersiz_strateji():
    with pytest.raises(ValueError):
        truncate_ids([1, 2], 1, "orta")


def test_dolgu_maskesi():
    b = pad_batch([[5, 6, 7], [8]], pad_id=1)
    assert b["input_ids"].tolist() == [[5, 6, 7], [8, 1, 1]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


# --- LIME bolumleyici ---

@pytest.mark.parametrize("kod", [KOD, "", "x=1", "  a\tb \n\n\t\tc", "s = r'\\d+' + b\"\\\"\"\n", "def f(x) -> int: ...\n"])
def test_bolumleme_metni_bozmaz(kod):
    assert "".join(t for _, t in segment_code(kod)) == kod


def test_bolumleme_birim_turleri():
    units = segment_code(KOD)
    kinds = {t: k for k, t in units}
    assert kinds["def"] == "name" and kinds["topla"] == "name"
    assert kinds["    "] == "indent" and kinds["        "] == "indent"
    assert kinds["  "] == "space"
    assert kinds[">="] == "op" and kinds["**"] == "op" and kinds["//"] == "op"
    assert kinds["f'{a}'"] == "str" and kinds['"""x\ny"""'] == "str"
    assert kinds["1.5e"] == "num"


def test_ozellikler_gruplanir_ve_satir_sonu_maskelenmez():
    units = segment_code(KOD)
    names, members = build_features(units, group=True)
    assert "<girinti:4>" in names and "<bosluk:2>" in names
    assert "\n" not in names and " " not in names
    j = names.index("a")
    assert len(members[j]) >= 3
    names_pos, _ = build_features(units, group=False)
    assert len(names_pos) > len(names)


def test_maskeleme_hepsi_acikken_orijinal():
    units = segment_code(KOD)
    names, members = build_features(units)
    assert render_masked(units, members, np.ones(len(names))) == KOD


def test_maskeleme_satir_yapisini_korur():
    units = segment_code(KOD)
    names, members = build_features(units)
    out = render_masked(units, members, np.zeros(len(names)), "<unk>")
    # Cok satirli karakter dizisi tek birimdir; icindeki satir sonu da maskelenir.
    assert out.count("\n") == KOD.count("\n") - 1
    assert "topla" not in out and "<unk>" in out
    assert "    " not in out


def test_lime_sahte_modelle_etkili_birimi_bulur():
    # Sahte model: kodda "**" varsa AI olasiligi yuksek.
    def predict(codes):
        return np.array([0.9 if "**" in c else 0.1 for c in codes])

    exp = explain_code(KOD, predict, num_features=5, num_samples=300, seed=0)
    assert exp["features"][0][0] == "**" and exp["features"][0][1] > 0.5
    assert len(exp["unit_weights"]) == len(exp["units"])
    assert "".join(exp["units"]) == KOD


def test_lime_deterministik():
    def predict(codes):
        return np.array([len(c) / 1000 for c in codes])

    a = explain_code(KOD, predict, num_samples=100, seed=3)
    b = explain_code(KOD, predict, num_samples=100, seed=3)
    assert a["features"] == b["features"]


# --- esik ve checkpoint secimi ---

def test_esik_kurali_compute_metrics_ile_ayni():
    y, p = [0, 0, 1, 1, 1], [0.2, 0.55, 0.55, 0.7, 0.4]
    t = best_threshold(y, p)
    pred = apply_threshold(p, t)
    m = compute_metrics(y, p, t)
    assert m["tp"] == int(((pred == 1) & (np.array(y) == 1)).sum())
    assert m["fp"] == int(((pred == 1) & (np.array(y) == 0)).sum())


def test_esik_esitte_ai_sayilir():
    assert apply_threshold([0.5, 0.4999], 0.5).tolist() == [1, 0]


def test_checkpoint_secimi_auc_sonra_f1():
    a = {"roc_auc": 0.90, "f1": 0.80}
    assert is_better(a, None)
    assert is_better({"roc_auc": 0.91, "f1": 0.10}, a)
    assert is_better({"roc_auc": 0.90, "f1": 0.81}, a)
    assert not is_better({"roc_auc": 0.90, "f1": 0.80}, a)
    assert not is_better({"roc_auc": math.nan, "f1": 1.0}, a)
    assert is_better({"roc_auc": 0.5, "f1": 0.95}, a, primary="f1")


# --- model gerektiren (yavas) testler ---

@pytest.mark.slow
@pytest.mark.skipif(not _hf_onbellekte(), reason="CodeBERT HF onbellekte yok")
def test_predict_proba_sira_ve_aralik():
    from tez.codebert import CodeBertDetector

    det = CodeBertDetector.from_dir("microsoft/codebert-base", device="cpu")
    codes = [KOD, "def f():\n    return 1\n", KOD * 40]
    p = det.predict_proba(codes, batch_size=2)
    assert p.shape == (3,) and ((p >= 0) & (p <= 1)).all()
    p_tek = np.array([det.predict_proba([c])[0] for c in codes])
    assert np.allclose(p, p_tek, atol=1e-4)
