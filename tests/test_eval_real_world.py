import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tez import real_world_eval as rwe
from tez.metrics import compute_metrics

_SPEC = importlib.util.spec_from_file_location(
    "eval_rw08", Path(__file__).resolve().parents[1] / "scripts" / "08_eval_real_world.py")
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)

PROJELER = ["p1", "p2", "p3", "p4", "p5"]
ESIK = 0.55


def _insan(i, proj):
    # Tek id'liler f-string'li, uzunluklar degisken.
    govde = "".join(f"    x{j} = a + {j}\n" for j in range(i % 4))
    son = "    return f'{a}'\n" if i % 2 else "    return str(a)\n"
    return {"id": f"{i:012x}", "func_name": f"f{i}", "project": proj, "label": 0,
            "code": f"def f{i}(a):\n{govde}{son}"}


def _ai(h, model, pt, kopya=False):
    code = h["code"] if kopya else (
        f"def {h['func_name']}(a):\n    result = [a] * 2\n    print(result)\n    return result\n")
    return {"id": h["id"], "func_name": h["func_name"], "model": model, "prompt_type": pt,
            "code": code, "label": 1, "temperature": 0.2, "project": h["project"]}


@pytest.fixture(scope="module")
def veri():
    human = [_insan(i, PROJELER[i % 5]) for i in range(20)]
    ai = []
    for k, h in enumerate(human):
        ai.append(_ai(h, "m1", "skeleton", kopya=(k < 3)))  # ilk 3'u insan kodunun kopyasi
        ai.append(_ai(h, "m2", "natural"))
    return human, ai


class SahteDedektor:
    """Sabit kurallar: 'result' -> 0.9; f-string -> 0.6; diger 0.2."""

    threshold = ESIK

    def predict_proba(self, codes):
        return np.array([0.9 if "result" in c else 0.6 if "f'" in c else 0.2 for c in codes])


def _hesapla(human, ai, n_boot=30, seed=7):
    rows = rwe.build_rows(human, ai)
    det = SahteDedektor()
    p = det.predict_proba([r["code"] for r in rows])
    return rows, p, rwe.evaluate(rows, p, det.threshold, n_boot=n_boot, seed=seed)


@pytest.fixture(scope="module")
def sonuc(veri):
    return _hesapla(*veri)


def test_esik_degismez(sonuc):
    rows, p, res = sonuc
    alt = rwe.all_subset_rows(res)
    assert res["threshold"] == ESIK
    assert all(r["threshold"] == ESIK for r in alt if r["durum"] == "tamam")
    # Bu sette F1'i yukselten baska bir esik (or. 0.7) olsa da kullanilmaz.
    y = [r["label"] for r in rows]
    assert compute_metrics(y, p, 0.7)["f1"] > res["overall"]["f1"]
    assert res["overall"]["f1"] == pytest.approx(compute_metrics(y, p, ESIK)["f1"])


def test_bootstrap_deterministik_ve_proje_bazli(veri):
    _, _, a = _hesapla(*veri, n_boot=15, seed=11)
    _, _, b = _hesapla(*veri, n_boot=15, seed=11)
    _, _, c = _hesapla(*veri, n_boot=15, seed=12)
    assert a["overall"] == b["overall"]
    assert a["memorization"]["haric"] == b["memorization"]["haric"]
    sinir = [k for k in a["overall"] if k.endswith(("_lo", "_hi"))]
    assert [a["overall"][k] for k in sinir] != [c["overall"][k] for k in sinir]
    assert a["overall"]["n_projects"] == 5


def test_cluster_ci_projeyi_birlikte_ceker():
    # Ayni projenin ornekleri hep birlikte cekilir; esik sabit.
    y = np.array([0, 0, 1, 1, 0, 0])
    p = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])
    g = ["a", "a", "b", "b", "c", "c"]
    ci = rwe.cluster_ci(y, p, 0.5, g, n_boot=100, seed=1)
    assert ci["roc_auc"]["lo"] == ci["roc_auc"]["hi"] == 1.0
    # Iki sinifli cekilisler: 'b' en az bir kez + 'a'/'c'; tek sinifli cekilisler atlanir.
    assert ci["roc_auc"]["n_valid"] < 100
    assert ci["tnr"]["point"] == 1.0


def test_alt_kume_metrikleri_dogru(sonuc):
    rows, p, res = sonuc
    lab = np.array([r["label"] for r in rows])
    for r in res["per_model"]:
        m = np.array([x["label"] == 0 or x["model"] == r["model"] for x in rows])
        beklenen = compute_metrics(lab[m], p[m], ESIK)
        for k in ("tp", "fp", "tn", "fn", "f1", "roc_auc"):
            assert r[k] == pytest.approx(beklenen[k]), (r["model"], k)
        assert r["n_human"] == 20 and r["n_ai"] == 20
    m1 = next(r for r in res["per_model"] if r["model"] == "m1")
    # m1: 17 'result' (0.9) + 3 kopya (id 0,1,2); kopyalardan yalniz f-string'li id 1 yakalanir.
    assert (m1["tp"], m1["fn"]) == (18, 2)
    # Insan: 10 f-string'li (0.6 >= 0.55) -> FP.
    assert (m1["tn"], m1["fp"]) == (10, 10)
    assert m1["tnr"] == pytest.approx(0.5)
    assert {r["prompt_type"] for r in res["per_prompt"]} == {"skeleton", "natural"}
    # 2 uretici x 2 istem = 4 hucre; m1-natural ve m2-skeleton AI'siz (yalniz insan).
    assert len(res["model_x_prompt"]) == 4
    hucre = {(r["model"], r["prompt_type"]): r for r in res["model_x_prompt"]}
    assert hucre[("m1", "natural")]["n_ai"] == 0
    assert hucre[("m2", "natural")]["tp"] == 20


def test_proje_tnr(sonuc):
    rows, p, res = sonuc
    assert [r["project"] for r in res["per_project"]] == PROJELER
    for r in res["per_project"]:
        hum = [i for i, x in enumerate(rows) if x["project"] == r["project"] and x["label"] == 0]
        assert r["tn"] == sum(p[i] < ESIK for i in hum)
        assert r["tnr_lo"] <= r["tnr"] <= r["tnr_hi"]


def test_ezber_ve_fstring(sonuc):
    rows, p, res = sonuc
    m = res["memorization"]
    assert m["n_high"] == 3 and m["n_ai_unmatched"] == 0  # 3 birebir kopya
    assert m["haric"]["n_ai"] == 37 and m["haric"]["n_human"] == 20
    assert m["dahil"]["n"] == len(rows)
    assert m["yalniz_yuksek"]["n_ai"] == 3
    f = res["fstring"]
    assert f["with"]["n"] + f["without"]["n"] == len(rows)
    assert f["frac_fstring_human"] == pytest.approx(0.5)
    mask = np.array([r["fstring"] == 1 for r in rows])
    lab = np.array([r["label"] for r in rows])
    assert f["with"]["fp"] == compute_metrics(lab[mask], p[mask], ESIK)["fp"] == 10
    assert f["without"]["fp"] == 0


def test_uzunluk_eslesik_dengeli(sonuc):
    lm = sonuc[2]["length_matched"]
    assert lm["n_ai"] == lm["n_human"] > 0


def test_tekrarlanan_ai_hata(veri):
    human, ai = veri
    with pytest.raises(ValueError):
        rwe.build_rows(human, ai + ai[:1])


def test_ai_yoksa_anlasilir_cikis(tmp_path, veri):
    human, _ = veri
    (tmp_path / "human.jsonl").write_text(
        "\n".join(json.dumps(h) for h in human) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="ai.jsonl"):
        ev.main(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")])
    (tmp_path / "ai.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="bos"):
        ev.main(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")])


def test_markdown_uretilir(sonuc):
    res = {**sonuc[2], "model": "sahte", "model_dir": "-"}
    md = ev.summary_markdown(res)
    assert "Ezber kontrolu" in md and "f-string" in md and "UYARI" in md
