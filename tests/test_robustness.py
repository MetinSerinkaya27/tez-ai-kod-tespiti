import ast
import random

import numpy as np
import pytest

from tez.features import FEATURE_NAMES, features_frame
from tez.normalize import normalize_code
from tez.robustness import (
    PerturbationError,
    XGBCodeDetector,
    apply_perturbation,
    ast_renamed_dump,
    black_format,
    bootstrap_ci,
    bootstrap_delta_ci,
    calibration_summary,
    confident_errors,
    cross_source_eval,
    error_markdown,
    group_cv_xgb,
    leakage_report,
    local_rename_map,
    make_xgb_fit_fn,
    normalize_layout,
    rename_locals,
    restrict_to_sources,
    robustness_report,
)

KUCUK_IZGARA = {"max_depth": [2], "learning_rate": [0.3], "min_child_weight": [1]}

ORNEKLER = [
    "def f(a, b=2):\n    x = a+b\n    return x\n",
    # self/cls/attribute/builtin/kwarg/global/sinif govdesi/except/lambda/comprehension
    (
        "def g(self, data, *args, **kw):\n"
        "    global SAYAC\n"
        "    SAYAC = 1\n"
        "    total = len(data)\n"
        "    class Ic:\n"
        "        alan = 3\n"
        "        def m(self, q):\n"
        "            return self.alan + q\n"
        "    try:\n"
        "        total += sum(x*2 for x in data)\n"
        "    except (ValueError, TypeError) as err:\n"
        "        print(f'hata {err!r} {total}', file=None)\n"
        "    h = lambda z: z + total\n"
        "    self.data = dict(total=total)\n"
        "    return [h(v) for v in args], kw, Ic\n"
    ),
    # cok satirli dize, devam satirlari, ic ice fonksiyon + nonlocal, unicode
    (
        "def h(yol, deger):\n"
        "    metin = '''satir1\n"
        "      girintili   \n"
        "satir3'''\n"
        "    def ic():\n"
        "        nonlocal deger\n"
        "        deger += 1\n"
        "        return deger\n"
        "    sonuc = ic() + \\\n"
        "        len(metin)\n"
        "    s = 'ğüşı'; t = (s,\n"
        "                     yol)\n"
        "    return sonuc, t\n"
    ),
    # 2 bosluk girinti ve bos satirlar
    (
        "def k(n):\n"
        "  if n > 0:\n"
        "\n"
        "      return [i for i in range(n)]\n"
        "  else:\n"
        "      return None\n"
    ),
    # mevcut v1 adi ile carpisma
    "def c(v1, a):\n    b = v1 + a\n    return b\n",
]


def _isimsiz_dump(code: str) -> str:
    """Adlar haric AST yapisi."""
    tree = ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            n.id = "_"
        elif isinstance(n, ast.arg):
            n.arg = "_"
        elif isinstance(n, ast.ExceptHandler) and n.name:
            n.name = "_"
    return ast.dump(tree, include_attributes=False)


@pytest.mark.parametrize("code", ORNEKLER)
def test_black_anlami_korur(code):
    out = black_format(code)
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(code))


@pytest.mark.parametrize("code", ORNEKLER)
def test_girinti_anlami_korur(code):
    out = normalize_layout(code)
    assert ast.dump(ast.parse(out)) == ast.dump(ast.parse(code))
    assert "\n\n" not in out.rstrip("\n").replace("\n      girintili", "")
    # Blok girintileri 4'un kati.
    for t in ast.walk(ast.parse(out)):
        if isinstance(t, ast.stmt):
            assert t.col_offset % 4 == 0


def test_girinti_cok_satirli_dizeye_dokunmaz():
    out = normalize_layout(ORNEKLER[2])
    assert "      girintili   \n" in out


@pytest.mark.parametrize("code", ORNEKLER)
def test_yeniden_adlandirma_anlami_korur(code):
    out = rename_locals(code)
    m = local_rename_map(code)
    # Esleme uygulanmis orijinal AST ile birebir ayni; adlar haric yapi ayni.
    assert ast.dump(ast.parse(out), include_attributes=False) == ast_renamed_dump(code, m)
    assert _isimsiz_dump(out) == _isimsiz_dump(code)
    # Esleme birebir (iki ad ayni yeni ada gitmez) ve var olan adlarla carpismaz.
    assert len(set(m.values())) == len(m)
    eski = {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)}
    assert not set(m.values()) & eski


def test_yeniden_adlandirma_dokunulmayanlar():
    code = ORNEKLER[1]
    m = local_rename_map(code)
    for ad in ("self", "len", "sum", "print", "dict", "SAYAC", "alan", "Ic", "g", "m"):
        assert ad not in m
    assert {"data", "args", "kw", "total", "err", "h", "z", "x", "v", "q"} <= set(m)
    out = rename_locals(code)
    assert "self.data" in out and "self.alan" in out and "dict(total=" in out
    assert "file=None" in out


def test_yeniden_adlandirma_calisma_davranisi():
    code = ORNEKLER[2]
    ns1, ns2 = {}, {}
    exec(code, ns1)
    exec(rename_locals(code), ns2)
    assert ns1["h"]("a", 1) == ns2["h"]("a", 1)


def test_uygulama_normalize_ve_geri_dusme():
    res = apply_perturbation(ORNEKLER + ["def bozuk(:\n"], black_format)
    assert res.n == 6 and res.failed == 1
    assert res.codes[-1] == "def bozuk(:\n"
    assert all(normalize_code(c) == c for c in res.codes[:-1])


def test_bootstrap_mukemmel_ve_kume_genisligi():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 50)
    p = np.where(y == 1, 0.9, 0.1)
    ci = bootstrap_ci(y, p, 0.5, groups=np.repeat(np.arange(20), 5), n_boot=200)
    assert ci["roc_auc"]["lo"] == ci["roc_auc"]["hi"] == 1.0

    # Depo etkisi: ayni depodaki tum ornekler ayni yonde hatali -> kume GA'si daha genis.
    groups = np.repeat(np.arange(20), 10)
    y = np.tile([0, 1], 100)
    depo_kaymasi = rng.normal(0, 1.5, 20)[groups]
    p = 1 / (1 + np.exp(-(2 * (y - 0.5) + depo_kaymasi * np.where(y == 1, -1, 1))))
    kume = bootstrap_ci(y, p, 0.5, groups=groups, n_boot=500)
    iid = bootstrap_ci(y, p, 0.5, groups=None, n_boot=500)
    assert kume["roc_auc"]["point"] == iid["roc_auc"]["point"]
    w_k = kume["roc_auc"]["hi"] - kume["roc_auc"]["lo"]
    w_i = iid["roc_auc"]["hi"] - iid["roc_auc"]["lo"]
    assert w_k > 1.3 * w_i


def test_bootstrap_deterministik_ve_delta():
    y = np.array([0, 1] * 20)
    p = np.linspace(0.2, 0.8, 40)
    g = np.repeat(np.arange(10), 4)
    assert bootstrap_ci(y, p, 0.5, g, 100) == bootstrap_ci(y, p, 0.5, g, 100)
    d = bootstrap_delta_ci(y, p, p, 0.5, g, 100)
    assert d["roc_auc"]["point"] == d["roc_auc"]["lo"] == d["roc_auc"]["hi"] == 0.0


def test_kalibrasyon():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    iyi = calibration_summary(y, p)
    assert iyi["ece"] < 0.02
    kotu = calibration_summary(y, np.clip(p * 0.3 + 0.7, 0, 1))
    assert kotu["ece"] > 0.2 and kotu["brier"] > iyi["brier"]


def test_emin_hatalar_ve_markdown():
    recs = [{"id": str(i), "code": f"def f{i}():\n    return {i}\n", "label": i % 2,
             "repo": "r", "source": "human" if i % 2 == 0 else "m"} for i in range(8)]
    p = np.array([0.9, 0.1, 0.6, 0.4, 0.2, 0.8, 0.95, 0.05])
    e = confident_errors(recs, p, 0.5, k=2)
    assert e["FP"] == [6, 0] and e["FN"] == [7, 1]
    md = error_markdown(recs, p, 0.5, k=2)
    assert "## FP (2 ornek)" in md and "`6` - p(AI) = 0.9500" in md


def test_sizinti_raporu():
    a = [{"repo": "x", "pair_id": "1", "code": "a"}]
    b = [{"repo": "y", "pair_id": "2", "code": "a "}]
    rep = leakage_report({"train": a, "test": b})
    assert rep["repo:train&test"] == 0 and rep["code:train&test"] == 1


# ---------------------------------------------------------------------------
# Sentetik kisa yol: paketin kisa yolu YAKALADIGININ kaniti
# ---------------------------------------------------------------------------

def _govde(rng: random.Random, i: int, yapisal_ai: bool | None) -> str:
    """Rastgele ama siniftan bagimsiz govde; yapisal_ai verilirse yapi farki ekler."""
    satir = []
    for k in range(rng.randint(1, 4)):
        satir.append(f"    val_{k} = data.get(QQ{k}QQ, {rng.randint(0, 9)})")
    if yapisal_ai is True:
        satir.append("    out = [v for v in data if v]")
    elif yapisal_ai is False:
        satir += ["    out = []", "    for v in data:", "        if v:",
                  "            out.append(v)"]
    satir.append("    return QQretQQ")
    return f"def fn_{i}(data):\n" + "\n".join(satir) + "\n"


def _sentetik(n: int, seed: int, kisa_yol: str) -> list[dict]:
    """kisa_yol='tirnak': AI tek tirnak, insan cift tirnak; kod govdeleri ayni dagilim.
    kisa_yol='yapi': tirnak sinifla ilgisiz, AI comprehension / insan for dongusu.
    """
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        label = i % 2
        repo = f"r{seed}_{i // 4}"
        if kisa_yol == "tirnak":
            code = _govde(rng, i, None).replace("QQ", "'" if label else '"')
        else:
            q = rng.choice(["'", '"'])
            code = _govde(rng, i, bool(label)).replace("QQ", q)
        recs.append({"id": f"{seed}_{i}", "pair_id": f"{seed}_{i // 2}", "code": normalize_code(code),
                     "label": label, "repo": repo, "source": "ai-a" if label else "human"})
    return recs


def _egit(kisa_yol: str):
    fit = make_xgb_fit_fn(list(FEATURE_NAMES), grid=KUCUK_IZGARA, n_jobs=1)
    return fit(_sentetik(200, 1, kisa_yol), _sentetik(80, 2, kisa_yol))


def test_sentetik_tirnak_kisa_yolu_black_ile_coker():
    det = _egit("tirnak")
    te = _sentetik(200, 3, "tirnak")
    rows, _ = robustness_report(
        det, [r["code"] for r in te], [r["label"] for r in te], [r["repo"] for r in te],
        {"black": black_format, "girinti_bosluk": normalize_layout}, n_boot=200)
    r = {row["perturbation"]: row for row in rows}
    print({k: (round(v["roc_auc"], 3), round(v["f1"], 3)) for k, v in r.items()})
    assert r["orijinal"]["roc_auc"] >= 0.99
    # black tum tirnaklari cift yapar -> siniflar ayirt edilemez.
    assert r["black"]["roc_auc"] <= 0.65
    assert r["black"]["d_roc_auc_hi"] < -0.3  # dusus GA'si sifiri icermez
    # Girinti/bos satir normallestirme tirnaga dokunmaz -> kisa yol yasar.
    assert r["girinti_bosluk"]["roc_auc"] >= 0.99


def test_sentetik_yapisal_sinyal_black_ile_yasar():
    det = _egit("yapi")
    te = _sentetik(200, 3, "yapi")
    rows, _ = robustness_report(
        det, [r["code"] for r in te], [r["label"] for r in te], [r["repo"] for r in te],
        {"black": black_format, "yeniden_adlandirma": rename_locals}, n_boot=200)
    r = {row["perturbation"]: row for row in rows}
    print({k: (round(v["roc_auc"], 3), round(v["f1"], 3)) for k, v in r.items()})
    assert r["orijinal"]["roc_auc"] >= 0.99
    assert r["black"]["roc_auc"] >= 0.95
    assert r["yeniden_adlandirma"]["roc_auc"] >= 0.95


def test_capraz_kaynak_ve_kisitlama():
    a = _sentetik(120, 5, "yapi")
    # Ikinci uretici: farkli kisa yol (tirnak); insanlari a ile ayni cift yapisinda.
    b = [{**r, "id": r["id"] + "b", "pair_id": r["pair_id"] + "b", "source": "ai-b"}
         if r["label"] else {**r, "id": r["id"] + "b", "pair_id": r["pair_id"] + "b"}
         for r in _sentetik(120, 6, "tirnak")]
    recs = a + b
    assert all(r["source"] in ("human", "ai-a") for r in restrict_to_sources(recs, {"ai-a"}))
    tr = [r for r in recs if int(r["pair_id"].split("_")[1].rstrip("b")) < 40]
    va = [r for r in recs if 40 <= int(r["pair_id"].split("_")[1].rstrip("b")) < 50]
    te = [r for r in recs if int(r["pair_id"].split("_")[1].rstrip("b")) >= 50]
    fit = make_xgb_fit_fn(list(FEATURE_NAMES), grid=KUCUK_IZGARA, n_jobs=1)
    rows = cross_source_eval(tr, va, te, fit)
    m = {(r["train_source"], r["test_source"]): r for r in rows}
    assert set(m) == {(s, t) for s in ("ai-a", "ai-b") for t in ("ai-a", "ai-b")}
    assert m[("ai-a", "ai-a")]["gorulen"] and m[("ai-a", "ai-a")]["roc_auc"] >= 0.95
    # Gorulmeyen uretici: farkli sinyal -> belirgin dusus.
    assert m[("ai-a", "ai-b")]["roc_auc"] < m[("ai-a", "ai-a")]["roc_auc"] - 0.2


def test_grup_cv_sizintisiz():
    recs = _sentetik(200, 7, "yapi")
    X = features_frame(r["code"] for r in recs)
    rows = group_cv_xgb(X, [r["label"] for r in recs], [r["repo"] for r in recs],
                        list(FEATURE_NAMES), n_folds=5, grid=KUCUK_IZGARA, n_jobs=1)
    assert len(rows) == 5 and all(r["durum"] == "tamam" for r in rows)
    assert sum(r["n_eval"] for r in rows) == 200
    assert all(r["roc_auc"] >= 0.9 for r in rows)


def test_xgb_sarmalayici_shap():
    det = _egit("yapi")
    assert isinstance(det, XGBCodeDetector)
    top = det.top_features([_sentetik(2, 9, "yapi")[1]["code"]], 5)
    assert len(top) == 1 and len(top[0]) == 5
    assert [abs(s) for *_, s in top[0]] == sorted((abs(s) for *_, s in top[0]), reverse=True)


def test_black_u_onekini_kabul_eder():
    out = black_format("def f():\n    return u'a'\n")
    assert out == 'def f():\n    return "a"\n'


def test_hatali_kodda_perturbasyon_hatasi():
    with pytest.raises(PerturbationError):
        normalize_layout("def f(:\n")
