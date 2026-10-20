import pytest

from tez.dataset import (
    assign_repos,
    check_splits,
    load_jsonl,
    model_short,
    pair_records,
    split_by_repo,
    split_stats,
    write_jsonl,
)

MODEL = "deepseek-coder:6.7b-instruct"


def _human(i, repo, code=None):
    return {
        "id": f"h{i}", "func_name": f"f{i}", "signature": f"def f{i}():",
        "docstring": "Aciklama.", "code": code or f"def f{i}():\n    return {i}\n",
        "repo": repo, "url": f"u{i}", "label": 0,
    }


def _ai(i, code=None, model=MODEL):
    return {
        "id": f"h{i}", "func_name": f"f{i}", "model": model,
        "code": code or f"def f{i}():\n    x = {i}\n    return x\n",
        "raw": "", "repo": "", "label": 1,
    }


def _synthetic(n_repos=40, seed=0):
    """Boyutlari dengesiz depolardan sentetik insan/AI kayitlari."""
    import random
    rng = random.Random(seed)
    human, ai, i = [], [], 0
    for r in range(n_repos):
        for _ in range(rng.choice([1, 2, 3, 5, 10, 25])):
            human.append(_human(i, f"org/repo{r}"))
            ai.append(_ai(i))
            i += 1
    return human, ai


def test_jsonl_yaz_oku(tmp_path):
    recs = [{"a": 1, "b": "ç"}, {"a": 2}]
    path = tmp_path / "alt" / "x.jsonl"
    write_jsonl(path, recs)
    assert load_jsonl(path) == recs


def test_yalnizca_ai_karsiligi_olan_insanlar_tutulur():
    human = [_human(0, "a/a"), _human(1, "a/a"), _human(2, "b/b")]
    out, dropped = pair_records(human, [_ai(0), _ai(2)])
    assert {r["pair_id"] for r in out} == {"h0", "h2"}
    assert [r["label"] for r in out].count(0) == [r["label"] for r in out].count(1)
    assert dropped["insan_ai_yok"] == 1
    assert set(out[0]) == {"id", "pair_id", "code", "label", "repo", "source"}
    assert out[0]["id"] == "h0_h" and out[0]["source"] == "human"
    assert out[1]["id"] == f"h0_{model_short(MODEL)}" and out[1]["source"] == MODEL


def test_ai_kaydi_insanin_deposunu_alir():
    out, _ = pair_records([_human(0, "a/a")], [_ai(0)])
    assert {r["repo"] for r in out} == {"a/a"}


def test_insanla_ayni_ai_kodu_ve_cift_atilir():
    h = _human(0, "a/a")
    out, dropped = pair_records([h], [_ai(0, code=h["code"] + "\n")])
    assert out == []
    assert dropped["ai_insanla_ayni"] == 1


def test_tekrar_eden_kodlar_atilir():
    same = "def g():\n    return 1\n"
    human = [_human(0, "a/a", code=same), _human(1, "b/b", code=same), _human(2, "c/c")]
    ai = [_ai(0), _ai(1), _ai(2, code=_ai(0)["code"])]
    out, dropped = pair_records(human, ai)
    assert {r["pair_id"] for r in out} == {"h0"}
    assert dropped["insan_tekrar_kod"] == 1
    assert dropped["ai_tekrar_kod"] == 1


def test_coklu_model_ayni_cifte_eklenir():
    human = [_human(0, "a/a")]
    ai = [_ai(0), _ai(0, code="def f0():\n    return 0 + 0\n", model="qwen2.5-coder:7b"),
          _ai(0, model=MODEL)]
    out, dropped = pair_records(human, ai)
    assert [r["source"] for r in out] == ["human", MODEL, "qwen2.5-coder:7b"]
    assert len({r["id"] for r in out}) == 3
    assert dropped["ai_ayni_id_model"] == 1


def test_depo_kesisimi_yok_ve_cift_ayni_bolmede():
    human, ai = _synthetic()
    splits = split_by_repo(pair_records(human, ai)[0])
    check_splits(splits)
    where = {}
    for name, recs in splits.items():
        for r in recs:
            assert where.setdefault(r["pair_id"], name) == name
    repos = [{r["repo"] for r in recs} for recs in splits.values()]
    assert not (repos[0] & repos[1] or repos[0] & repos[2] or repos[1] & repos[2])


def test_bolme_deterministik():
    records = pair_records(*_synthetic())[0]
    a = split_by_repo(records, seed=7)
    b = split_by_repo(list(records), seed=7)
    assert {s: [r["id"] for r in v] for s, v in a.items()} == \
           {s: [r["id"] for r in v] for s, v in b.items()}


def test_oranlar_yaklasik_tutar():
    human, ai = _synthetic(n_repos=200, seed=1)
    splits = split_by_repo(pair_records(human, ai)[0])
    stats = {row["bolme"]: row["oran"] for row in split_stats(splits)}
    assert stats["train"] == pytest.approx(0.70, abs=0.02)
    assert stats["val"] == pytest.approx(0.15, abs=0.02)
    assert stats["test"] == pytest.approx(0.15, abs=0.02)


def test_acgozlu_atama_buyuk_depoyu_trainde_tutar():
    sizes = {"buyuk": 60, **{f"k{i}": 4 for i in range(10)}}
    assignment = assign_repos(sizes, seed=3)
    assert assignment["buyuk"] == "train"
    counts = {s: sum(n for r, n in sizes.items() if assignment[r] == s)
              for s in ("train", "val", "test")}
    assert counts == {"train": 68, "val": 16, "test": 16}


def test_check_splits_ihlalleri_yakalar():
    a = {"id": "1_h", "pair_id": "1", "code": "x = 1\n", "label": 0, "repo": "r1", "source": "human"}
    b = {"id": "1_m", "pair_id": "1", "code": "x = 2\n", "label": 1, "repo": "r1", "source": "m"}
    c = {"id": "2_h", "pair_id": "2", "code": "y = 1\n", "label": 0, "repo": "r2", "source": "human"}
    d = {"id": "2_m", "pair_id": "2", "code": "y = 2\n", "label": 1, "repo": "r2", "source": "m"}
    e = {"id": "3_h", "pair_id": "3", "code": "z = 1\n", "label": 0, "repo": "r3", "source": "human"}
    f = {"id": "3_m", "pair_id": "3", "code": "z = 2\n", "label": 1, "repo": "r3", "source": "m"}
    check_splits({"train": [a, b], "val": [c, d], "test": [e, f]})
    with pytest.raises(ValueError, match="repo"):
        check_splits({"train": [a, b], "val": [c, d], "test": [{**e, "repo": "r1"}, f]})
    with pytest.raises(ValueError, match="code"):
        check_splits({"train": [a, b], "val": [c, d], "test": [{**e, "code": "x = 1"}, f]})
    with pytest.raises(ValueError, match="tek sinif"):
        check_splits({"train": [a, b], "val": [c], "test": [e, f]})
    g = {**f, "id": "3_m2", "code": "z = 3\n"}
    with pytest.raises(ValueError, match="dengesi"):
        check_splits({"train": [a, b], "val": [c, d], "test": [e, f, g]})
    with pytest.raises(ValueError, match="bos"):
        check_splits({"train": [a, b, c, d, e, f], "val": [], "test": []})


def test_gecersiz_oran_hata_verir():
    with pytest.raises(ValueError):
        assign_repos({"r": 1}, ratios=(0.5, 0.5, 0.5))
