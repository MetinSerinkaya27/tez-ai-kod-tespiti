import json
import random
from collections import Counter

import pytest

from tez.audit import (
    code_tokens,
    drop_near_duplicates,
    duplicate_clusters,
    jaccard,
    length_matched_sample,
    load_jsonl_tolerant,
    modern_markers,
    near_duplicate_pairs,
    pair_similarity,
    quality_flags,
    repo_groups,
    seq_ratio,
    shingles,
    shortcut_features,
    signature_diff,
    single_feature_auc,
)

A = "def f(x, y):\n    z = x + y\n    if z > 10:\n        return z * 2\n    return z\n"
A_RENAMED = "def g(a, b):\n    c = a + b\n    if c > 10:\n        return c * 2\n    return c\n"
B = "def h(items):\n    out = []\n    for it in items:\n        out.append(str(it))\n    return out\n"


# --- benzerlik -------------------------------------------------------------

def test_tokens_skip_layout_and_abstract():
    assert code_tokens("def f():\n    return 1\n") == ["def", "f", "(", ")", ":", "return", "1"]
    assert code_tokens(A, abstract=True) == code_tokens(A_RENAMED, abstract=True)
    assert code_tokens(A) != code_tokens(A_RENAMED)


def test_body_only_drops_signature():
    toks = code_tokens("def f(a,\n      b):\n    return a\n", body_only=True)
    assert toks == ["return", "a"]


def test_seq_ratio_and_jaccard_bounds():
    t = code_tokens(A)
    assert seq_ratio(t, t) == 1.0
    assert jaccard(t, t) == 1.0
    assert jaccard([], []) == 1.0
    assert jaccard({1, 2}, {3}) == 0.0
    assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(0.5)
    assert 0 <= seq_ratio(t, code_tokens(B)) < 0.5


def test_pair_similarity_identical_and_different():
    same = pair_similarity(A, A)
    assert all(v == pytest.approx(1.0) for v in same.values())
    diff = pair_similarity(A, B)
    assert diff["shingle5_jaccard_body"] == 0.0
    assert diff["seq_ratio_body"] < 0.5


def test_shingles_short_sequence():
    assert len(shingles(["a", "b"], k=5)) == 1
    assert shingles([], k=5) == set()
    assert len(shingles(list("abcdef"), k=5)) == 2


# --- yakin kopya -----------------------------------------------------------

def _brute(sets, t):
    return [(i, j) for i in range(len(sets)) for j in range(i + 1, len(sets))
            if jaccard(sets[i], sets[j]) >= t]


@pytest.mark.parametrize("t", [0.3, 0.5, 0.8, 1.0])
def test_near_duplicate_pairs_matches_brute_force(t):
    rng = random.Random(0)
    base = [set(rng.sample(range(60), rng.randint(3, 25))) for _ in range(40)]
    # Bilinen yakin kopyalar ekle: bir eleman degistir.
    sets = base + [(s - {min(s)}) | {1000 + i} for i, s in enumerate(base[:10])]
    got = [(i, j) for i, j, _ in near_duplicate_pairs(sets, t)]
    assert got == _brute(sets, t)


def test_near_duplicate_pairs_query_mode():
    sets = [{1, 2, 3, 4}, {10, 11, 12}, {1, 2, 3, 5}]
    q = [{1, 2, 3, 4}, {99}]
    got = near_duplicate_pairs(sets, 0.6, queries=q)
    assert [(i, j) for i, j, _ in got] == [(0, 0), (0, 2)]


def test_known_code_near_duplicates_and_clusters():
    recs = [
        {"id": "a", "code": A, "repo": "r1"},
        {"id": "b", "code": B, "repo": "r2"},
        {"id": "c", "code": A.replace("10", "11"), "repo": "r3"},
        {"id": "d", "code": A_RENAMED, "repo": "r4"},
    ]
    keep, drop = drop_near_duplicates(recs, threshold=0.5)
    assert [r["id"] for r in drop] == ["c"]
    # Soyut token'larla yeniden adlandirma da yakalanir.
    keep, drop = drop_near_duplicates(recs, threshold=0.5, abstract=True)
    assert [r["id"] for r in keep] == ["a", "b"]
    assert duplicate_clusters(4, [(0, 2, 1.0), (2, 3, 1.0)]) == [0, 1, 0, 0]


def test_repo_groups_merge_connected_repos():
    groups = repo_groups(["x", "y", "z", "y"], [(0, 1, 0.9)])
    assert groups == {"x": "x", "y": "x", "z": "z"}


# --- uzunluk eslestirme ----------------------------------------------------

def _length_records():
    rng = random.Random(1)
    recs = []
    for i in range(300):
        recs.append({"id": f"h{i}", "label": 0, "code": "x\n" * rng.randint(5, 60)})
    for i in range(200):
        recs.append({"id": f"a{i}", "label": 1, "code": "x\n" * rng.randint(5, 25)})
    return recs


def test_length_matched_sample_deterministic():
    recs = _length_records()
    a = length_matched_sample(recs, bin_width=2, seed=7)
    b = length_matched_sample(list(recs), bin_width=2, seed=7)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    c = length_matched_sample(recs, bin_width=2, seed=8)
    assert [r["id"] for r in a] != [r["id"] for r in c]


def test_length_matched_sample_equalizes_distribution():
    recs = _length_records()
    out = length_matched_sample(recs, bin_width=2, seed=7)
    hist = {lab: Counter(len(r["code"].splitlines()) // 2 for r in out if r["label"] == lab)
            for lab in (0, 1)}
    assert hist[0] == hist[1]
    assert sum(hist[0].values()) > 0
    lens = [len(r["code"].splitlines()) for r in out]
    auc = single_feature_auc(lens, [r["label"] for r in out])["auc"]
    assert abs(auc - 0.5) < 0.05


# --- ozellikler / kalite ---------------------------------------------------

def test_shortcut_features_basic():
    code = ("def f(a):\n    import os\n    s = u'x' + \"y\"\n    t = '%s' % a  \n"
            "    print(f'{a}')\n    return s\n")
    f = shortcut_features(code)
    assert f["n_lines"] == 6
    assert f["local_imports"] == 1
    assert f["u_prefix_strings"] == 1
    assert f["percent_format"] == 1
    assert f["fstrings"] == 1
    assert f["print_calls"] == 1
    assert f["trailing_ws_lines"] == 1
    assert f["indent_unit"] == 4


def test_single_feature_auc():
    r = single_feature_auc([1, 2, 3, 4], [0, 0, 1, 1])
    assert r["auc"] == 1.0
    assert single_feature_auc([0, 0, 0, 0], [0, 0, 1, 1])["separability"] == 0.5
    assert single_feature_auc([4, 3, 2, 1], [0, 0, 1, 1])["separability"] == 1.0


def test_modern_markers():
    m = modern_markers(
        "def f(self, p: int) -> str:\n    if (n := len(p)):\n        super().g()\n"
        "    return f'{n}'\n")
    assert m["walrus"] == m["super_noarg"] == m["fstring"] == m["type_hint_signature"] == 1
    assert modern_markers(A)["fstring"] == 0


def test_signature_diff():
    assert signature_diff(A, A.replace("z", "w")) == "ayni"
    assert signature_diff(A, A.replace("f(x, y)", "f(x, q)")) == "parametre_farkli"
    assert signature_diff(A, A.replace("f(x, y)", "f(x, y=1)")) == "varsayilan_farkli"
    assert signature_diff(A, A.replace("f(x, y)", "f(x, y: int)")) == "anotasyon_farkli"


def test_quality_flags():
    q = quality_flags("def f(a, b):\n    return 1\n")
    assert q["tek_deyim_govde"] == q["sabit_donus_govde"] == q["kullanilmayan_parametre"] == 1
    q = quality_flags("def f(a):\n    a.x()\n    a.x()\n    a.x()\n")
    assert q["ayni_satir_tekrari_3+"] == 1 and q["kullanilmayan_parametre"] == 0


def test_load_jsonl_tolerant_skips_partial_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"a": 1}) + "\n\n" + '{"a": 2', encoding="utf-8")
    assert load_jsonl_tolerant(p) == [{"a": 1}]
    assert load_jsonl_tolerant(tmp_path / "yok.jsonl") == []
