"""Veri denetimi: ezber, yakin kopya, kisa yol sinyalleri, donem karistiricisi ve
uretim kalitesi icin saf (yan etkisiz) fonksiyonlar.

Amac, modelin "yanlis nedenlerle" (sizinti, etiket gurultusu, uzunluk/bicim gibi
kisa yollar) yuksek dogruluk almasini olcmek. Bu moduldeki filtreler veri
hattina baglanmaz; 03_build_dataset.py'ye nasil eklenecegi rapordadir.

Tum fonksiyonlar `normalize_code` ciktisi (docstring/yorum silinmis) uzerinde
calisir. Yalnizca stdlib + numpy/sklearn kullanilir.
"""
from __future__ import annotations

import ast
import difflib
import io
import json
import keyword
import math
import random
import re
import tokenize
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_SKIP_TOKENS = {
    tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
    tokenize.COMMENT, tokenize.ENDMARKER,
}


# --------------------------------------------------------------------------
# Okuma
# --------------------------------------------------------------------------

def load_jsonl_tolerant(path: str | Path) -> list[dict]:
    """JSONL okur; yarim yazilmis (bozuk) satirlari atlar.

    ai.jsonl uretim sirasinda yazildigi icin son satir yarim olabilir.
    """
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# --------------------------------------------------------------------------
# Token dizileri ve benzerlik
# --------------------------------------------------------------------------

def _main_func(tree: ast.Module) -> ast.AST | None:
    return next((n for n in tree.body if isinstance(n, _FUNC_TYPES)), None)


def body_start_line(code: str) -> int:
    """Fonksiyon govdesinin ilk deyiminin (1 tabanli) satir numarasi; yoksa 1."""
    try:
        fn = _main_func(ast.parse(code))
    except SyntaxError:
        return 1
    return fn.body[0].lineno if fn is not None and fn.body else 1


def code_tokens(code: str, abstract: bool = False, body_only: bool = False) -> list[str]:
    """Anlamli token dizisi (bosluk/girinti/yorum token'lari haric).

    abstract=True: anahtar kelime olmayan isimler "ID", sayilar "NUM", stringler
    "STR" olur (tip-2 klon / yeniden adlandirma tespiti icin).
    body_only=True: imza satir(lar)i atilir; insan ve AI ayni imzayla
    basladigi icin benzerlik imzadan sisirilmesin.
    """
    start = body_start_line(code) if body_only else 1
    out: list[str] = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code.split()
    fstring_depth = 0
    for tok in toks:
        if tok.type in _SKIP_TOKENS or tok.start[0] < start:
            continue
        s = tok.string
        if abstract:
            if tok.type == tokenize.NAME and not keyword.iskeyword(s):
                s = "ID"
            elif tok.type == tokenize.NUMBER:
                s = "NUM"
            elif tok.type == tokenize.STRING:
                s = "STR"
            elif tok.type == tokenize.FSTRING_START:
                fstring_depth += 1
                s = "FSTR"
            elif tok.type == tokenize.FSTRING_END:
                fstring_depth -= 1
                continue
            elif fstring_depth and tok.type == tokenize.FSTRING_MIDDLE:
                continue
        out.append(s)
    return out


def seq_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    """difflib SequenceMatcher orani (autojunk kapali; 0..1)."""
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def jaccard(a: Iterable, b: Iterable) -> float:
    """Kume Jaccard benzerligi; iki bos kume icin 1."""
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def shingles(tokens: Sequence[str], k: int = 5) -> set[int]:
    """Token k-gram'larinin (hash'lenmis) kumesi. Kisa diziler tek shingle olur."""
    if not tokens:
        return set()
    if len(tokens) < k:
        return {hash(tuple(tokens))}
    return {hash(tuple(tokens[i:i + k])) for i in range(len(tokens) - k + 1)}


def pair_similarity(human_code: str, ai_code: str, k: int = 5) -> dict[str, float]:
    """Bir AI kodunun eslestigi insan koduna benzerligi (tum kod ve yalniz govde)."""
    out = {}
    for suffix, body in (("", False), ("_body", True)):
        h = code_tokens(human_code, body_only=body)
        a = code_tokens(ai_code, body_only=body)
        out[f"seq_ratio{suffix}"] = seq_ratio(h, a)
        out[f"token_jaccard{suffix}"] = jaccard(h, a)
        out[f"shingle{k}_jaccard{suffix}"] = jaccard(shingles(h, k), shingles(a, k))
    return out


# --------------------------------------------------------------------------
# Yakin kopya bulucu (prefix filtreli tam Jaccard self-join, O(n^2) degil)
# --------------------------------------------------------------------------

def _prefix_len(n: int, t: float) -> int:
    # J(x,y) >= t ise |x & y| >= t*max(|x|,|y|) >= ceil(t*|x|); bu yuzden nadirden
    # sik'a sirali iki kumenin ilk n - ceil(t*n) + 1 elemani kesismek zorunda.
    return n - math.ceil(t * n - 1e-9) + 1


def near_duplicate_pairs(
    sets: Sequence[set],
    threshold: float,
    queries: Sequence[set] | None = None,
) -> list[tuple[int, int, float]]:
    """Jaccard >= threshold olan ciftleri tam (yaklasiksiz) bulur.

    Prefix filtreleme (AllPairs): elemanlar global sikliga gore nadirden sika
    siralanir; her kumenin yalnizca on eki indekslenir ve yalnizca on ekleri
    kesisen + uzunluk filtresini (t*|x| <= |y| <= |x|/t) gecen adaylar icin
    gercek Jaccard hesaplanir. Sonuc kesindir (aday kacirilmaz).

    queries None ise self-join: (i, j, J) ile i < j. Verilirse her sorgu kumesi
    `sets` ile karsilastirilir: (sorgu_indeksi, set_indeksi, J).
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold (0, 1] araliginda olmali")
    freq: Counter = Counter()
    for s in sets:
        freq.update(s)
    if queries is not None:
        for q in queries:
            freq.update(q)

    def prefix(s: set) -> list:
        order = sorted(s, key=lambda e: (freq[e], e))
        return order[:_prefix_len(len(order), threshold)]

    index: dict = defaultdict(list)
    for i, s in enumerate(sets):
        if s:
            for e in prefix(s):
                index[e].append(i)

    probes = sets if queries is None else queries
    out: list[tuple[int, int, float]] = []
    for qi, q in enumerate(probes):
        if not q:
            continue
        n = len(q)
        cands: set[int] = set()
        for e in prefix(q):
            cands.update(index.get(e, ()))
        for j in cands:
            if queries is None and j <= qi:
                continue
            m = len(sets[j])
            if m < threshold * n - 1e-9 or n < threshold * m - 1e-9:
                continue
            inter = len(q & sets[j])
            jac = inter / (n + m - inter)
            if jac >= threshold - 1e-12:
                out.append((qi, j, jac))
    out.sort()
    return out


class _UnionFind:
    def __init__(self, items: Iterable):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministik kok: kucuk olan.
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def duplicate_clusters(n: int, pairs: Iterable[tuple[int, int, float]]) -> list[int]:
    """Ciftlerden baglantili bilesenler; her indeks icin kume temsilcisi (en kucuk indeks)."""
    uf = _UnionFind(range(n))
    for i, j, _ in pairs:
        uf.union(i, j)
    return [uf.find(i) for i in range(n)]


def drop_near_duplicates(
    records: Sequence[dict],
    threshold: float = 0.8,
    k: int = 5,
    abstract: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Yakin kopya kumelerinden yalnizca ilk (girdi sirasinda) kaydi tutar.

    Donus: (tutulanlar, atilanlar). Girdi sirasi korunur; deterministiktir.
    """
    sets = [shingles(code_tokens(r["code"], abstract=abstract), k) for r in records]
    rep = duplicate_clusters(len(records), near_duplicate_pairs(sets, threshold))
    keep, drop = [], []
    for i, r in enumerate(records):
        (keep if rep[i] == i else drop).append(r)
    return keep, drop


def repo_groups(repos: Sequence[str], pairs: Iterable[tuple[int, int, float]]) -> dict[str, str]:
    """Yakin kopya ile birbirine bagli depolari tek bir bolme birimine toplar.

    repos[i] i. kaydin deposu. Donus: depo -> grup temsilcisi (alfabetik en kucuk
    depo). split_by_repo'da "repo" yerine bu grup anahtari kullanilirsa depolar
    arasi yakin kopyalar farkli bolmelere dusemez.
    """
    uf = _UnionFind(sorted(set(repos)))
    for i, j, _ in pairs:
        uf.union(repos[i], repos[j])
    return {r: uf.find(r) for r in uf.parent}


# --------------------------------------------------------------------------
# Kisa yol sinyalleri, donem isaretleri, uretim kalitesi
# --------------------------------------------------------------------------

_NONASCII_RE = re.compile(r"[^\x00-\x7f]")


def _strings(tokens) -> list[tokenize.TokenInfo]:
    return [t for t in tokens if t.type in (tokenize.STRING, tokenize.FSTRING_START)]


def _indent_unit(code: str) -> int:
    """Govdedeki ic ice bloklar arasi en yaygin girinti farki; tek seviyede taban."""
    widths = []
    for line in code.splitlines():
        if line.strip():
            widths.append(len(line) - len(line.lstrip(" ")))
    levels = sorted(set(widths))
    diffs = [b - a for a, b in zip(levels, levels[1:])]
    return Counter(diffs).most_common(1)[0][0] if diffs else 0


def shortcut_features(code: str) -> dict[str, float]:
    """Tek basina sinifi ele verebilecek basit (stil disi/yuzeysel) ozellikler."""
    lines = code.splitlines()
    n_lines = max(1, len(lines))
    tree = ast.parse(code)
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    strs = _strings(toks)
    quotes = Counter()
    u_prefix = 0
    for t in strs:
        prefix = re.match(r"[rRbBuUfF]*", t.string).group(0)
        u_prefix += "u" in prefix.lower()
        q = t.string[len(prefix):][:1]
        if q in "'\"" and q:
            quotes["single" if q == "'" else "double"] += 1
    nodes = list(ast.walk(tree))
    fn = _main_func(tree)
    local_imports = sum(
        isinstance(n, (ast.Import, ast.ImportFrom)) for n in (ast.walk(fn) if fn else [])
    )
    print_calls = sum(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
        for n in nodes
    )
    pct_fmt = sum(
        isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
        and isinstance(n.left, (ast.Constant, ast.JoinedStr))
        and isinstance(getattr(n.left, "value", ""), str)
        for n in nodes
    )
    dot_format = sum(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "format" and isinstance(n.func.value, ast.Constant)
        for n in nodes
    )
    n_q = quotes["single"] + quotes["double"]
    return {
        "n_lines": len(lines),
        "n_chars": len(code),
        "n_tokens": sum(t.type not in _SKIP_TOKENS for t in toks),
        "max_line_len": max((len(x) for x in lines), default=0),
        "indent_unit": _indent_unit(code),
        "single_quote_frac": quotes["single"] / n_q if n_q else 0.5,
        "has_strings": float(n_q > 0),
        "blank_line_frac": sum(not x.strip() for x in lines) / n_lines,
        "trailing_ws_lines": sum(x != x.rstrip() for x in lines),
        "has_tab": float("\t" in code),
        "non_ascii_chars": len(_NONASCII_RE.findall(code)),
        "print_calls": print_calls,
        "u_prefix_strings": u_prefix,
        "percent_format": pct_fmt,
        "dot_format": dot_format,
        "fstrings": sum(isinstance(n, ast.JoinedStr) for n in nodes),
        "local_imports": local_imports,
        "backslash_cont": sum(x.rstrip().endswith("\\") for x in lines),
        "semicolons": sum(t.type == tokenize.OP and t.string == ";" for t in toks),
        "n_returns": sum(isinstance(n, ast.Return) for n in nodes),
        "self_first_arg": float(bool(fn and fn.args.args and fn.args.args[0].arg in ("self", "cls"))),
    }


def single_feature_auc(values: Sequence[float], labels: Sequence[int]) -> dict:
    """Tek ozelligin ROC-AUC'u ve yon-bagimsiz ayirt edicilik max(AUC, 1-AUC)."""
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2 or len(set(values)) < 2:
        return {"auc": 0.5, "separability": 0.5}
    auc = float(roc_auc_score(labels, values))
    return {"auc": auc, "separability": max(auc, 1 - auc)}


def modern_markers(code: str) -> dict[str, int]:
    """2019 sonrasi / modern Python isaretleri (var/yok, 0/1)."""
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    fn = _main_func(tree)
    sig_ann = bool(fn) and (
        fn.returns is not None
        or any(a.annotation is not None for a in
               fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs
               + [x for x in (fn.args.vararg, fn.args.kwarg) if x])
    )
    names = {n.id for n in nodes if isinstance(n, ast.Name)}
    attrs = {n.attr for n in nodes if isinstance(n, ast.Attribute)}
    imported = set()
    for n in nodes:
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    return {
        "fstring": int(any(isinstance(n, ast.JoinedStr) for n in nodes)),
        "type_hint_signature": int(sig_ann),
        "type_hint_body": int(any(isinstance(n, ast.AnnAssign) for n in nodes)),
        "walrus": int(any(isinstance(n, ast.NamedExpr) for n in nodes)),
        "super_noarg": int(any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "super" and not n.args for n in nodes)),
        "pathlib": int("pathlib" in imported or "Path" in names),
        "dataclass": int("dataclass" in names or "dataclasses" in imported
                         or "dataclass" in attrs),
        "async_await": int(any(isinstance(n, (ast.AsyncFunctionDef, ast.Await,
                                              ast.AsyncFor, ast.AsyncWith)) for n in nodes)),
        "match_stmt": int(any(isinstance(n, ast.Match) for n in nodes)),
        "typing_names": int(bool(names & {"Optional", "List", "Dict", "Tuple", "Union",
                                           "Any", "Callable", "Iterable"})
                            or "typing" in imported),
    }


def signature_diff(human_code: str, ai_code: str) -> str:
    """AI imzasi insan imzasindan farkli mi? 'ayni' ya da fark turu."""
    def sig(code):
        fn = _main_func(ast.parse(code))
        return fn

    h, a = sig(human_code), sig(ai_code)
    if h is None or a is None:
        return "fonksiyon_yok"
    if ast.dump(h.args) == ast.dump(a.args) and ast.dump(h.returns or ast.Pass()) == \
            ast.dump(a.returns or ast.Pass()) and type(h) is type(a):
        return "ayni"
    hn = [x.arg for x in h.args.posonlyargs + h.args.args + h.args.kwonlyargs]
    an = [x.arg for x in a.args.posonlyargs + a.args.args + a.args.kwonlyargs]
    if hn != an:
        return "parametre_farkli"
    if [ast.dump(d) for d in h.args.defaults + h.args.kw_defaults if d] != \
            [ast.dump(d) for d in a.args.defaults + a.args.kw_defaults if d]:
        return "varsayilan_farkli"
    if (h.args.vararg is None) != (a.args.vararg is None) or \
            (h.args.kwarg is None) != (a.args.kwarg is None):
        return "star_arg_farkli"
    return "anotasyon_farkli"


def quality_flags(code: str) -> dict[str, int]:
    """Sozdizimi gecerli ama supheli uretim isaretleri."""
    tree = ast.parse(code)
    fn = _main_func(tree)
    if fn is None:
        return {"fonksiyon_yok": 1}
    body = fn.body
    lines = [x.strip() for x in code.splitlines() if x.strip()]
    max_run = run = 1
    for prev, cur in zip(lines, lines[1:]):
        run = run + 1 if cur == prev else 1
        max_run = max(max_run, run)
    params = {x.arg for x in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    params |= {x.arg for x in (fn.args.vararg, fn.args.kwarg) if x}
    params -= {"self", "cls"}
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    n_pass = sum(isinstance(n, ast.Pass) for n in ast.walk(fn))
    only_const_return = len(body) == 1 and isinstance(body[0], ast.Return) and (
        body[0].value is None or isinstance(body[0].value, ast.Constant))
    return {
        "tek_deyim_govde": int(len(body) == 1),
        "sabit_donus_govde": int(only_const_return),
        "pass_var": int(n_pass > 0),
        "ayni_satir_tekrari_3+": int(max_run >= 3),
        "kullanilmayan_parametre": int(bool(params - used)),
        "not_implemented": int("NotImplementedError" in code),
        "ellipsis": int(any(isinstance(n, ast.Constant) and n.value is Ellipsis
                            for n in ast.walk(fn))),
    }


# --------------------------------------------------------------------------
# Uzunluk eslestirmeli orneklem
# --------------------------------------------------------------------------

def length_matched_sample(
    records: Sequence[dict],
    length_fn: Callable[[dict], float] = lambda r: len(r["code"].splitlines()),
    bin_width: float = 1,
    seed: int = 42,
    label_key: str = "label",
) -> list[dict]:
    """Iki sinifin uzunluk dagilimini esitleyen deterministik alt orneklem.

    Her kayit floor(uzunluk / bin_width) kovasina konur; her kovada iki siniftan
    da min(n_0, n_1) kayit (id'ye gore siralanip seed ile karistirilarak) secilir.
    Sonucta her kovada sinif sayilari esittir, yani kova duzeyinde uzunluk
    dagilimlari aynidir. Cikti girdi sirasini korur.
    """
    if bin_width <= 0:
        raise ValueError("bin_width pozitif olmali")
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        b = math.floor(length_fn(r) / bin_width)
        buckets[(b, r[label_key])].append(i)
    labels = sorted({r[label_key] for r in records})
    if len(labels) != 2:
        return list(records)
    keep: set[int] = set()
    for b in sorted({b for b, _ in buckets}):
        groups = [buckets.get((b, lab), []) for lab in labels]
        m = min(len(g) for g in groups)
        if not m:
            continue
        for lab, g in zip(labels, groups):
            g = sorted(g, key=lambda i: str(records[i].get("id", i)))
            random.Random(f"{seed}-{b}-{lab}").shuffle(g)
            keep.update(g[:m])
    return [r for i, r in enumerate(records) if i in keep]


def percentiles(values: Sequence[float], qs=(5, 25, 50, 75, 95)) -> dict[str, float]:
    """Basit yuzdelik ozeti (numpy, lineer interpolasyon)."""
    import numpy as np

    if not len(values):
        return {f"p{q}": float("nan") for q in qs}
    arr = np.asarray(values, dtype=float)
    out = {f"p{q}": float(np.percentile(arr, q)) for q in qs}
    out["mean"] = float(arr.mean())
    return out
