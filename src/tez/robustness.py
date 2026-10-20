"""Model-bagimsiz degerlendirme ve saglamlik paketi.

Her dedektor `predict_proba(list[str]) -> np.ndarray` (AI olasiligi) ve
`threshold` (dogrulama kumesinde secilmis sabit esik) sunmalidir. XGBoost
dedektoru ozellik tablosu bekledigi icin `XGBCodeDetector` ile sarmalanir.

Bolumler:
1. Guven araliklari: depo bazli (cluster) bootstrap. Ayni depodaki ornekler ve
   ayni ciftin (insan + AI) uyeleri bagimsiz degildir; cift uyeleri her zaman
   ayni depoda oldugu icin depo duzeyinde yeniden ornekleme her iki bagimliligi
   da kapsar. Karsilastirma icin ornek bazli (iid) bootstrap da raporlanir.
2. Capraz dogrulama (XGBoost): train+val uzerinde depo bazli GroupKFold; her
   katta ic dogrulama (yine depo ayrik) ile model secimi ve esik. Test yok.
3. Kalibrasyon: Brier, ECE (esit genislikte 10 kova), guvenilirlik diyagrami.
4. Saglamlik: black, yerel ad degistirme, bos satir/girinti normallestirme.
   Her donusum AST esdegerligi ile dogrulanir; basarisizsa orijinal kod kalir
   ve oran raporlanir. Donusumden sonra kod `normalize_code`'dan gecer.
5. Kaynak arasi genelleme: kaynak basina metrik + "gorulmeyen uretici" deneyi.
6. Hata analizi: en emin yanlis siniflandirmalar (FP/FN ayri) markdown.
"""
from __future__ import annotations

import ast
import builtins
import io
import tokenize
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from tez.dataset import code_key
from tez.features import features_frame
from tez.metrics import best_threshold, compute_metrics
from tez.normalize import normalize_code
from tez.paths import SEED


class CodeDetector(Protocol):
    threshold: float

    def predict_proba(self, codes: Sequence[str]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# XGBoost sarmalayici
# ---------------------------------------------------------------------------

@dataclass
class XGBCodeDetector:
    """XGBDetector'u ham kod listesi alan arayuze cevirir (+ SHAP ilk-k)."""

    det: object  # tez.xgb_model.XGBDetector
    name: str = "xgb"

    @property
    def threshold(self) -> float:
        return float(self.det.threshold)

    def frame(self, codes: Sequence[str]) -> pd.DataFrame:
        return features_frame(codes)

    def predict_proba(self, codes: Sequence[str]) -> np.ndarray:
        if len(codes) == 0:
            return np.zeros(0)
        return self.det.predict_proba(self.frame(codes))

    def top_features(self, codes: Sequence[str], k: int = 5) -> list[list[tuple[str, float, float]]]:
        """Her kod icin |SHAP|'e gore ilk k (ozellik, deger, shap log-odds)."""
        X = self.frame(codes)[self.det.features]
        sv, _ = self.det.shap_values(X)
        out = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(sv[i]))[:k]
            out.append([(self.det.features[j], float(X.iloc[i, j]), float(sv[i, j]))
                        for j in order])
        return out


# ---------------------------------------------------------------------------
# 1) Bootstrap guven araliklari
# ---------------------------------------------------------------------------

CI_METRICS = ("roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall", "brier")


def fast_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Bootstrap icin hizli metrikler (compute_metrics ile ayni tanimlar)."""
    pred = p >= threshold
    pos = y == 1
    tp = float(np.sum(pred & pos))
    fp = float(np.sum(pred & ~pos))
    fn = float(np.sum(~pred & pos))
    n = len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    two = 0 < pos.sum() < n
    return {
        "roc_auc": float(roc_auc_score(y, p)) if two else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if two else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "accuracy": float(np.mean(pred == pos)),
        "precision": prec,
        "recall": rec,
        "brier": float(np.mean((p - y) ** 2)),
    }


def _cluster_index(groups: Sequence) -> list[np.ndarray]:
    """Her kume (depo) icin ornek indeksleri; sira deterministik (ilk gorulme)."""
    idx: dict = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    return [np.asarray(v) for v in idx.values()]


def bootstrap_samples(groups: Sequence | None, n: int, n_boot: int, seed: int = SEED):
    """Bootstrap indeks dizileri uretir.

    groups verilirse kumeler (depolar) yerine koyarak cekilir ve secilen her
    kumenin tum ornekleri alinir (cluster bootstrap); verilmezse ornek bazli.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, n)
        return
    clusters = _cluster_index(groups)
    k = len(clusters)
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        yield np.concatenate([clusters[j] for j in pick])


def _summarize(point: dict, draws: list[dict], alpha: float) -> dict[str, dict]:
    out = {}
    for m in point:
        vals = np.array([d[m] for d in draws], dtype=float)
        vals = vals[~np.isnan(vals)]
        lo, hi = (np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                  if len(vals) else (np.nan, np.nan))
        out[m] = {"point": float(point[m]), "lo": float(lo), "hi": float(hi),
                  "se": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                  "n_valid": int(len(vals))}
    return out


def bootstrap_ci(y, p, threshold: float, groups: Sequence | None = None,
                 n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05) -> dict[str, dict]:
    """Metrik -> {point, lo, hi, se, n_valid}; yuzdelik (percentile) GA.

    Esik sabittir (val'de secilmis); bootstrap icinde yeniden secilmez.
    Tek sinif iceren cekilislerde AUC tanimsizdir ve atlanir (n_valid).
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    point = fast_metrics(y, p, threshold)
    draws = [fast_metrics(y[i], p[i], threshold)
             for i in bootstrap_samples(groups, len(y), n_boot, seed)]
    return _summarize(point, draws, alpha)


def bootstrap_delta_ci(y, p_ref, p_new, threshold: float, groups: Sequence | None = None,
                       n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05) -> dict[str, dict]:
    """Eslestirilmis bootstrap: (p_new - p_ref) metrik farki icin GA.

    Ayni cekilis indeksleri her iki tahmin kumesine uygulanir.
    """
    y = np.asarray(y, dtype=int)
    a = np.asarray(p_ref, dtype=float)
    b = np.asarray(p_new, dtype=float)

    def diff(i):
        ma, mb = fast_metrics(y[i], a[i], threshold), fast_metrics(y[i], b[i], threshold)
        return {k: mb[k] - ma[k] for k in ma}

    point = diff(np.arange(len(y)))
    draws = [diff(i) for i in bootstrap_samples(groups, len(y), n_boot, seed)]
    return _summarize(point, draws, alpha)


def ci_rows(ci: dict[str, dict], **extra) -> list[dict]:
    return [{**extra, "metric": m, **v} for m, v in ci.items()]


# ---------------------------------------------------------------------------
# 3) Kalibrasyon
# ---------------------------------------------------------------------------

def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Esit genislikte kovalar: n, ortalama tahmin, gercek AI orani."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"bin": k, "lo": edges[k], "hi": edges[k + 1], "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()) if m.any() else float("nan"),
                     "frac_pos": float(y[m].mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    t = t[t["n"] > 0]
    return float((t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["n"].sum())


def calibration_summary(y, p, n_bins: int = 10) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {"brier": float(np.mean((p - y) ** 2)),
            "ece": expected_calibration_error(y, p, n_bins),
            "n_bins": n_bins,
            "base_rate_brier": float(np.mean((y.mean() - y) ** 2))}


def plot_reliability(table: pd.DataFrame, path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    t = table[table["n"] > 0]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="mukemmel kalibrasyon")
    ax.plot(t["mean_pred"], t["frac_pos"], "o-", label="model")
    ax.set_ylabel("Gercek AI orani")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title(title)
    centers = (table["lo"] + table["hi"]) / 2
    axh.bar(centers, table["n"], width=table["hi"] - table["lo"], edgecolor="black")
    axh.set_xlabel("Tahmin edilen AI olasiligi")
    axh.set_ylabel("Ornek")
    axh.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) Perturbasyonlar
# ---------------------------------------------------------------------------

class PerturbationError(ValueError):
    """Donusum uygulanamadi ya da AST esdegerligi bozuldu."""


def _dump(code: str) -> str:
    """Konumsuz AST dokumu. Constant.kind ('u' oneki) anlamsizdir; black onu
    sildigi icin karsilastirmada yok sayilir."""
    tree = ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant):
            n.kind = None
    return ast.dump(tree, include_attributes=False)


def black_format(code: str, line_length: int = 88) -> str:
    """black ile yeniden bicimlendirir (tirnak, bosluk, satir kirma sinyalini siler)."""
    import black

    try:
        out = black.format_str(code, mode=black.Mode(line_length=line_length))
    except Exception as e:  # black.InvalidInput vb.
        raise PerturbationError(f"black basarisiz: {e}") from e
    if _dump(out) != _dump(code):
        raise PerturbationError("black AST'i degistirdi")
    return out


def _string_lines(tokens) -> tuple[set[int], set[int]]:
    """(cok satirli dize icinde kalan satirlar, sonu dize icinde olan satirlar)."""
    inside: set[int] = set()
    open_end: set[int] = set()
    fstack: list[int] = []

    def mark(start: int, end: int):
        if end > start:
            inside.update(range(start + 1, end + 1))
            open_end.update(range(start, end))

    for t in tokens:
        if t.type == tokenize.STRING:
            mark(t.start[0], t.end[0])
        elif t.type == tokenize.FSTRING_START:
            fstack.append(t.start[0])
        elif t.type == tokenize.FSTRING_END and fstack:
            mark(fstack.pop(), t.end[0])
    return inside, open_end


def normalize_layout(code: str, indent: int = 4) -> str:
    """Girintiyi blok duzeyi x `indent` bosluga cevirir, bos satirlari ve satir
    sonu bosluklarini siler. Devam satirlari ayni kayma ile tasinir; cok satirli
    dizelerin icine dokunulmaz.
    """
    code = code.replace("\r\n", "\n").expandtabs(4)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        raise PerturbationError(f"tokenize basarisiz: {e}") from e
    lines = code.split("\n")
    inside, open_end = _string_lines(tokens)

    # Mantiksal satirlar: (ilk satir, son satir, blok duzeyi, eski girinti).
    spans = []
    level, start = 0, None
    skip = (tokenize.NL, tokenize.COMMENT, tokenize.ENDMARKER, tokenize.ENCODING)
    for t in tokens:
        if t.type == tokenize.INDENT:
            level += 1
        elif t.type == tokenize.DEDENT:
            level -= 1
        elif t.type == tokenize.NEWLINE:
            if start is not None:
                spans.append((start[0], t.start[0], start[1], start[2]))
            start = None
        elif t.type not in skip and start is None:
            start = (t.start[0], level, t.start[1])

    new: dict[int, str | None] = {}
    for s, e, lvl, old_col in spans:
        delta = lvl * indent - old_col
        for ln in range(s, e + 1):
            line = lines[ln - 1]
            if ln in inside:
                new[ln] = line
                continue
            if not line.strip():
                new[ln] = None
                continue
            lead = len(line) - len(line.lstrip(" "))
            width = lvl * indent if ln == s else max(0, lead + delta)
            body = line.lstrip(" ")
            new[ln] = " " * width + (body if ln in open_end else body.rstrip())
    out = []
    for ln, line in enumerate(lines, start=1):
        if ln in new:
            if new[ln] is not None:
                out.append(new[ln])
        elif ln in inside:
            out.append(line)
        elif line.strip():
            out.append(line.rstrip())
    result = "\n".join(out).strip("\n") + "\n"
    try:
        same = _dump(result) == _dump(code)
    except SyntaxError as e:
        raise PerturbationError(f"girinti sonrasi parse hatasi: {e}") from e
    if not same:
        raise PerturbationError("girinti normallestirme AST'i degistirdi")
    return result


_BUILTINS = frozenset(dir(builtins))
_KEEP = frozenset({"self", "cls", "_"})


class _LocalCollector(ast.NodeVisitor):
    """Fonksiyon kapsamlarindaki parametre ve yerel atama adlarini toplar."""

    def __init__(self):
        self.names: list[str] = []
        self.excluded: set[str] = set()
        self.func_names: set[str] = set()
        self._func = 0
        self._class_body = False

    def _add(self, name: str | None):
        if name and name not in self.names:
            self.names.append(name)

    def _args(self, args: ast.arguments):
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
            if a is not None:
                self._add(a.arg)

    def _scope(self, node):
        self._args(node.args)
        saved = self._class_body
        self._class_body = False
        self._func += 1
        self.generic_visit(node)
        self._func -= 1
        self._class_body = saved

    def visit_FunctionDef(self, node):
        self.func_names.add(node.name)
        self._scope(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._scope(node)

    def visit_ClassDef(self, node):
        self.excluded.add(node.name)
        saved = self._class_body
        self._class_body = True
        self.generic_visit(node)
        self._class_body = saved

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            if self._class_body or not self._func:
                self.excluded.add(node.id)
            else:
                self._add(node.id)

    def visit_ExceptHandler(self, node):
        if node.name:
            if self._class_body or not self._func:
                self.excluded.add(node.name)
            else:
                self._add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node):
        self.excluded.update(node.names)

    visit_Nonlocal = visit_Global

    def visit_MatchAs(self, node):
        if node.name:
            self.excluded.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node):
        if node.name:
            self.excluded.add(node.name)

    def visit_MatchMapping(self, node):
        if node.rest:
            self.excluded.add(node.rest)
        self.generic_visit(node)


def local_rename_map(code: str, prefix: str = "v") -> dict[str, str]:
    """Yerel ad -> yeni ad (v1, v2, ... ilk gorulme sirasinda; carpisma yok).

    Dokunulmayanlar: self/cls/_, builtin adlari, global/nonlocal, sinif
    govdesinde atanan adlar, match kaliplari, fonksiyon adlari, oznitelikler
    (`obj.x`) ve cagri anahtar kelimeleri (`f(x=1)`).
    """
    tree = ast.parse(code)
    col = _LocalCollector()
    col.visit(tree)
    skip = col.excluded | col.func_names | _KEEP | _BUILTINS
    used = {t.string for t in tokenize.generate_tokens(io.StringIO(code).readline)
            if t.type == tokenize.NAME}
    mapping, k = {}, 0
    for name in col.names:
        if name in skip:
            continue
        k += 1
        while f"{prefix}{k}" in used:
            k += 1
        mapping[name] = f"{prefix}{k}"
    return mapping


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping):
        self.m = mapping

    def visit_Name(self, node):
        node.id = self.m.get(node.id, node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.m.get(node.arg, node.arg)
        self.generic_visit(node)
        return node

    def visit_ExceptHandler(self, node):
        if node.name:
            node.name = self.m.get(node.name, node.name)
        self.generic_visit(node)
        return node


def ast_renamed_dump(code: str, mapping: dict[str, str]) -> str:
    """Koda ait AST'e ad eslemesi uygulanmis halinin dokumu (dogrulama icin)."""
    tree = _Renamer(mapping).visit(ast.parse(code))
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant):
            n.kind = None
    return ast.dump(tree, include_attributes=False)


def rename_locals(code: str, prefix: str = "v") -> str:
    """Yerel degisken/parametre adlarini tutarli bicimde v1, v2, ... yapar.

    Duzen (girinti, bosluk, tirnak) korunur; yalnizca ad tokenlari degisir.
    Sonuc, orijinal AST'e ayni eslemenin uygulanmis haliyle birebir esit olmalidir.
    """
    mapping = local_rename_map(code, prefix)
    if not mapping:
        return code
    tree = ast.parse(code)
    lines = code.split("\n")
    raw = [ln.encode("utf-8") for ln in lines]

    def char_col(lineno: int, byte_col: int) -> int:
        return len(raw[lineno - 1][:byte_col].decode("utf-8"))

    edits: dict[tuple[int, int], str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapping:
            edits[(node.lineno, char_col(node.lineno, node.col_offset))] = node.id
        elif isinstance(node, ast.arg) and node.arg in mapping:
            edits[(node.lineno, char_col(node.lineno, node.col_offset))] = node.arg

    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)
                and n.name in mapping]
    if handlers:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
        for h in handlers:
            last = h.body[0].lineno if h.body else h.end_lineno
            for a, b in zip(toks, toks[1:]):
                if (a.type == tokenize.NAME and a.string == "as" and b.string == h.name
                        and h.lineno <= b.start[0] <= last
                        and b.start >= (h.lineno, char_col(h.lineno, h.col_offset))):
                    edits[b.start] = h.name
                    break
            else:
                raise PerturbationError(f"except adi bulunamadi: {h.name}")

    for (ln, col), name in sorted(edits.items(), reverse=True):
        line = lines[ln - 1]
        if line[col:col + len(name)] != name:
            raise PerturbationError(f"ad konumu tutmadi: {name} @ {ln}:{col}")
        lines[ln - 1] = line[:col] + mapping[name] + line[col + len(name):]
    out = "\n".join(lines)
    try:
        ok = _dump(out) == ast_renamed_dump(code, mapping)
    except SyntaxError as e:
        raise PerturbationError(f"ad degistirme sonrasi parse hatasi: {e}") from e
    if not ok:
        raise PerturbationError("ad degistirme AST esdegerligini bozdu")
    return out


def compose(*fns: Callable[[str], str]) -> Callable[[str], str]:
    def run(code: str) -> str:
        for fn in fns:
            code = fn(code)
        return code
    return run


PERTURBATIONS: dict[str, Callable[[str], str]] = {
    "black": black_format,
    "yeniden_adlandirma": rename_locals,
    "girinti_bosluk": normalize_layout,
    "black+yeniden_adlandirma": compose(rename_locals, black_format),
}


@dataclass
class PerturbResult:
    codes: list[str]
    n: int = 0
    changed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def stats(self) -> dict:
        return {"n": self.n, "changed": self.changed, "failed": self.failed,
                "changed_frac": self.changed / self.n if self.n else 0.0,
                "failed_frac": self.failed / self.n if self.n else 0.0}


def apply_perturbation(codes: Sequence[str], fn: Callable[[str], str]) -> PerturbResult:
    """Donusumu uygular, sonucu normalize_code'dan gecirir.

    Basarisiz (hata ya da normalize sonrasi parse edilemeyen) orneklerde orijinal
    kod korunur; boylece orijinal ve donusmus tahminler eslesik kalir.
    """
    res = PerturbResult(codes=[], n=len(codes))
    for code in codes:
        try:
            new = normalize_code(fn(code))
            if new is None:
                raise PerturbationError("normalize_code sonrasi parse edilemedi")
        except PerturbationError as e:
            res.failed += 1
            if len(res.errors) < 20:
                res.errors.append(str(e)[:200])
            new = code
        res.changed += code_key(new) != code_key(code)
        res.codes.append(new)
    return res


def robustness_report(detector: CodeDetector, codes: Sequence[str], y, groups: Sequence,
                      perturbations: dict[str, Callable[[str], str]] | None = None,
                      p_orig: np.ndarray | None = None, n_boot: int = 1000,
                      seed: int = SEED) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Her donusum icin orijinale gore metrikler + eslestirilmis depo bootstrap GA.

    Esik detector.threshold'dur (val'de secilmis); donusum sonrasi yeniden
    ayarlanmaz. Donus: (satirlar, {ad: olasiliklar}).
    """
    perturbations = PERTURBATIONS if perturbations is None else perturbations
    y = np.asarray(y, dtype=int)
    thr = detector.threshold
    p0 = detector.predict_proba(list(codes)) if p_orig is None else np.asarray(p_orig)
    base = compute_metrics(y, p0, thr)
    rows = [{"perturbation": "orijinal", **base, "n_changed": 0, "n_failed": 0,
             "flip_frac": 0.0, "mean_abs_dp": 0.0}]
    probs = {"orijinal": p0}
    for name, fn in perturbations.items():
        res = apply_perturbation(codes, fn)
        p = detector.predict_proba(res.codes)
        probs[name] = p
        m = compute_metrics(y, p, thr)
        delta = bootstrap_delta_ci(y, p0, p, thr, groups, n_boot, seed)
        row = {"perturbation": name, **m,
               "n_changed": res.changed, "n_failed": res.failed,
               "flip_frac": float(np.mean((p >= thr) != (p0 >= thr))),
               "mean_abs_dp": float(np.mean(np.abs(p - p0)))}
        for k in ("roc_auc", "f1", "accuracy"):
            d = delta[k]
            row[f"d_{k}"] = d["point"]
            row[f"d_{k}_lo"] = d["lo"]
            row[f"d_{k}_hi"] = d["hi"]
        row["ornek_hatalar"] = res.errors[:3]
        rows.append(row)
    return rows, probs


# ---------------------------------------------------------------------------
# 5) Kaynak arasi genelleme
# ---------------------------------------------------------------------------

def ai_sources(records: Sequence[dict]) -> list[str]:
    return sorted({r["source"] for r in records if r["label"] == 1})


def restrict_to_sources(records: Sequence[dict], sources: set[str] | Sequence[str]) -> list[dict]:
    """Yalnizca verilen AI kaynaklari + bu kaynaklarla eslesmis insan ornekleri.

    Insan ornegi, cifti secilen kaynaklardan en az bir AI icerirse tutulur;
    boylece sinif dengesi ve cift yapisi korunur.
    """
    sources = set(sources)
    pairs = {r["pair_id"] for r in records if r["label"] == 1 and r["source"] in sources}
    return [r for r in records
            if (r["label"] == 1 and r["source"] in sources)
            or (r["label"] == 0 and r["pair_id"] in pairs)]


FitFn = Callable[[list[dict], list[dict]], CodeDetector]


def cross_source_eval(train: Sequence[dict], val: Sequence[dict], test: Sequence[dict],
                      fit_fn: FitFn, min_pairs: int = 5) -> list[dict]:
    """"Egitimde gorulmeyen uretici" deneyi.

    Her egitim kaynagi s icin: model yalnizca s (train) ile egitilir, model
    secimi ve esik s (val) ile yapilir; sonra testteki her kaynak t'de
    (t'nin insan eslesleri dahil) degerlendirilir. s == t satiri gorulen,
    digerleri gorulmeyen uretici performansidir. Tek kaynak varsa yalnizca
    gorulen satiri uretilir.
    """
    rows = []
    test_sources = ai_sources(test)
    for s in ai_sources(train):
        tr, va = restrict_to_sources(train, {s}), restrict_to_sources(val, {s})
        n_tr = len({r["pair_id"] for r in tr})
        if n_tr < min_pairs or len({r["label"] for r in va}) < 2:
            rows.append({"train_source": s, "test_source": None, "durum": "atlandi_az_veri",
                         "n_train": len(tr), "n_val": len(va)})
            continue
        det = fit_fn(tr, va)
        for t in test_sources:
            te = restrict_to_sources(test, {t})
            y = np.array([r["label"] for r in te])
            p = det.predict_proba([r["code"] for r in te])
            rows.append({"train_source": s, "test_source": t, "gorulen": s == t,
                         "durum": "tamam", "n_train": len(tr), "n_val": len(va),
                         **compute_metrics(y, p, det.threshold)})
    return rows


def make_xgb_fit_fn(feature_cols: list[str], grid: dict | None = None,
                    n_jobs: int | None = None) -> FitFn:
    """cross_source_eval icin XGBoost egitim fonksiyonu (izgara + val esigi)."""
    from tez.xgb_model import XGBDetector, select_and_fit

    def fit(tr: list[dict], va: list[dict]) -> XGBCodeDetector:
        X_tr = features_frame(r["code"] for r in tr)[feature_cols]
        X_va = features_frame(r["code"] for r in va)[feature_cols]
        y_tr = np.array([r["label"] for r in tr])
        y_va = np.array([r["label"] for r in va])
        model, params, _ = select_and_fit(X_tr, y_tr, X_va, y_va, grid=grid, n_jobs=n_jobs)
        det = XGBDetector(model, feature_cols, 0.5, {"params": params})
        det.threshold = best_threshold(y_va, det.predict_proba(X_va))
        return XGBCodeDetector(det)

    return fit


# ---------------------------------------------------------------------------
# 2) Depo bazli capraz dogrulama (XGBoost)
# ---------------------------------------------------------------------------

def group_cv_xgb(X: pd.DataFrame, y, groups: Sequence, feature_cols: list[str],
                 n_folds: int = 5, inner_val_frac: float = 0.2, grid: dict | None = None,
                 n_jobs: int | None = None, seed: int = SEED) -> list[dict]:
    """train+val uzerinde depo bazli GroupKFold.

    Her katta: dis kat = degerlendirme; kalan depolar GroupShuffleSplit ile ic
    egitim / ic dogrulama olarak bolunur. Izgara secimi, erken durdurma ve esik
    yalnizca ic dogrulamada yapilir; dis kat hicbir secime katilmaz. Uc kumenin
    depo kesisimi bos oldugu dogrulanir. Test kumesi bu fonksiyona hic verilmez.
    """
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    from tez.xgb_model import XGBDetector, select_and_fit

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    X = X[feature_cols].reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for k, (fit_idx, out_idx) in enumerate(gkf.split(X, y, groups)):
        gss = GroupShuffleSplit(n_splits=1, test_size=inner_val_frac, random_state=seed + k)
        tr_rel, va_rel = next(gss.split(fit_idx, y[fit_idx], groups[fit_idx]))
        tr_idx, va_idx = fit_idx[tr_rel], fit_idx[va_rel]
        g_tr, g_va, g_out = (set(groups[i]) for i in (tr_idx, va_idx, out_idx))
        if g_tr & g_va or g_tr & g_out or g_va & g_out:
            raise RuntimeError(f"Kat {k}: depo sizintisi")
        if len(np.unique(y[va_idx])) < 2 or len(np.unique(y[out_idx])) < 2:
            rows.append({"fold": k, "durum": "atlandi_tek_sinif"})
            continue
        model, params, _ = select_and_fit(X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx],
                                          y[va_idx], grid=grid, n_jobs=n_jobs)
        det = XGBDetector(model, feature_cols, 0.5)
        thr = best_threshold(y[va_idx], det.predict_proba(X.iloc[va_idx]))
        p = det.predict_proba(X.iloc[out_idx])
        rows.append({"fold": k, "durum": "tamam", "n_train": len(tr_idx),
                     "n_inner_val": len(va_idx), "n_eval": len(out_idx),
                     "repos_eval": len(g_out), "params": params,
                     **compute_metrics(y[out_idx], p, thr),
                     "brier": float(np.mean((p - y[out_idx]) ** 2))})
    return rows


def cv_summary(rows: list[dict], metrics=("roc_auc", "pr_auc", "f1", "accuracy", "brier")) -> dict:
    ok = [r for r in rows if r.get("durum") == "tamam"]
    out = {"n_folds_ok": len(ok)}
    for m in metrics:
        v = np.array([r[m] for r in ok], dtype=float)
        out[m] = {"mean": float(np.nanmean(v)) if len(v) else float("nan"),
                  "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"),
                  "min": float(np.nanmin(v)) if len(v) else float("nan"),
                  "max": float(np.nanmax(v)) if len(v) else float("nan")}
    return out


# ---------------------------------------------------------------------------
# 6) Hata analizi
# ---------------------------------------------------------------------------

def confident_errors(records: Sequence[dict], p, threshold: float, k: int = 20) -> dict[str, list[int]]:
    """En emin yanlislar: FP (insan, en yuksek p) ve FN (AI, en dusuk p) indeksleri."""
    p = np.asarray(p, dtype=float)
    y = np.array([r["label"] for r in records])
    pred = p >= threshold
    fp = np.where((y == 0) & pred)[0]
    fn = np.where((y == 1) & ~pred)[0]
    return {"FP": [int(i) for i in fp[np.argsort(-p[fp], kind="stable")][:k]],
            "FN": [int(i) for i in fn[np.argsort(p[fn], kind="stable")][:k]]}


def error_markdown(records: Sequence[dict], p, threshold: float, detector=None,
                   k: int = 20, shap_k: int = 5) -> str:
    """FP/FN listelerini kod + olasilik + (varsa) ilk SHAP ozellikleriyle yazar."""
    p = np.asarray(p, dtype=float)
    errs = confident_errors(records, p, threshold, k)
    parts = [f"# En emin yanlis siniflandirmalar\n\nEsik: {threshold:.4f} (val'de secildi). "
             "FP: insan kodu AI dendi (en yuksek olasilik once). "
             "FN: AI kodu insan dendi (en dusuk olasilik once).\n"]
    for kind, idx in errs.items():
        parts.append(f"\n## {kind} ({len(idx)} ornek)\n")
        tops = (detector.top_features([records[i]["code"] for i in idx], shap_k)
                if idx and detector is not None and hasattr(detector, "top_features") else None)
        for j, i in enumerate(idx):
            r = records[i]
            parts.append(f"\n### {j + 1}. `{r['id']}` - p(AI) = {p[i]:.4f}\n\n"
                         f"depo: `{r.get('repo')}`, kaynak: `{r.get('source')}`, "
                         f"etiket: {r['label']}\n")
            if tops:
                parts.append("\n| ozellik | deger | SHAP (log-odds) |\n|---|---:|---:|\n")
                parts += [f"| {f} | {v:.4g} | {s:+.4f} |\n" for f, v, s in tops[j]]
            parts.append(f"\n```python\n{r['code'].rstrip()}\n```\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Sizinti kontrolu
# ---------------------------------------------------------------------------

def leakage_report(splits: dict[str, Sequence[dict]]) -> dict[str, int]:
    """Bolmeler arasi ortak depo / cift / kod sayilari (hepsi 0 olmali)."""
    out = {}
    names = list(splits)
    for field_, key in (("repo", None), ("pair_id", None), ("code", code_key)):
        sets = {s: {key(r[field_]) if key else r[field_] for r in splits[s]} for s in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out[f"{field_}:{a}&{b}"] = len(sets[a] & sets[b])
    return out
