"""AST stilometrisi: tek bir fonksiyondan uslupsal ozellik cikarimi.

Girdi `normalize_code` ciktisidir (docstring ve yorumlar silinmis). Ozellikler
SHAP ile aciklanacagi icin insanin okuyabilecegi isimlerdedir ve dort gruba
ayrilir: AST yapisi, tanimlayicilar, yerlesim/bicim, deyimsel tercihler.
Oranlar uzunluga gore normalize edilir (dugum, deyim, satir veya token basina);
ham boyut ozellikleri bilincli olarak az tutulmustur.

Not: CodeSearchNet'teki metotlarda `def` satiri sola yaslanmis, govde ise
orijinal (8 bosluk) girintisini korumustur. Bu kaynak sizintisini onlemek icin
girinti birimi ic ice bloklar arasi farktan olculur ve satir uzunluklari
govdedeki fazla girinti cikarilarak hesaplanir.
"""
from __future__ import annotations

import ast
import io
import keyword
import re
import statistics
import tokenize
import warnings
from collections import Counter
from collections.abc import Iterable

import pandas as pd

# Eski kodlardaki gecersiz kacis dizileri icin uyarilari sustur.
warnings.filterwarnings("ignore", category=SyntaxWarning)

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
# Not: |r|>0.9 tekrarlar (raise, f-string dugumu, medyan isim uzunlugu) ile
# seyrek/sabit ya da kaynaga ozgu ozellikler (dekorator: CSN'de hep kesik,
# async, yalniz-anahtar parametre) bilincli olarak alinmadi.
# Load/Store ve operator dugumleri yapisal bilgi tasimaz; sayima katilmaz.
_SKIP_NODES = (ast.expr_context, ast.boolop, ast.operator, ast.unaryop, ast.cmpop)
_BLOCK_NODES = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar,
    ast.With, ast.AsyncWith, ast.Match,
)

# Goreli sikligi (toplam dugum icindeki pay) olculen dugum tipleri.
_NODE_GROUPS: dict[str, tuple[type, ...]] = {
    "if": (ast.If,),
    "for": (ast.For, ast.AsyncFor),
    "while": (ast.While,),
    "try": (ast.Try, ast.TryStar),
    "with": (ast.With, ast.AsyncWith),
    "return": (ast.Return,),
    "call": (ast.Call,),
    "attribute": (ast.Attribute,),
    "subscript": (ast.Subscript,),
    "slice": (ast.Slice,),
    "name": (ast.Name,),
    "constant": (ast.Constant,),
    "listcomp": (ast.ListComp,),
    "dict_set_comp": (ast.DictComp, ast.SetComp),
    "genexp": (ast.GeneratorExp,),
    "lambda": (ast.Lambda,),
    "boolop": (ast.BoolOp,),
    "compare": (ast.Compare,),
    "binop": (ast.BinOp,),
    "unaryop": (ast.UnaryOp,),
    "ifexp": (ast.IfExp,),
    "assign": (ast.Assign,),
    "augassign": (ast.AugAssign,),
    "annassign": (ast.AnnAssign,),
    "raise": (ast.Raise,),
    "assert": (ast.Assert,),
    "yield": (ast.Yield, ast.YieldFrom),
    "await": (ast.Await,),
    "tuple": (ast.Tuple,),
    "list": (ast.List,),
    "dict": (ast.Dict,),
    "starred": (ast.Starred,),
    "expr_stmt": (ast.Expr,),
    "pass": (ast.Pass,),
    "break_continue": (ast.Break, ast.Continue),
    "delete": (ast.Delete,),
    "import": (ast.Import, ast.ImportFrom),
}

_LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical"}
_LOG_OWNERS = {"logging", "logger", "log", "_logger", "_log", "LOGGER", "LOG"}
_REDUCE_BUILTINS = {"sum", "any", "all", "min", "max", "sorted"}

# Binary operatorler (bosluk orani icin); "=" ayrica ele alinir.
_BIN_OPS = {
    "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "//", "%", "**",
    "|", "&", "^", "<<", ">>", "+=", "-=", "*=", "/=", "//=", "%=", "**=",
    "|=", "&=", "^=", "<<=", ">>=", ":=",
}
_OPEN = {"(", "[", "{"}
_CLOSE = {")", "]", "}"}
_OPERAND_TYPES = {tokenize.NAME, tokenize.NUMBER, tokenize.STRING, tokenize.FSTRING_END}

_CAMEL_RE = re.compile(r"^_*[a-z][a-z0-9]*[A-Z]")
_SNAKE_RE = re.compile(r"^_*[a-z][a-z0-9]*(_[a-z0-9]+)+_*$")
_UPPER_RE = re.compile(r"^_*[A-Z][A-Z0-9]*(_[A-Z0-9]+)*_*$")

_STRUCT = [
    "num_nodes", "num_statements", "max_depth", "mean_depth", "mean_branching",
    "max_block_nesting", "cyclomatic_complexity", "cyclomatic_per_stmt",
    "nested_function_count", "return_count", "early_return_count",
    "early_return_frac", "top_level_stmt_frac",
    "statements_per_line",
]
_IDENT = [
    "param_count", "param_annotation_frac", "param_default_frac",
    "has_return_annotation", "has_varargs", "has_kwargs",
    "local_name_count", "local_name_per_stmt", "local_name_mean_len",
    "local_name_max_len", "local_name_single_char_frac",
    "local_name_snake_frac", "local_name_camel_frac", "local_name_upper_frac",
    "local_name_digit_frac", "local_name_underscore_prefix_frac",
    "name_unique_frac", "attr_name_mean_len",
]
_FMT = [
    "num_lines", "line_len_mean", "line_len_max", "line_len_std",
    "line_over79_frac", "blank_line_frac", "trailing_ws_frac", "indent_unit",
    "indent_consistency", "binop_spaced_frac", "assign_spaced_frac",
    "kwarg_eq_spaced_frac", "comma_space_frac", "bracket_inner_space_frac",
    "trailing_comma_rate", "single_quote_frac", "string_per_line",
    "fstring_rate", "format_call_rate", "percent_format_rate",
    "backslash_continuation_rate", "bracket_line_break_rate",
    "tokens_per_line",
]
_IDIOM = [
    "is_none", "eq_none", "compare_bool_literal", "enumerate", "range_len",
    "zip", "isinstance", "type_compare", "with_open", "open_without_with",
    "for_append", "comprehension", "print", "logging", "raise_from",
    "bare_except", "broad_except", "else_after_return",
    "dict_get", "dict_items", "str_join", "len_compare", "return_none_explicit",
    "builtin_reduce", "super_call", "self_attr",
    "try_finally", "star_args_call",
]

FEATURE_NAMES: list[str] = (
    [f"node_{k}_frac" for k in _NODE_GROUPS]
    + _STRUCT
    + _IDENT
    + [f"fmt_{k}" for k in _FMT]
    + [f"idiom_{k}_rate" for k in _IDIOM]
)


def _div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def _main_function(tree: ast.Module) -> ast.AST:
    for node in tree.body:
        if isinstance(node, _FUNC_TYPES):
            return node
    return tree


def _depth_stats(root: ast.AST) -> tuple[int, float, float]:
    """(maksimum derinlik, ortalama derinlik, ic dugumlerde ortalama dallanma)."""
    depths: list[int] = []
    branching: list[int] = []
    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        depths.append(d)
        kids = [c for c in ast.iter_child_nodes(node) if not isinstance(c, _SKIP_NODES)]
        if kids:
            branching.append(len(kids))
        stack.extend((c, d + 1) for c in kids)
    return max(depths), statistics.fmean(depths), _div(sum(branching), len(branching))


def _block_nesting(node: ast.AST, level: int = 0) -> int:
    best = level
    for child in ast.iter_child_nodes(node):
        inner = level + 1 if isinstance(child, _BLOCK_NODES) else level
        best = max(best, _block_nesting(child, inner))
    return best


def _cyclomatic(nodes: list[ast.AST]) -> int:
    score = 1
    for n in nodes:
        if isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp,
                          ast.ExceptHandler, ast.match_case)):
            score += 1
        elif isinstance(n, ast.BoolOp):
            score += len(n.values) - 1
        elif isinstance(n, ast.comprehension):
            score += 1 + len(n.ifs)
    return score


def _is_none(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and n.value is None


def _call_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _is_log_call(call: ast.Call) -> bool:
    f = call.func
    if not (isinstance(f, ast.Attribute) and f.attr in _LOG_METHODS):
        return False
    owner = f.value
    if isinstance(owner, ast.Name):
        return owner.id in _LOG_OWNERS
    if isinstance(owner, ast.Attribute):
        return owner.attr in _LOG_OWNERS
    return False


def _ends_with_jump(body: list[ast.stmt]) -> bool:
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise, ast.Continue, ast.Break))


def _ast_features(tree: ast.Module, n_lines: int) -> dict[str, float]:
    fn = _main_function(tree)
    nodes = [n for n in ast.walk(fn) if not isinstance(n, _SKIP_NODES)]
    total = len(nodes)
    stmts = [n for n in nodes if isinstance(n, ast.stmt) and n is not fn]
    n_stmt = max(1, len(stmts))
    f: dict[str, float] = {}

    # --- 1. AST yapisi ---
    counts = Counter(type(n) for n in nodes)
    for key, types in _NODE_GROUPS.items():
        f[f"node_{key}_frac"] = _div(sum(counts[t] for t in types), total)

    max_d, mean_d, branch = _depth_stats(fn)
    returns = [n for n in nodes if isinstance(n, ast.Return)]
    body = getattr(fn, "body", [])
    final_return = 1 if body and isinstance(body[-1], ast.Return) else 0
    early = max(0, len(returns) - final_return)
    cc = _cyclomatic(nodes)
    f.update({
        "num_nodes": total,
        "num_statements": len(stmts),
        "max_depth": max_d,
        "mean_depth": mean_d,
        "mean_branching": branch,
        "max_block_nesting": _block_nesting(fn),
        "cyclomatic_complexity": cc,
        "cyclomatic_per_stmt": cc / n_stmt,
        "nested_function_count": sum(
            1 for n in nodes if isinstance(n, (*_FUNC_TYPES, ast.ClassDef)) and n is not fn
        ),
        "return_count": len(returns),
        "early_return_count": early,
        "early_return_frac": _div(early, len(returns)),
        "top_level_stmt_frac": len(body) / n_stmt if body else 0.0,
        "statements_per_line": len(stmts) / max(1, n_lines),
    })

    # --- 2. Tanimlayicilar ---
    args = getattr(fn, "args", None)
    params: list[ast.arg] = []
    if args is not None:
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if params and params[0].arg in ("self", "cls"):
            params = params[1:]
        n_defaults = len(args.defaults) + sum(d is not None for d in args.kw_defaults)
        all_params = [*params, *(a for a in (args.vararg, args.kwarg) if a)]
        f.update({
            "param_count": len(params),
            "param_annotation_frac": _div(sum(a.annotation is not None for a in all_params), len(all_params)),
            "param_default_frac": _div(n_defaults, len(params)),
            "has_return_annotation": float(getattr(fn, "returns", None) is not None),
            "has_varargs": float(args.vararg is not None),
            "has_kwargs": float(args.kwarg is not None),
        })
    else:
        f.update(dict.fromkeys(
            ["param_count", "param_annotation_frac", "param_default_frac",
             "has_return_annotation", "has_varargs", "has_kwargs"], 0.0))

    # Yerel isimler: imzadan (prompt'tan) gelmeyen, govdede tanimlanan isimler.
    local: set[str] = set()
    for n in nodes:
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            local.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            local.add(n.name)
        elif isinstance(n, (*_FUNC_TYPES, ast.ClassDef)) and n is not fn:
            local.add(n.name)
        elif isinstance(n, ast.arg) and n not in params and n.arg not in ("self", "cls"):
            if args is None or n not in (args.vararg, args.kwarg):
                local.add(n.arg)
    lens = [len(x) for x in local]
    nl = len(local)
    f.update({
        "local_name_count": nl,
        "local_name_per_stmt": nl / n_stmt,
        "local_name_mean_len": statistics.fmean(lens) if lens else 0.0,
        "local_name_max_len": max(lens, default=0),
        "local_name_single_char_frac": _div(sum(x == 1 for x in lens), nl),
        "local_name_snake_frac": _div(sum(bool(_SNAKE_RE.match(x)) for x in local), nl),
        "local_name_camel_frac": _div(sum(bool(_CAMEL_RE.match(x)) and "_" not in x.strip("_") for x in local), nl),
        "local_name_upper_frac": _div(sum(bool(_UPPER_RE.match(x)) and len(x) > 1 for x in local), nl),
        "local_name_digit_frac": _div(sum(any(c.isdigit() for c in x) for x in local), nl),
        "local_name_underscore_prefix_frac": _div(sum(x.startswith("_") for x in local), nl),
    })
    names = [n.id for n in nodes if isinstance(n, ast.Name)]
    attrs = [len(n.attr) for n in nodes if isinstance(n, ast.Attribute)]
    f["name_unique_frac"] = _div(len(set(names)), len(names))
    f["attr_name_mean_len"] = statistics.fmean(attrs) if attrs else 0.0

    # --- 4. Deyimsel tercihler (deyim basina oran) ---
    calls = [n for n in nodes if isinstance(n, ast.Call)]
    call_names = Counter(_call_name(c) for c in calls)
    with_calls = {
        id(item.context_expr)
        for n in nodes if isinstance(n, (ast.With, ast.AsyncWith))
        for item in n.items
    }
    idiom = Counter()
    for n in nodes:
        if isinstance(n, ast.Compare):
            comps = [n.left, *n.comparators]
            has_none = any(_is_none(c) for c in comps)
            for op in n.ops:
                if has_none and isinstance(op, (ast.Is, ast.IsNot)):
                    idiom["is_none"] += 1
                if has_none and isinstance(op, (ast.Eq, ast.NotEq)):
                    idiom["eq_none"] += 1
            if any(isinstance(c, ast.Constant) and isinstance(c.value, bool) for c in comps):
                idiom["compare_bool_literal"] += 1
            if any(isinstance(c, ast.Call) and _call_name(c) == "type" for c in comps):
                idiom["type_compare"] += 1
            if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                   and c.func.id == "len" for c in comps):
                idiom["len_compare"] += 1
        elif isinstance(n, ast.Call):
            name = _call_name(n)
            if isinstance(n.func, ast.Name) and name == "range" and n.args and \
                    isinstance(n.args[0], ast.Call) and _call_name(n.args[0]) == "len":
                idiom["range_len"] += 1
            if isinstance(n.func, ast.Name) and name == "open":
                idiom["with_open" if id(n) in with_calls else "open_without_with"] += 1
            if _is_log_call(n):
                idiom["logging"] += 1
            if isinstance(n.func, ast.Attribute) and name == "format":
                idiom["format_call"] += 1
            if any(isinstance(a, ast.Starred) for a in n.args) or \
                    any(k.arg is None for k in n.keywords):
                idiom["star_args_call"] += 1
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            for s in n.body:
                if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and \
                        isinstance(s.value.func, ast.Attribute) and s.value.func.attr == "append":
                    idiom["for_append"] += 1
                    break
        elif isinstance(n, ast.Raise):
            if n.cause is not None:
                idiom["raise_from"] += 1
        elif isinstance(n, ast.ExceptHandler):
            if n.type is None:
                idiom["bare_except"] += 1
            elif isinstance(n.type, ast.Name) and n.type.id in ("Exception", "BaseException"):
                idiom["broad_except"] += 1
        elif isinstance(n, ast.If):
            if n.orelse and _ends_with_jump(n.body):
                idiom["else_after_return"] += 1
        elif isinstance(n, ast.Return):
            if _is_none(n.value):
                idiom["return_none_explicit"] += 1
        elif isinstance(n, (ast.Try, ast.TryStar)):
            if n.finalbody:
                idiom["try_finally"] += 1
        elif isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name) and n.value.id == "self":
                idiom["self_attr"] += 1
        elif isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Mod) and isinstance(n.left, ast.Constant) and \
                    isinstance(n.left.value, str):
                idiom["percent_format"] += 1
        elif isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            idiom["comprehension"] += 1

    idiom["enumerate"] = call_names["enumerate"]
    idiom["zip"] = call_names["zip"]
    idiom["isinstance"] = call_names["isinstance"]
    idiom["print"] = call_names["print"]
    idiom["dict_get"] = sum(1 for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "get")
    idiom["dict_items"] = sum(1 for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "items")
    idiom["str_join"] = sum(1 for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "join")
    idiom["builtin_reduce"] = sum(call_names[b] for b in _REDUCE_BUILTINS)
    idiom["super_call"] = call_names["super"]
    for key in _IDIOM:
        f[f"idiom_{key}_rate"] = idiom[key] / n_stmt

    f["fmt_fstring_rate"] = counts[ast.JoinedStr] / n_stmt
    f["fmt_format_call_rate"] = idiom["format_call"] / n_stmt
    f["fmt_percent_format_rate"] = idiom["percent_format"] / n_stmt
    return f


def _indent_info(tokens: list[tokenize.TokenInfo]) -> tuple[int, int, float]:
    """(govde taban girintisi, girinti birimi, birim tutarliligi)."""
    stack = [0]
    deltas: list[int] = []
    base = 0
    for tok in tokens:
        if tok.type == tokenize.INDENT:
            width = len(tok.string)
            if len(stack) == 1:
                base = base or width
            else:
                deltas.append(width - stack[-1])
            stack.append(width)
        elif tok.type == tokenize.DEDENT and len(stack) > 1:
            stack.pop()
    if deltas:
        unit, hits = Counter(deltas).most_common(1)[0]
        return base, unit, hits / len(deltas)
    if base == 0:
        return 0, 0, 1.0
    # Tek seviyeli govdede 8 vb. taban girinti CSN metot artigi sayilir.
    unit = 4 if base > 4 and base % 4 == 0 else base
    return base, unit, 1.0


def _quote_char(tok: tokenize.TokenInfo) -> str | None:
    s = tok.string.lstrip("rRbBuUfF")
    return s[0] if s and s[0] in "'\"" else None


def _layout_features(code: str, tokens: list[tokenize.TokenInfo]) -> dict[str, float]:
    lines = code.splitlines()
    n_lines = max(1, len(lines))
    base, unit, consistency = _indent_info(tokens)
    extra = max(0, base - unit) if unit else 0

    # Satir uzunluklari: govdedeki fazla (artik) girinti cikarilarak.
    lengths = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        lead = len(line) - len(line.lstrip(" "))
        cut = min(extra, lead) if i > 0 else 0
        lengths.append(len(line.rstrip()) - cut)
    blank = sum(1 for line in lines if not line.strip())

    # Tokenize tabanli bosluk sayimlari.
    sig = [t for t in tokens if t.type not in (
        tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
        tokenize.COMMENT, tokenize.ENDMARKER)]
    c = Counter()
    depth = 0
    for tok in tokens:
        if tok.type == tokenize.OP:
            if tok.string in _OPEN:
                depth += 1
            elif tok.string in _CLOSE:
                depth = max(0, depth - 1)
        elif tok.type == tokenize.NL and depth > 0:
            c["bracket_break"] += 1
        elif tok.type in (tokenize.STRING, tokenize.FSTRING_START):
            c["string"] += 1
            q = _quote_char(tok)
            if q:
                c["single" if q == "'" else "double"] += 1

    sig_depth = _bracket_depths(sig)
    for i, tok in enumerate(sig):
        if tok.type != tokenize.OP:
            continue
        prev = sig[i - 1] if i > 0 else None
        nxt = sig[i + 1] if i + 1 < len(sig) else None
        if prev is None or nxt is None:
            continue
        same_line = prev.end[0] == tok.start[0] == nxt.start[0]
        left = prev.end != tok.start
        right = nxt.start != tok.end
        s = tok.string
        if s == ",":
            if nxt.string in _CLOSE:
                if nxt.start[0] != tok.start[0]:
                    c["trailing_comma"] += 1
                continue
            if nxt.start[0] != tok.start[0]:
                continue
            c["comma"] += 1
            c["comma_spaced"] += right
        elif s in _OPEN:
            if nxt.string in _CLOSE or nxt.start[0] != tok.start[0]:
                continue
            c["bracket"] += 1
            c["bracket_spaced"] += right
        elif s == "=" and same_line:
            # Parantez icinde "=" anahtar kelime argumani / varsayilan degerdir.
            key = "kwarg" if sig_depth[i] > 0 else "assign"
            c[key] += 1
            c[key + "_spaced"] += left and right
        elif s in _BIN_OPS and same_line:
            unary = prev.type not in _OPERAND_TYPES and prev.string not in _CLOSE
            if unary or (prev.type == tokenize.NAME and keyword.iskeyword(prev.string)
                         and prev.string not in ("None", "True", "False")):
                continue
            c["binop"] += 1
            c["binop_spaced"] += left and right

    backslash = sum(1 for line in lines if line.rstrip().endswith("\\"))
    return {
        "fmt_num_lines": len(lines),
        "fmt_line_len_mean": statistics.fmean(lengths) if lengths else 0.0,
        "fmt_line_len_max": max(lengths, default=0),
        "fmt_line_len_std": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
        "fmt_line_over79_frac": _div(sum(x > 79 for x in lengths), len(lengths)),
        "fmt_blank_line_frac": blank / n_lines,
        "fmt_trailing_ws_frac": sum(1 for line in lines if line != line.rstrip()) / n_lines,
        "fmt_indent_unit": unit,
        "fmt_indent_consistency": consistency,
        "fmt_binop_spaced_frac": _div(c["binop_spaced"], c["binop"], 0.5),
        "fmt_assign_spaced_frac": _div(c["assign_spaced"], c["assign"], 0.5),
        "fmt_kwarg_eq_spaced_frac": _div(c["kwarg_spaced"], c["kwarg"], 0.5),
        "fmt_comma_space_frac": _div(c["comma_spaced"], c["comma"], 0.5),
        "fmt_bracket_inner_space_frac": _div(c["bracket_spaced"], c["bracket"], 0.5),
        "fmt_trailing_comma_rate": c["trailing_comma"] / n_lines,
        "fmt_single_quote_frac": _div(c["single"], c["single"] + c["double"], 0.5),
        "fmt_string_per_line": c["string"] / n_lines,
        "fmt_backslash_continuation_rate": backslash / n_lines,
        "fmt_bracket_line_break_rate": c["bracket_break"] / n_lines,
        "fmt_tokens_per_line": len(sig) / n_lines,
    }


def _bracket_depths(sig: list[tokenize.TokenInfo]) -> list[int]:
    """Her token'in onundeki acik parantez derinligi."""
    out, depth = [], 0
    for tok in sig:
        out.append(depth)
        if tok.type == tokenize.OP:
            if tok.string in _OPEN:
                depth += 1
            elif tok.string in _CLOSE:
                depth = max(0, depth - 1)
    return out


def extract_features(code: str) -> dict[str, float]:
    """Tek bir fonksiyonun ozelliklerini FEATURE_NAMES sirasinda dondurur.

    Kod parse edilemezse ValueError firlatir.
    """
    try:
        tree = ast.parse(code)
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError) as e:
        raise ValueError(f"Kod parse edilemedi: {e}") from e
    feats = _ast_features(tree, len(code.splitlines()))
    feats.update(_layout_features(code, tokens))
    return {k: float(feats[k]) for k in FEATURE_NAMES}


def features_frame(codes: Iterable[str]) -> pd.DataFrame:
    """Kod listesinden ozellik tablosu (kolonlar FEATURE_NAMES sirasinda)."""
    rows = [extract_features(c) for c in codes]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
