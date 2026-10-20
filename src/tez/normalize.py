"""Kaynak kod normalizasyonu.

Modelin yuzeysel ipuclarini (docstring, yorum satiri, markdown cercevesi) degil,
kodun yapisal ve uslupsal ozelliklerini ogrenmesi icin her iki siniftaki kodlara
ayni temizlik uygulanir. Girinti, bosluk ve degisken isimleri korunur; bunlar
stilometrik sinyalin parcasidir.
"""
from __future__ import annotations

import ast
import io
import re
import textwrap
import tokenize
import warnings

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Eski kodlardaki gecersiz kacis dizileri ("\i" gibi) icin uyarilari sustur.
warnings.filterwarnings("ignore", category=SyntaxWarning)

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_DOC_PARENTS = (ast.Module, ast.ClassDef, *_FUNC_TYPES)


def _docstring_node(node: ast.AST) -> ast.Expr | None:
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first
    return None


def strip_docstrings(code: str) -> str:
    """Modul, sinif ve fonksiyon docstring'lerini satir duzeyinde siler."""
    tree = ast.parse(code)
    lines = code.splitlines()
    drop: set[int] = set()
    replace_with_pass: dict[int, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, _DOC_PARENTS):
            continue
        doc = _docstring_node(node)
        if doc is None:
            continue
        start, end = doc.lineno - 1, doc.end_lineno - 1
        # "def f(): '''doc'''" gibi tek satirlik tanimlara dokunma.
        if not isinstance(node, ast.Module) and doc.lineno == node.lineno:
            continue
        if len(node.body) == 1 and not isinstance(node, ast.Module):
            indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
            replace_with_pass[start] = f"{indent}pass"
            drop.update(range(start + 1, end + 1))
        else:
            drop.update(range(start, end + 1))

    out = []
    for i, line in enumerate(lines):
        if i in replace_with_pass:
            out.append(replace_with_pass[i])
        elif i not in drop:
            out.append(line)
    return "\n".join(out) + "\n"


def strip_comments(code: str) -> str:
    """'#' yorumlarini siler; yalnizca yorum iceren satirlari tamamen kaldirir."""
    lines = code.splitlines()
    comment_cols: dict[int, int] = {}
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            comment_cols[row - 1] = col

    out = []
    for i, line in enumerate(lines):
        if i in comment_cols:
            kept = line[: comment_cols[i]].rstrip()
            if kept:
                out.append(kept)
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def fix_first_line_indent(code: str) -> str:
    """Yalnizca ilk satiri sola yaslanmis fonksiyonlarin govdesini yeniden girintiler.

    CodeSearchNet metotlari kaynaktan kesilirken yalnizca "def" satirinin girintisi
    atilmis; govde ve cok satirli imza eski girintiyi tasir. Bu artik birakilirsa
    model uslup yerine veri kaynagini ogrenir.
    """
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], _FUNC_TYPES):
        return code
    fn = tree.body[0]
    lines = code.splitlines()
    body_col = fn.body[0].col_offset
    # Girinti birimi: govdedeki en kucuk pozitif ic ice girinti farki, yoksa 4.
    cols = sorted({
        n.col_offset for n in ast.walk(fn)
        if isinstance(n, ast.stmt) and n is not fn and n.col_offset > body_col
    })
    unit = cols[0] - body_col if cols else 4
    extra = body_col - unit
    if extra <= 0:
        return code
    out = [lines[0]]
    for line in lines[1:]:
        lead = len(line) - len(line.lstrip(" "))
        out.append(line[min(extra, lead):])
    fixed = "\n".join(out) + "\n"
    try:
        ast.parse(fixed)
    except SyntaxError:
        return code
    return fixed


def normalize_code(code: str) -> str | None:
    """Docstring ve yorumlari temizler. Sonuc parse edilemiyorsa None doner."""
    try:
        code = textwrap.dedent(code.replace("\r\n", "\n").expandtabs(4))
        code = fix_first_line_indent(code)
        code = strip_comments(strip_docstrings(code))
        # Ardisik bos satirlari teke indir; bastaki/sondaki bos satirlari at.
        code = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", code).strip("\n") + "\n"
        ast.parse(code)
    except (SyntaxError, tokenize.TokenError, IndentationError, ValueError):
        return None
    return code


def extract_function(text: str, func_name: str | None = None) -> str | None:
    """LLM ciktisindan tek bir fonksiyon tanimini cikarir.

    Once ```python``` blogu aranir; bulunamazsa metnin tamami denenir. func_name
    verilirse o isimdeki fonksiyon, yoksa ilk ust duzey fonksiyon alinir.
    """
    candidates = _FENCE_RE.findall(text) or [text]
    for block in candidates:
        block = textwrap.dedent(block.replace("\r\n", "\n").expandtabs(4))
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        funcs = [n for n in tree.body if isinstance(n, _FUNC_TYPES)]
        # Sinif icinde metot olarak yazildiysa onu da kabul et.
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            funcs.extend(n for n in cls.body if isinstance(n, _FUNC_TYPES))
        if func_name:
            named = [f for f in funcs if f.name == func_name]
            funcs = named or funcs
        if not funcs:
            continue
        fn = funcs[0]
        # Dekoratorler atilir: CodeSearchNet kodlari dogrudan "def" ile basliyor.
        lines = block.splitlines()[fn.lineno - 1 : fn.end_lineno]
        return textwrap.dedent("\n".join(lines)) + "\n"
    return None


def function_header(code: str) -> tuple[str, str] | None:
    """Fonksiyonun imza satir(lar)ini ve docstring'ini dondurur."""
    code = textwrap.dedent(code.replace("\r\n", "\n").expandtabs(4))
    tree = ast.parse(code)
    funcs = [n for n in tree.body if isinstance(n, _FUNC_TYPES)]
    if len(funcs) != 1:
        return None
    fn = funcs[0]
    doc = _docstring_node(fn)
    if doc is None or doc.lineno == fn.lineno:
        return None
    lines = code.splitlines()
    signature = "\n".join(lines[fn.lineno - 1 : doc.lineno - 1])
    return signature, (ast.get_docstring(fn) or "").strip()
