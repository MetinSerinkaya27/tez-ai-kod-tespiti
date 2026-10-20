"""Insan orneklerinin imza + docstring'inden yerel LLM ile AI sinifini uretir.

Her istem yalnizca fonksiyon iskeletini (imza ve docstring) icerir; insan
kodunun govdesi modele asla gosterilmez. Ciktilar satir satir eklenir, boylece
betik yarida kesilse de kaldigi yerden devam edebilir.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

from tez.normalize import extract_function, normalize_code
from tez.paths import AI_JSONL, HUMAN_JSONL, SEED

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-coder:6.7b-instruct"
TEMPERATURES = (0.2, 0.5, 0.8, 1.0)

# 01_build_human.py ile ayni satir sinirlari.
MIN_LINES, MAX_LINES = 5, 60

PROMPT = (
    "Implement the body of the following Python function. Write complete, "
    "working code; do not leave placeholders such as `pass` or TODO comments.\n"
    "Return only this one function (signature, docstring and body) in a single "
    "```python code block, with no explanation.\n\n"
    "```python\n{skeleton}\n```"
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample(records: list[dict], n: int, per_repo: int) -> list[dict]:
    """Depo basina en fazla per_repo kayit alir, sonra karistirip n tane secer."""
    rng = random.Random(SEED)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_repo[rec["repo"]].append(rec)

    pool = []
    for repo in sorted(by_repo):
        recs = sorted(by_repo[repo], key=lambda r: r["id"])
        rng.shuffle(recs)
        pool.extend(recs[:per_repo])
    rng.shuffle(pool)
    return pool[:n]


def skeleton(rec: dict) -> str:
    """Imza satir(lar)i + girintili docstring'den olusan iskelet."""
    doc = rec["docstring"].replace('"""', '\\"\\"\\"')
    doc = "\n".join(("    " + ln) if ln.strip() else "" for ln in doc.splitlines())
    return f'{rec["signature"]}\n    """\n{doc}\n    """'


def check_ollama(model: str) -> None:
    """Ollama sunucusu ve model yoksa anlasilir mesajla cikar."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(
            f"Ollama'ya ulasilamadi ({OLLAMA_URL}): {e}\n"
            "Ollama'nin calistigindan emin olun (ornegin 'ollama serve')."
        )
    names = {m["name"] for m in r.json().get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        raise SystemExit(
            f"Model bulunamadi: {model}\n"
            f"Kurulu modeller: {', '.join(sorted(names)) or '(yok)'}\n"
            f"Indirmek icin: ollama pull {model}"
        )


def generate(prompt: str, model: str, options: dict, timeout: float,
             retries: int = 2) -> str:
    """Tek istem gonderir; gecici sunucu hatalarinda (5xx) yeniden dener."""
    for attempt in range(retries + 1):
        try:
            return _chat(prompt, model, options, timeout)
        except requests.HTTPError:
            if attempt == retries:
                raise
            time.sleep(5)


def _chat(prompt: str, model: str, options: dict, timeout: float) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
            "keep_alive": "30m",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


_STUB_RE = re.compile(r"placeholder|your (code|implementation)|implement (this|me)|\bTODO\b", re.IGNORECASE)


def _is_empty_stmt(node: ast.stmt) -> bool:
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return True
    if isinstance(node, ast.Raise):
        exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"
    if isinstance(node, ast.Return):
        v = node.value
        return v is None or (isinstance(v, ast.Constant) and v.value is None) or (
            isinstance(v, (ast.List, ast.Dict, ast.Tuple, ast.Set)) and not getattr(v, "elts", getattr(v, "keys", []))
        )
    return False


def is_stub(code: str) -> bool:
    """Govdesi yalnizca pass / ... / NotImplementedError / bos return olan taslak mi."""
    fn = ast.parse(code).body[0]
    return all(_is_empty_stmt(s) for s in fn.body)


def process(raw: str, func_name: str) -> tuple[str | None, str | None]:
    """Ham ciktidan normalize kodu cikarir. (kod, None) ya da (None, sebep) doner."""
    fn = extract_function(raw, func_name)
    if fn is None:
        return None, "no_function"
    name = next(
        (n.name for n in ast.parse(fn).body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if name != func_name:
        return None, "wrong_name"
    if _STUB_RE.search(fn):
        return None, "stub"
    norm = normalize_code(fn)
    if norm is None:
        return None, "normalize_failed"
    if is_stub(norm):
        return None, "stub"
    n_lines = len(norm.splitlines())
    if n_lines < MIN_LINES:
        return None, "too_short"
    if n_lines > MAX_LINES:
        return None, "too_long"
    return norm, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000, help="secilecek ornek sayisi")
    ap.add_argument("--per-repo", type=int, default=15, help="depo basina ust sinir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="deneme icin ilk K istem")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sabit sicaklik; verilmezse ornek basina TEMPERATURES'tan secilir")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=300, help="istek basina saniye")
    ap.add_argument("--out", type=Path, default=AI_JSONL, help="cikti jsonl yolu")
    args = ap.parse_args()

    out_path: Path = args.out
    failed_path = out_path.with_name(out_path.stem + "_failed.jsonl")

    human = load_jsonl(HUMAN_JSONL)
    if not human:
        raise SystemExit(f"Insan verisi yok: {HUMAN_JSONL} (once 01_build_human.py)")
    chosen = sample(human, args.n, args.per_repo)
    if args.limit is not None:
        chosen = chosen[: args.limit]

    done = {r["id"] for r in load_jsonl(out_path)} | {
        r["id"] for r in load_jsonl(failed_path)
    }
    todo = [r for r in chosen if r["id"] not in done]
    print(f"{len(chosen)} secildi, {len(chosen) - len(todo)} zaten islenmis, "
          f"{len(todo)} uretilecek -> {out_path}")
    if not todo:
        return

    check_ollama(args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    with open(out_path, "a", encoding="utf-8") as fo, \
            open(failed_path, "a", encoding="utf-8") as ff:
        bar = tqdm(todo, desc="ai")
        for rec in bar:
            # Ornek basina sabit tohum: ayni id ayni ciktiyi verir.
            seed = int(rec["id"], 16) % (2**31)
            # Tek bir ornekleme ayarinin uslubu ezberlenmesin diye sicaklik cesitlenir.
            temperature = (args.temperature if args.temperature is not None
                           else TEMPERATURES[seed % len(TEMPERATURES)])
            options = {
                "temperature": temperature,
                "top_p": args.top_p,
                "num_predict": args.max_tokens,
                "seed": seed,
            }
            base = {"id": rec["id"], "func_name": rec["func_name"],
                    "model": args.model, "repo": rec["repo"]}
            t0 = time.perf_counter()
            try:
                raw = generate(PROMPT.format(skeleton=skeleton(rec)),
                               args.model, options, args.timeout)
            except requests.Timeout:
                raw, code, reason = None, None, "timeout"
            except requests.RequestException as e:
                raw, code, reason = None, None, f"http_error: {e}"
            else:
                code, reason = process(raw, rec["func_name"])
            elapsed = round(time.perf_counter() - t0, 2)

            if code is not None:
                fo.write(json.dumps({**base, "code": code, "raw": raw, "label": 1,
                                     **options, "elapsed": elapsed},
                                    ensure_ascii=False) + "\n")
                fo.flush()
                n_ok += 1
            else:
                ff.write(json.dumps({**base, "reason": reason, "raw": raw,
                                     **options, "elapsed": elapsed},
                                    ensure_ascii=False) + "\n")
                ff.flush()
                n_fail += 1
            bar.set_postfix(ok=n_ok, fail=n_fail)

    print(f"Basarili: {n_ok}, basarisiz: {n_fail} (-> {failed_path})")


if __name__ == "__main__":
    main()
