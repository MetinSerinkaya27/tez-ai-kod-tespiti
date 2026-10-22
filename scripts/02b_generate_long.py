"""Uzun fonksiyonlar icin ek AI uretimi (deepseek-coder, ChatGPT tarzi istemler).

02 ile uretilen AI kodlari kisa (medyan 9 satir); egitimde 35+ satirlik fonksiyonlarin
yalnizca %15'i AI oldugu icin model "uzun ve duzenli kod = insan" ogreniyordu. Bu
betik 20-60 satirlik insan fonksiyonlari icin, kullanicilarin buyuk modellere
sordugu gibi "eksiksiz ve duzenli bir uygulama" isteyerek uzun AI kodu uretir.
Uretici yine deepseek oldugu icin qwen ve llama gorulmemis ureticiler olarak kalir.

Iki adim:
  uv run python scripts/02b_generate_long.py select            # istem dosyasini yazar
  uv run python scripts/02b_generate_long.py gen --shard 0/2    # isin yarisini uretir
Is parcalara bolunebilir (orn. yarisi bu bilgisayarda, yarisi Kaggle'da); her parca
ayri dosyaya yazar ve kaldigi yerden devam eder. 03_build_dataset.py tum
data/interim/ai_long*.jsonl dosyalarini okur.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

from tez.normalize import normalize_code
from tez.paths import AI_JSONL, HUMAN_JSONL, INTERIM, ROOT, SEED

LONG_PROMPTS = INTERIM / "long_prompts.jsonl"
MIN_HUMAN_LINES, MAX_HUMAN_LINES = 20, 60

LONG_PROMPT = (
    "Write a complete, well-structured Python function `{name}({params})`. {task}\n\n"
    "Details from the original documentation:\n{doc}\n\n"
    "Write a full, production-quality implementation and handle edge cases. "
    "Return only this one function in a single ```python code block, with no explanation."
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(rw, gen, rec: dict) -> tuple[str, str]:
    """Kayit id'sine gore iki istem tipinden birini secer (deterministik)."""
    if int(rec["id"], 16) % 2 == 0:
        natural = rw.natural_prompt(rec)  # "Write a Python function `f(a)` that ..."
        name_params, task = natural.split("` that ", 1)
        name, params = name_params.split("`", 1)[1].split("(", 1)
        return "long_natural", LONG_PROMPT.format(
            name=name, params=params.rstrip(")"), task="It " + task, doc=rec["docstring"])
    return "long_skeleton", gen.PROMPT.format(skeleton=gen.skeleton(rec)).replace(
        "Write complete, working code;",
        "Write a complete, well-structured, production-quality implementation that handles "
        "edge cases;")


def cmd_select(args) -> None:
    human = load_jsonl(HUMAN_JSONL)
    used = {r["id"] for r in load_jsonl(AI_JSONL) + load_jsonl(
        AI_JSONL.with_name(AI_JSONL.stem + "_failed.jsonl"))}
    pool = [r for r in human if r["id"] not in used
            and MIN_HUMAN_LINES <= len(r["code"].splitlines()) <= MAX_HUMAN_LINES]
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["id"]):
        by_repo[r["repo"]].append(r)
    rng = random.Random(SEED)
    chosen = []
    for repo in sorted(by_repo):
        recs = by_repo[repo]
        rng.shuffle(recs)
        chosen.extend(recs[: args.per_repo])
    rng.shuffle(chosen)
    chosen = chosen[: args.n]
    keys = ("id", "func_name", "signature", "docstring", "repo")
    with open(LONG_PROMPTS, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n")
    print(f"{len(pool)} uygun uzun insan fonksiyonu, {len(by_repo)} depo; "
          f"{len(chosen)} istem -> {LONG_PROMPTS}")


def cmd_gen(args) -> None:
    gen = _load_module("gen02", ROOT / "scripts" / "02_generate_ai.py")
    rw = _load_module("rw07", ROOT / "scripts" / "07_real_world.py")
    prompts = load_jsonl(args.prompts)
    if not prompts:
        raise SystemExit(f"{args.prompts} yok; once 'select' calistirilmali")
    i, n = (int(x) for x in args.shard.split("/"))
    recs = [r for k, r in enumerate(prompts) if k % n == i]
    if args.limit is not None:
        recs = recs[: args.limit]
    out_path: Path = args.out
    failed_path = out_path.with_name(out_path.stem + "_failed.jsonl")
    done = {r["id"] for r in load_jsonl(out_path) + load_jsonl(failed_path)}
    todo = [r for r in recs if r["id"] not in done]
    print(f"parca {i}/{n}: {len(recs)} istem, {len(recs) - len(todo)} zaten islenmis, "
          f"{len(todo)} uretilecek -> {out_path}")
    if not todo:
        return
    gen.check_ollama(args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    with open(out_path, "a", encoding="utf-8") as fo, open(failed_path, "a", encoding="utf-8") as ff:
        bar = tqdm(todo, desc="uzun")
        for rec in bar:
            seed = int(rec["id"], 16) % (2**31)
            temperature = gen.TEMPERATURES[seed % len(gen.TEMPERATURES)]
            options = {"temperature": temperature, "top_p": 0.95,
                       "num_predict": args.max_tokens, "seed": seed}
            prompt_type, prompt = build_prompt(rw, gen, rec)
            base = {"id": rec["id"], "func_name": rec["func_name"], "model": args.model,
                    "prompt_type": prompt_type, "repo": rec["repo"], "set": "long"}
            t0 = time.perf_counter()
            try:
                raw = gen.generate(prompt, args.model, options, args.timeout)
            except requests.Timeout:
                raw, code, reason = None, None, "timeout"
            except requests.RequestException as e:
                raw, code, reason = None, None, f"http_error: {e}"
            else:
                code, reason = gen.process(raw, rec["func_name"])
            elapsed = round(time.perf_counter() - t0, 2)
            if code is not None and normalize_code(code) is not None:
                fo.write(json.dumps({**base, "code": code, "raw": raw, "label": 1,
                                     "temperature": temperature, "seed": seed,
                                     "elapsed": elapsed}, ensure_ascii=False) + "\n")
                fo.flush()
                n_ok += 1
            else:
                ff.write(json.dumps({**base, "reason": reason or "normalize_failed", "raw": raw,
                                     "temperature": temperature, "seed": seed,
                                     "elapsed": elapsed}, ensure_ascii=False) + "\n")
                ff.flush()
                n_fail += 1
            bar.set_postfix(ok=n_ok, fail=n_fail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--per-repo", type=int, default=25)
    g = sub.add_parser("gen")
    g.add_argument("--prompts", type=Path, default=LONG_PROMPTS)
    g.add_argument("--out", type=Path, default=INTERIM / "ai_long.jsonl")
    g.add_argument("--shard", default="0/1", help="i/n: istemlerin k %% n == i olanlari")
    g.add_argument("--model", default="deepseek-coder:6.7b-instruct")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--max-tokens", type=int, default=2048)
    g.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()
    cmd_select(args) if args.cmd == "select" else cmd_gen(args)


if __name__ == "__main__":
    main()
