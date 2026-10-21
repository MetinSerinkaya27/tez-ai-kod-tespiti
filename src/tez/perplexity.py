"""Nedensel dil modeli tabanli "perplexity" ozellikleri.

Her kod (normalize edilmis hali) kucuk bir kod dil modelinden gecirilir; her
token icin log-olasilik, rank (dogru tokenin model siralamasindaki yeri) ve
tahmin dagiliminin entropisi hesaplanir, sonra kod basina ozetlenir.

Model secimi: varsayilan `Salesforce/codegen-350M-mono` (ucretsiz, token
gerektirmez, Python'a ozel egitilmis, ~350M parametre). BILEREK deepseek
ailesinden (deepseek-coder) bir model KULLANILMAZ: egitim verisindeki AI kodu
deepseek-coder ile uretildi; ayni aile kendi ciktisina cok yuksek olasilik
verir ve detektor "deepseek ciktisi mi" sorusunu ogrenir. Bu da gorulmemis
ureticilere (qwen, llama) genellemeyi bozar.

Ozellikler kasitli olarak uzunluk icermez (token sayisi yok); uzunluk zaten AST
ozelliklerinde var ve burada ayrica sizmasin. Ilk token (onceki baglam yok) ve
dolgu tokenlari ortalamalara girmez. Uzun kodlar `max_length` tokende kesilir.
Hesap deterministiktir: eval modu, no_grad, float32.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MODEL = "Salesforce/codegen-350M-mono"
DEFAULT_MAX_LENGTH = 512

PPL_FEATURES = [
    "ppl_mean_logprob",
    "ppl_std_logprob",
    "ppl_median_logprob",
    "ppl_p10_logprob",
    "ppl_p90_logprob",
    "ppl_mean_entropy",
    "ppl_mean_rank_log",
    "ppl_top1_frac",
    "ppl_top10_frac",
    "ppl_logprob_entropy_ratio",
]

# Ozellik tanimi degisirse artir; onbellek anahtarina girer. (Modul dosyasinin
# tamami hash'lenmez: pahali GPU hesabi kozmetik degisikliklerle bozulmasin.)
PPL_VERSION = 1


def is_ppl_feature(name: str) -> bool:
    return name.startswith("ppl_")


def resolve_device(device: str) -> str:
    """'auto' -> cuda varsa cuda, yoksa cpu."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda istendi ama torch.cuda.is_available() False")
    return device


def load_lm(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """(tokenizer, model) ciftini bir kez yukler (lazy + onbellekli)."""
    return _load_lm(model_name, resolve_device(device))


@lru_cache(maxsize=2)
def _load_lm(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float32")
    model.to(device)
    model.eval()
    return tok, model


def sequence_stats(logits, input_ids, attention_mask) -> list[dict[str, float]]:
    """Bir yigin icin kod basina ozellikler.

    logits (B, T, V), input_ids (B, T), attention_mask (B, T) torch tensorleri.
    t konumundaki token, t-1 konumundaki dagilimla tahmin edilir; t=0 (ilk token)
    tahmin edilmez. Yalnizca maske[t-1] ve maske[t] 1 olan konumlar sayilir, yani
    dolgu (sag ya da sol) dislanir.
    """
    import torch

    logits = logits.detach()[:, :-1, :].float()
    target = input_ids[:, 1:]
    mask = (attention_mask[:, :-1] * attention_mask[:, 1:]).bool()
    out = []
    for b in range(logits.shape[0]):
        lg = logits[b][mask[b]]                     # (n, V)
        tg = target[b][mask[b]]                     # (n,)
        if lg.shape[0] == 0:
            out.append({k: float("nan") for k in PPL_FEATURES})
            continue
        logp = torch.log_softmax(lg, dim=-1)
        lp = logp.gather(1, tg[:, None])[:, 0]
        ent = -(logp.exp() * logp).sum(-1)
        actual = lg.gather(1, tg[:, None])
        rank = (lg > actual).sum(-1) + 1            # 1 = en olasi token
        lp_np = lp.double().cpu().numpy()
        ent_np = ent.double().cpu().numpy()
        rank_np = rank.cpu().numpy()
        mean_lp, mean_ent = float(lp_np.mean()), float(ent_np.mean())
        out.append({
            "ppl_mean_logprob": mean_lp,
            "ppl_std_logprob": float(lp_np.std()),
            "ppl_median_logprob": float(np.median(lp_np)),
            "ppl_p10_logprob": float(np.percentile(lp_np, 10)),
            "ppl_p90_logprob": float(np.percentile(lp_np, 90)),
            "ppl_mean_entropy": mean_ent,
            "ppl_mean_rank_log": float(np.log1p(rank_np).mean()),
            "ppl_top1_frac": float((rank_np == 1).mean()),
            "ppl_top10_frac": float((rank_np <= 10).mean()),
            # Binoculars benzeri: gozlenen surpriz / beklenen surpriz.
            "ppl_logprob_entropy_ratio": -mean_lp / mean_ent if mean_ent > 0 else float("nan"),
        })
    return out


def token_stats(
    codes, model_name: str = DEFAULT_MODEL, device: str = "cpu", batch_size: int = 8,
    max_length: int = DEFAULT_MAX_LENGTH, tokenizer=None, model=None, progress: bool = False,
) -> pd.DataFrame:
    """Kod listesi -> PPL_FEATURES tablosu (girdi sirasinda).

    `tokenizer`/`model` verilmezse `load_lm` ile yuklenir (testlerde sahte LM
    verilebilir). Kesilen kod sayisi `df.attrs["n_truncated"]` icindedir.
    """
    import torch

    codes = list(codes)
    if tokenizer is None or model is None:
        tokenizer, model = load_lm(model_name, device)
    dev = next(model.parameters()).device if hasattr(model, "parameters") else device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    ids = [tokenizer(c, add_special_tokens=False)["input_ids"] for c in codes]
    n_trunc = sum(len(x) > max_length for x in ids)
    ids = [x[:max_length] for x in ids]
    # Benzer uzunluklari ayni yigina koy (dolgu az olsun); sonra sira geri kurulur.
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    rows: list[dict | None] = [None] * len(ids)
    batches = range(0, len(order), batch_size)
    if progress:
        from tqdm import tqdm
        batches = tqdm(batches, desc="perplexity", unit="yigin")
    with torch.no_grad():
        for start in batches:
            idx = order[start:start + batch_size]
            T = max(1, max(len(ids[i]) for i in idx))
            inp = torch.full((len(idx), T), pad_id, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            for j, i in enumerate(idx):
                inp[j, :len(ids[i])] = torch.tensor(ids[i], dtype=torch.long)
                att[j, :len(ids[i])] = 1
            inp, att = inp.to(dev), att.to(dev)
            logits = model(input_ids=inp, attention_mask=att).logits
            for j, r in zip(idx, sequence_stats(logits, inp, att)):
                rows[j] = r
    df = pd.DataFrame(rows, columns=PPL_FEATURES).astype(float)
    df.attrs["n_truncated"] = int(n_trunc)
    return df


# --- Onbellek ---

def ppl_hash(records: list[dict], model_name: str = DEFAULT_MODEL,
             max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """Model adi, kesme uzunlugu, ozellik surumu ve (id, kod) iceriginin hash'i."""
    h = hashlib.sha256()
    h.update(json.dumps({"model": model_name, "max_length": max_length,
                         "version": PPL_VERSION, "features": PPL_FEATURES}).encode())
    for r in records:
        h.update(r["id"].encode() + b"\0" + r["code"].encode() + b"\0")
    return h.hexdigest()[:16]


def ppl_cache_path(records: list[dict], cache_dir: str | Path, name: str,
                   model_name: str = DEFAULT_MODEL, max_length: int = DEFAULT_MAX_LENGTH) -> Path:
    return Path(cache_dir) / f"ppl_{name}_{ppl_hash(records, model_name, max_length)}.parquet"


def cached_ppl(
    records: list[dict], cache_dir: str | Path, name: str, model_name: str = DEFAULT_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH, compute: bool = True, **kwargs,
) -> pd.DataFrame:
    """Perplexity ozellik tablosu (index = id); hash uyusursa parquet'ten okunur.

    compute=False iken onbellek yoksa FileNotFoundError (04 bunu kullanir; pahali
    hesap yalnizca 03c'de yapilir). kwargs token_stats'a gider.
    """
    path = ppl_cache_path(records, cache_dir, name, model_name, max_length)
    ids = [r["id"] for r in records]
    if path.exists():
        X = pd.read_parquet(path)
        if list(X.index) == ids and list(X.columns) == PPL_FEATURES:
            X.attrs["n_truncated"] = None  # onbellekten okundu
            return X
    if not compute:
        raise FileNotFoundError(
            f"{path} bulunamadi (ya da veri/model degisti); once "
            f"scripts/03c_perplexity.py calistirilmali.")
    X = token_stats([r["code"] for r in records], model_name=model_name,
                    max_length=max_length, **kwargs)
    n_trunc = X.attrs.get("n_truncated")
    X.index = pd.Index(ids, name="id")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob(f"ppl_{name}_*.parquet"):
        old.unlink()
    X.to_parquet(path)
    X.attrs["n_truncated"] = n_trunc
    return X
