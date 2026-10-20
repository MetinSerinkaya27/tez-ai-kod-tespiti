"""CodeBERT ince ayari, olasilik tahmini, LIME aciklamasi ve ONNX disa aktarimi.

Girdi metni normalize edilmis koddur (tez.normalize). Uzun kodlar token
duzeyinde "bas + son" stratejisiyle kesilir: imza ve ilk satirlar (bas) ile
donus/kapanis satirlari (son) birlikte korunur (Sun vd. 2019, BERT ile uzun
metin siniflandirma). Pozitif sinif AI (label=1).

Agir bagimliliklar (transformers, lime, onnx) fonksiyon icinde yuklenir;
boylece saf fonksiyonlarin testleri hizli kalir.
"""
from __future__ import annotations

import html
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

BASE_MODEL = "microsoft/codebert-base"
DEFAULT_MAX_LENGTH = 256
DEFAULT_HEAD_FRAC = 0.25
TRUNC_STRATEGIES = ("head_tail", "head")
CONFIG_NAME = "codebert_config.json"


# ---------------------------------------------------------------------------
# Tekrarlanabilirlik ve cihaz
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> torch.Generator:
    """random, numpy ve torch tohumlarini ayarlar; DataLoader icin uretec dondurur."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def select_device(name: str = "auto") -> torch.device:
    """'auto' ise CUDA varsa onu, yoksa CPU'yu secer."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


# ---------------------------------------------------------------------------
# Kesme ve kodlama
# ---------------------------------------------------------------------------

def truncate_ids(
    ids: Sequence[int],
    max_tokens: int,
    strategy: str = "head_tail",
    head_frac: float = DEFAULT_HEAD_FRAC,
) -> list[int]:
    """Token listesini en fazla max_tokens uzunluga indirir.

    "head": yalnizca bas korunur. "head_tail": basin round(max_tokens*head_frac)
    tokeni ile kalan kadar son token korunur (ortasi atilir).
    """
    if strategy not in TRUNC_STRATEGIES:
        raise ValueError(f"Bilinmeyen kesme stratejisi: {strategy}")
    if max_tokens <= 0:
        raise ValueError("max_tokens pozitif olmali")
    ids = list(ids)
    if len(ids) <= max_tokens:
        return ids
    if strategy == "head":
        return ids[:max_tokens]
    n_head = min(max_tokens, max(0, round(max_tokens * head_frac)))
    n_tail = max_tokens - n_head
    return ids[:n_head] + (ids[-n_tail:] if n_tail else [])


def encode_codes(
    tokenizer,
    codes: Sequence[str],
    max_length: int = DEFAULT_MAX_LENGTH,
    strategy: str = "head_tail",
    head_frac: float = DEFAULT_HEAD_FRAC,
) -> tuple[list[list[int]], int]:
    """Kodlari [CLS] + kesilmis tokenlar + [SEP] olarak kodlar.

    Ikinci deger kesilen ornek sayisidir. Dolgu (padding) yapilmaz; collate eder.
    """
    raw = tokenizer(list(codes), add_special_tokens=False, verbose=False)["input_ids"]
    budget = max_length - 2
    out, n_cut = [], 0
    for ids in raw:
        if len(ids) > budget:
            n_cut += 1
        out.append(
            [tokenizer.cls_token_id]
            + truncate_ids(ids, budget, strategy, head_frac)
            + [tokenizer.sep_token_id]
        )
    return out, n_cut


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int) -> dict[str, torch.Tensor]:
    """Degisken uzunluklu dizileri en uzuna gore sagdan doldurur."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return {"input_ids": ids, "attention_mask": mask}


class EncodedDataset(torch.utils.data.Dataset):
    """Onceden kodlanmis token dizileri ve etiketler."""

    def __init__(self, seqs: list[list[int]], labels: Sequence[int]):
        self.seqs, self.labels = seqs, list(labels)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int):
        return self.seqs[i], self.labels[i]


def make_collate(pad_id: int) -> Callable:
    def collate(batch):
        out = pad_batch([b[0] for b in batch], pad_id)
        out["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return out
    return collate


# ---------------------------------------------------------------------------
# Egitim ve degerlendirme
# ---------------------------------------------------------------------------

def autocast_ctx(device: torch.device, amp: str):
    """amp: 'none' | 'fp16' | 'bf16'. CPU'da her zaman kapali."""
    if device.type != "cuda" or amp == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def predict_loader(model, loader, device: torch.device, amp: str = "none") -> tuple[np.ndarray, float]:
    """Yukleyicideki tum orneklerin AI olasiligini ve ortalama kaybi dondurur."""
    model.eval()
    probs, losses, n = [], 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx(device, amp):
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        logits = logits.float()
        losses += torch.nn.functional.cross_entropy(logits, batch["labels"], reduction="sum").item()
        n += len(logits)
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs), losses / max(n, 1)


def selection_score(metrics: dict, primary: str = "roc_auc") -> tuple[float, float]:
    """Checkpoint secimi icin (birincil, ikincil) skor; NaN en kotu sayilir."""
    secondary = "f1" if primary == "roc_auc" else "roc_auc"
    def clean(x):
        return -math.inf if x is None or math.isnan(x) else float(x)
    return clean(metrics[primary]), clean(metrics[secondary])


def is_better(new: dict, best: dict | None, primary: str = "roc_auc") -> bool:
    """Yeni val metrikleri oncekinden kesin olarak iyiyse True (esitlikte eski kalir)."""
    if best is None:
        return True
    return selection_score(new, primary) > selection_score(best, primary)


# ---------------------------------------------------------------------------
# Kayit / yukleme ve tahmin
# ---------------------------------------------------------------------------

def save_bundle(model, tokenizer, model_dir: str | Path, config: dict) -> None:
    """Model, tokenizer ve tez yapilandirmasini (esik, kesme ayarlari) kaydeder."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(model_dir / CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_bundle_config(model_dir: str | Path) -> dict:
    path = Path(model_dir) / CONFIG_NAME
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class CodeBertDetector:
    """Kayitli (veya temel) CodeBERT modeliyle AI olasiligi tahmini ve LIME."""

    def __init__(self, model, tokenizer, config: dict | None = None, device: str | torch.device = "cpu"):
        self.config = config or {}
        self.device = select_device(device) if isinstance(device, str) else device
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_length = int(self.config.get("max_length", DEFAULT_MAX_LENGTH))
        self.trunc = self.config.get("trunc", "head_tail")
        self.head_frac = float(self.config.get("head_frac", DEFAULT_HEAD_FRAC))
        self.threshold = float(self.config.get("threshold", 0.5))

    @classmethod
    def from_dir(cls, model_dir: str | Path, device: str = "cpu", **model_kwargs) -> "CodeBertDetector":
        """Kayit dizininden (veya HF model adindan) yukler."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), num_labels=2, **model_kwargs
        )
        cfg = load_bundle_config(model_dir) if Path(model_dir).exists() else {}
        return cls(model, tok, cfg, device)

    @torch.no_grad()
    def predict_proba(self, codes: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Her kod icin AI (label=1) olasiligi; girdi sirasini korur."""
        if len(codes) == 0:
            return np.zeros(0, dtype=np.float32)
        seqs, _ = encode_codes(self.tokenizer, codes, self.max_length, self.trunc, self.head_frac)
        # Benzer uzunluklari ayni batch'e koy (dolgu azalir), sonra sirayi geri al.
        order = np.argsort([len(s) for s in seqs], kind="stable")
        out = np.empty(len(seqs), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = pad_batch([seqs[i] for i in idx], self.tokenizer.pad_token_id)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch).logits.float()
            out[idx] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return out

    def predict(self, codes: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Kayitli val esigiyle 0/1 tahmin."""
        return apply_threshold(self.predict_proba(codes, batch_size), self.threshold)

    def explain_one(self, code: str, num_features: int = 10, num_samples: int = 500,
                    seed: int = 42, batch_size: int = 32) -> dict:
        """LIME ile tek kodun en etkili birimlerini dondurur (bkz. explain_code)."""
        return explain_code(
            code, lambda xs: self.predict_proba(xs, batch_size),
            num_features=num_features, num_samples=num_samples, seed=seed,
            mask_token=self.tokenizer.unk_token,
        )


def apply_threshold(probs, threshold: float) -> np.ndarray:
    """Olasiliklari esikle 0/1'e cevirir (p >= esik -> AI); compute_metrics ile ayni kural."""
    return (np.asarray(probs, dtype=float) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# LIME: kod bolumleme ve aciklama
# ---------------------------------------------------------------------------

_STRING = r"""(?:[rRbBuUfF]{0,2})(?:'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*")"""
_OPS = r"\*\*=?|//=?|>>=?|<<=?|->|:=|\.\.\.|[-+*/%&|^@=<>!]="
_UNIT_RE = re.compile(
    rf"(?P<str>{_STRING})"
    r"|(?P<indent>(?m:^)[ \t]+)"
    r"|(?P<space>[ \t]+)"
    r"|(?P<newline>\r?\n)"
    r"|(?P<num>\d[\w.]*)"
    r"|(?P<name>\w+)"
    rf"|(?P<op>{_OPS})"
    r"|(?P<other>[\s\S])"
)
WHITESPACE_KINDS = ("indent", "space")


def segment_code(code: str) -> list[tuple[str, str]]:
    """Kodu (tur, metin) birimlerine boler; metinler birlestirilince kod aynen geri gelir.

    Turler: str (karakter dizisi), indent (satir basi girinti), space (satir ici
    bosluk), newline, num, name (tanimlayici/anahtar kelime), op, other.
    """
    return [(m.lastgroup, m.group()) for m in _UNIT_RE.finditer(code)]


def mask_unit(kind: str, text: str, mask_token: str = "<unk>") -> str:
    """Birimin maskelenmis hali: bosluk/girinti tek bosluga, diger birimler mask_token'a.

    Satir sonlari maskelenmez (kod yapisi korunur).
    """
    if kind == "newline":
        return text
    if kind in WHITESPACE_KINDS:
        return " "
    return mask_token


def feature_name(kind: str, text: str) -> str:
    """Rapor icin okunur ozellik adi."""
    if kind == "indent":
        return f"<girinti:{len(text.expandtabs(4))}>"
    if kind == "space":
        return f"<bosluk:{len(text)}>"
    return text if len(text) <= 40 else text[:37] + "..."


def build_features(units: list[tuple[str, str]], mask_token: str = "<unk>",
                   group: bool = True) -> tuple[list[str], list[list[int]]]:
    """Maskelenebilir LIME ozelliklerini ve her ozelligin kapsadigi birim indekslerini kurar.

    Maskelenince degismeyen birimler (satir sonu, tek bosluk) ozellik olmaz.
    group=True ise ayni ad/metne sahip tum gecisler tek ozelliktir (LIME bow=True).
    """
    names: list[str] = []
    members: list[list[int]] = []
    index: dict[tuple[str, str], int] = {}
    for i, (kind, text) in enumerate(units):
        if mask_unit(kind, text, mask_token) == text:
            continue
        name = feature_name(kind, text)
        key = (kind if kind in WHITESPACE_KINDS else "tok", name if kind in WHITESPACE_KINDS else text)
        if group and key in index:
            members[index[key]].append(i)
            continue
        index[key] = len(names)
        names.append(name)
        members.append([i])
    return names, members


def render_masked(units: list[tuple[str, str]], members: list[list[int]],
                  active: np.ndarray, mask_token: str = "<unk>") -> str:
    """active[j]==0 olan ozelliklerin birimlerini maskeleyerek kodu yeniden kurar."""
    masked = set()
    for j, on in enumerate(active):
        if not on:
            masked.update(members[j])
    return "".join(
        mask_unit(k, t, mask_token) if i in masked else t for i, (k, t) in enumerate(units)
    )


def explain_code(
    code: str,
    predict_fn: Callable[[list[str]], np.ndarray],
    num_features: int = 10,
    num_samples: int = 500,
    seed: int = 42,
    mask_token: str = "<unk>",
    group: bool = True,
    kernel_width: float = 25.0,
) -> dict:
    """LimeTextExplainer ile ayni ornekleme/cekirdek, kod bolumleyicisiyle.

    predict_fn kod listesi alip AI olasiliklarini (1-B dizi) dondurur. Donen
    sozluk: prob, features [(ad, agirlik)], unit_weights (her birim icin agirlik,
    HTML icin), score (yerel dogrusal modelin R^2'si), seconds.
    """
    from lime.lime_base import LimeBase
    from sklearn.metrics import pairwise_distances

    t0 = time.perf_counter()
    units = segment_code(code)
    names, members = build_features(units, mask_token, group)
    d = len(names)
    if d == 0:
        raise ValueError("Maskelenebilir birim yok")
    rng = np.random.RandomState(seed)
    data = np.ones((num_samples, d))
    texts = [code]
    sizes = rng.randint(1, d + 1, num_samples - 1)
    for i, size in enumerate(sizes, start=1):
        data[i, rng.choice(d, size, replace=False)] = 0
        texts.append(render_masked(units, members, data[i], mask_token))
    p_ai = np.asarray(predict_fn(texts), dtype=float).reshape(-1)
    labels = np.column_stack([1 - p_ai, p_ai])
    distances = pairwise_distances(data, data[:1], metric="cosine").ravel() * 100

    def kernel(dist, width=kernel_width):
        return np.sqrt(np.exp(-(dist ** 2) / width ** 2))

    base = LimeBase(kernel, random_state=np.random.RandomState(seed))
    intercept, exp, score, local_pred = base.explain_instance_with_data(
        data, labels, distances, label=1, num_features=min(num_features, d)
    )
    weights = dict(exp)
    unit_weights = [0.0] * len(units)
    for j, w in weights.items():
        for i in members[j]:
            unit_weights[i] = float(w)
    return {
        "prob": float(p_ai[0]),
        "features": [(names[j], float(w)) for j, w in exp],
        "unit_weights": unit_weights,
        "units": [t for _, t in units],
        "intercept": float(intercept),
        "score": float(score),
        "local_pred": float(np.ravel(local_pred)[0]),
        "n_features": d,
        "num_samples": num_samples,
        "seconds": time.perf_counter() - t0,
    }


def explanation_html(exp: dict, title: str = "") -> str:
    """Birimleri agirliga gore renklendirir: kirmizi AI yonune, mavi insan yonune iter."""
    ws = exp["unit_weights"]
    peak = max((abs(w) for w in ws), default=0) or 1.0
    spans = []
    for text, w in zip(exp["units"], ws):
        esc = html.escape(text)
        if w == 0:
            spans.append(esc)
            continue
        a = min(abs(w) / peak, 1.0) * 0.8
        color = f"rgba(220,40,40,{a:.2f})" if w > 0 else f"rgba(40,90,220,{a:.2f})"
        spans.append(f'<span style="background:{color}" title="{w:+.4f}">{esc}</span>')
    rows = "".join(
        f"<tr><td><code>{html.escape(n)}</code></td><td>{w:+.4f}</td></tr>" for n, w in exp["features"]
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>LIME</title></head><body>"
        f"<h3>{html.escape(title)}</h3><p>AI olasiligi: {exp['prob']:.4f} | yerel R2: {exp['score']:.3f}"
        f" | ozellik: {exp['n_features']} | ornek: {exp['num_samples']}</p>"
        f"<pre style='font-size:13px;line-height:1.5'>{''.join(spans)}</pre>"
        f"<table border='1' cellpadding='3'><tr><th>birim</th><th>agirlik (+AI)</th></tr>{rows}</table>"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------

class _LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(model_dir: str | Path, out_path: str | Path, opset: int = 17, seq_len: int = 32) -> Path:
    """Modeli ONNX'e aktarir (girdiler input_ids, attention_mask; cikti logits).

    Batch ve dizi uzunlugu dinamik eksendir. TorchScript tabanli disa aktarici
    (dynamo=False) ve eager dikkat kullanilir; SDPA izlemede sorun cikarabiliyor.
    """
    from transformers import AutoModelForSequenceClassification

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=2, attn_implementation="eager"
    ).eval()
    ids = torch.randint(5, 1000, (2, seq_len), dtype=torch.long)
    mask = torch.ones_like(ids)
    mask[1, seq_len // 2 :] = 0
    with torch.no_grad():
        torch.onnx.export(
            _LogitsOnly(model), (ids, mask), str(out_path),
            input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    return out_path


def quantize_onnx(in_path: str | Path, out_path: str | Path) -> Path:
    """Dinamik int8 nicemleme (agirliklar int8, aktivasyonlar calisma aninda)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(in_path), str(out_path), weight_type=QuantType.QInt8)
    return Path(out_path)


def onnx_logits(onnx_path: str | Path, batch: dict[str, torch.Tensor]) -> np.ndarray:
    """onnxruntime (CPU) ile logits hesaplar."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {k: batch[k].cpu().numpy().astype(np.int64) for k in ("input_ids", "attention_mask")}
    return sess.run(["logits"], feeds)[0]


def compare_onnx(detector: CodeBertDetector, onnx_path: str | Path, codes: Sequence[str]) -> dict:
    """PyTorch ve ONNX logits/olasiliklarini ayni dolgulu batch uzerinde karsilastirir."""
    seqs, _ = encode_codes(detector.tokenizer, codes, detector.max_length, detector.trunc, detector.head_frac)
    batch = pad_batch(seqs, detector.tokenizer.pad_token_id)
    with torch.no_grad():
        ref = detector.model(**{k: v.to(detector.device) for k, v in batch.items()}).logits.float().cpu().numpy()
    got = onnx_logits(onnx_path, batch)

    def softmax1(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e[:, 1] / e.sum(axis=1)

    p_ref, p_got = softmax1(ref), softmax1(got)
    thr = detector.threshold
    return {
        "n": len(codes),
        "max_abs_logit_diff": float(np.abs(ref - got).max()),
        "max_abs_prob_diff": float(np.abs(p_ref - p_got).max()),
        "label_agreement": float(np.mean((p_ref >= thr) == (p_got >= thr))),
        "size_mb": Path(onnx_path).stat().st_size / 2**20,
    }
