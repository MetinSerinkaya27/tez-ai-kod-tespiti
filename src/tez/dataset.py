"""Insan/AI kayitlarini birlestirme ve depo bazli egitim/dogrulama/test bolmesi.

Her insan fonksiyonu ile ondan uretilen AI fonksiyon(lar)i bir "cift" (pair_id)
olusturur. Bolme depo duzeyinde yapilir: bir depo yalnizca tek bir bolmede yer
alir; ciftin tum uyeleri insan kaydinin deposunu tasidigi icin ayni bolmeye duser.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from tez.paths import SEED

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def load_jsonl(path: str | Path) -> list[dict]:
    """JSONL dosyasini kayit listesi olarak okur (bos satirlar atlanir)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Kayitlari JSONL olarak yazar; ust dizini gerekirse olusturur."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def code_key(code: str) -> str:
    """Tekrar tespiti icin kod anahtari (satir sonu ve uc bosluklar yok sayilir)."""
    return code.replace("\r\n", "\n").strip()


def model_short(model: str) -> str:
    """Model adini id'de kullanilacak kisa bicime cevirir.

    "deepseek-coder:6.7b-instruct" -> "deepseek-coder-6-7b-instruct"
    """
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def pair_records(human: list[dict], ai: list[dict]) -> tuple[list[dict], Counter]:
    """Insan ve AI kayitlarini id uzerinden eslestirip birlesik kayit listesi dondurur.

    Yalnizca en az bir gecerli AI karsiligi olan insan kayitlari tutulur. Insan
    koduyla birebir ayni AI kodlari ve veri setinde daha once gorulen kodlar
    atilir. Ayni id icin farkli modellerden birden fazla AI kaydi olabilir; ayni
    (id, model) ikilisi tekrar ederse ilki alinir. Ikinci deger atilma nedenleri.
    """
    dropped: Counter = Counter()
    ai_by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in ai:
        models = ai_by_id[rec["id"]]
        if rec["model"] in models:
            dropped["ai_ayni_id_model"] += 1
            continue
        models[rec["model"]] = rec

    out: list[dict] = []
    seen_codes: set[str] = set()
    seen_ids: set[str] = set()
    for h in human:
        pid = h["id"]
        if pid in seen_ids:
            dropped["insan_tekrar_id"] += 1
            continue
        seen_ids.add(pid)
        if pid not in ai_by_id:
            dropped["insan_ai_yok"] += 1
            continue
        h_key = code_key(h["code"])
        if h_key in seen_codes:
            dropped["insan_tekrar_kod"] += 1
            continue

        ai_recs = []
        pair_codes = {h_key}
        for model, a in ai_by_id[pid].items():
            a_key = code_key(a["code"])
            if a_key == h_key:
                dropped["ai_insanla_ayni"] += 1
            elif a_key in seen_codes or a_key in pair_codes:
                dropped["ai_tekrar_kod"] += 1
            else:
                pair_codes.add(a_key)
                ai_recs.append({
                    "id": f"{pid}_{model_short(model)}",
                    "pair_id": pid,
                    "code": a["code"],
                    "label": 1,
                    "repo": h["repo"],
                    "source": model,
                })
        if not ai_recs:
            dropped["insan_gecerli_ai_yok"] += 1
            continue

        seen_codes |= pair_codes
        out.append({
            "id": f"{pid}_h",
            "pair_id": pid,
            "code": h["code"],
            "label": 0,
            "repo": h["repo"],
            "source": "human",
        })
        out.extend(ai_recs)

    dropped["ai_eslesmeyen"] = sum(
        len(m) for pid, m in ai_by_id.items() if pid not in seen_ids
    )
    return out, dropped


def drop_similar_pairs(
    records: list[dict], threshold: float, similarity: Callable[[str, str], float],
) -> tuple[list[dict], int]:
    """Insan cifti koduna fazla benzeyen AI kayitlarini atar (etiket gurultusu).

    LLM, egitim verisinde gordugu insan kodunu neredeyse aynen yazmis olabilir;
    boyle bir ornek "AI" etiketi tasisa da insan kodudur. AI karsiligi kalmayan
    insan kaydi da atilir ki ciftler dengeli kalsin. Ikinci deger atilan AI sayisi.
    """
    human = {r["pair_id"]: r["code"] for r in records if r["label"] == 0}
    bad = {
        r["id"] for r in records
        if r["label"] == 1 and similarity(human[r["pair_id"]], r["code"]) > threshold
    }
    kept = [r for r in records if r["id"] not in bad]
    with_ai = {r["pair_id"] for r in kept if r["label"] == 1}
    return [r for r in kept if r["pair_id"] in with_ai], len(bad)


def assign_repos(
    sizes: dict[str, int],
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    seed: int = SEED,
) -> dict[str, str]:
    """Depolari ornek sayilarina gore bolmelere atar (acgozlu, deterministik).

    Depolar once seed ile karistirilir, sonra buyukten kucuge siralanir (esit
    boyutlularda karisik sira korunur). Her depo, hedef ornek sayisina en uzak
    (eksigi en buyuk) bolmeye verilir; buyuk depolar once yerlesir, kucukler
    oranlari ince ayarlar.
    """
    if len(ratios) != len(SPLITS) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError(f"ratios {len(SPLITS)} elemanli olmali ve toplami 1 olmali: {ratios}")
    total = sum(sizes.values())
    targets = [r * total for r in ratios]
    counts = [0] * len(SPLITS)

    repos = sorted(sizes)
    random.Random(seed).shuffle(repos)
    repos.sort(key=lambda r: -sizes[r])

    assignment: dict[str, str] = {}
    for repo in repos:
        i = max(range(len(SPLITS)), key=lambda j: targets[j] - counts[j])
        assignment[repo] = SPLITS[i]
        counts[i] += sizes[repo]
    return assignment


def split_by_repo(
    records: list[dict],
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    seed: int = SEED,
) -> dict[str, list[dict]]:
    """Kayitlari depo bazli bolmelere ayirir; bolme ici sira girdi sirasidir."""
    sizes = Counter(r["repo"] for r in records)
    assignment = assign_repos(dict(sizes), ratios, seed)
    splits: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for rec in records:
        splits[assignment[rec["repo"]]].append(rec)
    return splits


def check_splits(splits: dict[str, list[dict]], balance_tol: float = 0.05) -> None:
    """Bolme butunlugunu dogrular; ihlalde ValueError firlatir.

    Kontroller: id'ler benzersiz; bolmeler arasi depo, cift ve kod kesisimi yok;
    hicbir bolme bos degil; her bolmede AI orani genel orandan en fazla
    balance_tol kadar sapar.
    """
    ids = [r["id"] for recs in splits.values() for r in recs]
    if len(ids) != len(set(ids)):
        raise ValueError("Tekrar eden id var")

    names = list(splits)
    for field, key in (("repo", None), ("pair_id", None), ("code", code_key)):
        sets = {
            s: {key(r[field]) if key else r[field] for r in splits[s]} for s in names
        }
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = sets[a] & sets[b]
                if common:
                    raise ValueError(
                        f"{a} ve {b} bolmeleri {len(common)} ortak '{field}' iceriyor"
                    )

    for s, recs in splits.items():
        if not recs:
            raise ValueError(f"{s} bolmesi bos")
        labels = Counter(r["label"] for r in recs)
        if not labels[0] or not labels[1]:
            raise ValueError(f"{s} bolmesinde tek sinif var: {dict(labels)}")

    all_recs = [r for recs in splits.values() for r in recs]
    overall = sum(r["label"] for r in all_recs) / len(all_recs)
    for s, recs in splits.items():
        ratio = sum(r["label"] for r in recs) / len(recs)
        if abs(ratio - overall) > balance_tol:
            raise ValueError(
                f"{s} bolmesinde sinif dengesi bozuk: AI orani {ratio:.3f}, "
                f"genel {overall:.3f}"
            )


def split_stats(splits: dict[str, list[dict]]) -> list[dict]:
    """Bolme basina ornek, depo, cift ve sinif sayilarini dondurur."""
    total = sum(len(recs) for recs in splits.values()) or 1
    rows = []
    for s, recs in splits.items():
        labels = Counter(r["label"] for r in recs)
        rows.append({
            "bolme": s,
            "ornek": len(recs),
            "oran": len(recs) / total,
            "depo": len({r["repo"] for r in recs}),
            "cift": len({r["pair_id"] for r in recs}),
            "insan": labels[0],
            "ai": labels[1],
            "kaynak": dict(Counter(r["source"] for r in recs)),
        })
    return rows
