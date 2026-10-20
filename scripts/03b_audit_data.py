"""Veri denetimi: ezber/etiket gurultusu, yakin kopyalar, kisa yol sinyalleri,
donem karistiricisi ve uretim kalitesi raporu.

Veri hattini degistirmez; yalnizca okur ve outputs/audit altina (veya --out-dir)
audit.md + audit.json yazar. ai.jsonl uretim sirasinda buyudugu icin once
anlik kopyasi alinir (--snapshot-dir) ve denetim kopya uzerinde yapilir.

Kullanim:
    uv run python scripts/03b_audit_data.py
    uv run python scripts/03b_audit_data.py --out-dir <dizin> --snapshot-dir <dizin>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from tez.audit import (
    code_tokens,
    duplicate_clusters,
    length_matched_sample,
    load_jsonl_tolerant,
    modern_markers,
    near_duplicate_pairs,
    pair_similarity,
    percentiles,
    quality_flags,
    repo_groups,
    shingles,
    shortcut_features,
    signature_diff,
    single_feature_auc,
)
from tez.dataset import assign_repos
from tez.paths import AI_JSONL, HUMAN_JSONL, OUTPUTS, ROOT, SEED

SIM_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
DUP_THRESHOLDS = (0.7, 0.8, 0.9, 1.0)
SHORTCUT_AUC = 0.8


def _load_sampler():
    """02_generate_ai.py'deki sample() fonksiyonunu (uretimde secilen kume) yukler."""
    path = ROOT / "scripts" / "02_generate_ai.py"
    spec = importlib.util.spec_from_file_location("gen_ai", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sample


def _r(x, nd=3):
    return round(float(x), nd)


# ---------------------------------------------------------------------------
# 1. Ezber / etiket gurultusu
# ---------------------------------------------------------------------------

def audit_memorization(pairs: list[tuple[dict, dict]], human: list[dict]) -> dict:
    sims = []
    for h, a in pairs:
        s = pair_similarity(h["code"], a["code"])
        s.update(id=a["id"], repo=h["repo"], temperature=a.get("temperature"))
        sims.append(s)

    # Bos model (null) dagilimi: AI kodu, eslesmedigi rastgele bir insan koduyla.
    rng = random.Random(SEED)
    others = sorted(human, key=lambda r: r["id"])
    null = []
    for h, a in pairs:
        o = rng.choice(others)
        while o["id"] == h["id"]:
            o = rng.choice(others)
        null.append(pair_similarity(o["code"], a["code"]))

    metrics = ["seq_ratio", "token_jaccard", "shingle5_jaccard",
               "seq_ratio_body", "token_jaccard_body", "shingle5_jaccard_body"]
    dist = {m: percentiles([s[m] for s in sims], (5, 25, 50, 75, 95, 99)) for m in metrics}
    null_dist = {m: percentiles([s[m] for s in null], (50, 95, 99)) for m in metrics}
    above = {
        m: {str(t): sum(s[m] > t for s in sims) for t in SIM_THRESHOLDS}
        for m in ("seq_ratio_body", "shingle5_jaccard_body", "seq_ratio")
    }
    by_temp = defaultdict(list)
    for s in sims:
        by_temp[str(s["temperature"])].append(s["seq_ratio_body"])
    top = sorted(sims, key=lambda s: -s["seq_ratio_body"])[:5]
    by_id = {a["id"]: (h, a) for h, a in pairs}
    top5 = [{
        "id": s["id"], "repo": s["repo"], "temperature": s["temperature"],
        "seq_ratio_body": _r(s["seq_ratio_body"]),
        "shingle5_jaccard_body": _r(s["shingle5_jaccard_body"]),
        "human_code": by_id[s["id"]][0]["code"], "ai_code": by_id[s["id"]][1]["code"],
    } for s in top]
    return {
        "n": len(sims),
        "dagilim": {m: {k: _r(v) for k, v in d.items()} for m, d in dist.items()},
        "null_dagilim_rastgele_insan": {m: {k: _r(v) for k, v in d.items()}
                                        for m, d in null_dist.items()},
        "esik_ustu_sayisi": above,
        "sicaklik_basina_seq_ratio_body": {
            t: {"n": len(v), **{k: _r(x) for k, x in percentiles(v, (50, 95)).items()}}
            for t, v in sorted(by_temp.items())},
        "en_benzer_5": top5,
        "kayit_basina": [{k: (_r(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
                         for s in sims],
    }


def audit_ai_vs_all_human(ai: list[dict], human: list[dict], threshold: float = 0.5) -> dict:
    """AI kodlarini TUM insan kodlariyla karsilastirir (baska fonksiyonu ezberleme)."""
    t0 = time.perf_counter()
    h_sets = [shingles(code_tokens(r["code"], body_only=True)) for r in human]
    a_sets = [shingles(code_tokens(r["code"], body_only=True)) for r in ai]
    pairs = near_duplicate_pairs(h_sets, threshold, queries=a_sets)
    hits = []
    for qi, hj, jac in pairs:
        if human[hj]["id"] == ai[qi]["id"]:
            continue
        hits.append({"ai_id": ai[qi]["id"], "human_id": human[hj]["id"],
                     "ayni_depo": ai[qi]["repo"] == human[hj]["repo"], "jaccard": _r(jac)})
    return {"esik": threshold, "eslesmeyen_insanla_benzer": len(hits),
            "ornekler": sorted(hits, key=lambda x: -x["jaccard"])[:10],
            "sure_sn": _r(time.perf_counter() - t0, 2)}


# ---------------------------------------------------------------------------
# 2. Yakin kopyalar
# ---------------------------------------------------------------------------

def _dup_report(records: list[dict], abstract: bool, min_t: float, split_sim: bool) -> dict:
    t0 = time.perf_counter()
    sets = [shingles(code_tokens(r["code"], abstract=abstract)) for r in records]
    t_tok = time.perf_counter() - t0
    pairs = near_duplicate_pairs(sets, min_t)
    t_join = time.perf_counter() - t0 - t_tok
    repos = [r["repo"] for r in records]
    counts = {}
    for t in DUP_THRESHOLDS:
        if t < min_t:
            continue
        sel = [p for p in pairs if p[2] >= t - 1e-12]
        cross = [p for p in sel if repos[p[0]] != repos[p[1]]]
        counts[str(t)] = {
            "cift": len(sel), "depo_ici_cift": len(sel) - len(cross),
            "depolar_arasi_cift": len(cross),
            "etkilenen_kayit": len({i for p in sel for i in p[:2]}),
            "fazla_kayit(kume_basi_1_tutulursa_atilan)":
                len(records) - len(set(duplicate_clusters(len(records), sel))),
        }
    out = {"n": len(records), "min_esik": min_t, "counts": counts,
           "sure_sn": {"token+shingle": _r(t_tok, 2), "join": _r(t_join, 2)}}

    if split_sim:
        sel = [p for p in pairs if p[2] >= 0.8 - 1e-12]
        cross = [p for p in sel if repos[p[0]] != repos[p[1]]]
        # Depo bazli bolme (insan depo boyutlariyla benzetim) altinda bolmeler
        # arasi sizinti: farkli bolmeye dusen depolar arasi yakin kopyalar.
        assign = assign_repos(dict(Counter(repos)))
        leak = [p for p in cross if assign[repos[p[0]]] != assign[repos[p[1]]]]
        groups = repo_groups(repos, sel)
        repo_pairs = Counter(tuple(sorted((repos[i], repos[j]))) for i, j, _ in cross)
        by_repo = defaultdict(set)
        for i, j, _ in sel:
            if repos[i] == repos[j]:
                by_repo[repos[i]].update((i, j))
        sizes = Counter(repos)
        out["esik_0.8"] = {
            "bolmeler_arasi_sizinti_cift(benzetim)": len(leak),
            "birlesen_depo_grubu": sum(1 for r, g in groups.items() if r != g),
            "en_cok_depolar_arasi_kopya": [
                {"depolar": list(k), "cift": v} for k, v in repo_pairs.most_common(8)],
            "buyuk_depolarda_depo_ici_tekrar_orani": [
                {"repo": r, "n": n, "kopyasi_olan": len(by_repo[r]),
                 "oran": _r(len(by_repo[r]) / n)}
                for r, n in sizes.most_common(10)],
            "ornek_depolar_arasi": [
                {"a": records[i]["id"], "a_repo": repos[i], "b": records[j]["id"],
                 "b_repo": repos[j], "jaccard": _r(jac)}
                for i, j, jac in sorted(cross, key=lambda p: -p[2])[:5]],
        }
    return out


def audit_near_duplicates(human: list[dict], chosen: list[dict], ai: list[dict]) -> dict:
    return {
        "insan_tum_ham_token": _dup_report(human, abstract=False, min_t=0.7, split_sim=True),
        "insan_tum_soyut_token(tip2)": _dup_report(human, abstract=True, min_t=0.8,
                                                    split_sim=False),
        "insan_02_secilen_ham": _dup_report(chosen, abstract=False, min_t=0.7, split_sim=False),
        "insan_02_secilen_soyut": _dup_report(chosen, abstract=True, min_t=0.8,
                                              split_sim=False),
        "ai_sinif_ici_ham": _dup_report(ai, abstract=False, min_t=0.7, split_sim=False),
    }


# ---------------------------------------------------------------------------
# 3. Kisa yol sinyalleri
# ---------------------------------------------------------------------------

def _pos_rate(hf: list[dict], af: list[dict], name: str) -> str | None:
    nh = sum(f[name] > 0 for f in hf)
    na = sum(f[name] > 0 for f in af)
    return f"{na / (nh + na):.2f} ({na}/{nh + na})" if nh + na else None


def audit_shortcuts(h_recs: list[dict], a_recs: list[dict], all_human: list[dict]) -> dict:
    hf = [shortcut_features(r["code"]) for r in h_recs]
    af = [shortcut_features(r["code"]) for r in a_recs]
    allf = [shortcut_features(r["code"]) for r in all_human]
    labels = [0] * len(hf) + [1] * len(af)
    rows = []
    for name in hf[0] if hf else []:
        vals = [f[name] for f in hf] + [f[name] for f in af]
        auc = single_feature_auc(vals, labels)
        auc_all = single_feature_auc([f[name] for f in allf] + [f[name] for f in af],
                                     [0] * len(allf) + [1] * len(af))
        rows.append({
            "ozellik": name, "auc": _r(auc["auc"]), "ayirt": _r(auc["separability"]),
            "ayirt_tum_insan": _r(auc_all["separability"]),
            "insan_ort": _r(sum(f[name] for f in hf) / len(hf)),
            "ai_ort": _r(sum(f[name] for f in af) / len(af)),
            "insan_sifirdan_buyuk": _r(sum(f[name] > 0 for f in hf) / len(hf)),
            "ai_sifirdan_buyuk": _r(sum(f[name] > 0 for f in af) / len(af)),
            # Ozellik varsa (>0) kaydin AI olma orani: nadir ama kesin isaretler
            # (yuksek kesinlik, dusuk kapsama) AUC'de gorunmez.
            "varsa_ai_orani": _pos_rate(hf, af, name),
            "kisa_yol_riski": auc["separability"] > SHORTCUT_AUC,
        })
    rows.sort(key=lambda r: -r["ayirt"])

    recs = [{"id": r["id"] + "_h", "label": 0, "code": r["code"]} for r in h_recs] + \
           [{"id": r["id"] + "_ai", "label": 1, "code": r["code"]} for r in a_recs]
    matched = length_matched_sample(recs, bin_width=2, seed=SEED)
    m_lab = [r["label"] for r in matched]
    m_len = [len(r["code"].splitlines()) for r in matched]
    return {
        "n_insan": len(hf), "n_ai": len(af),
        "ozellik_auc": rows,
        "satir_sayisi_yuzdelik": {
            "insan_eslesen": {k: _r(v, 1) for k, v in percentiles([f["n_lines"] for f in hf]).items()},
            "insan_tum": {k: _r(v, 1) for k, v in percentiles([f["n_lines"] for f in allf]).items()},
            "ai": {k: _r(v, 1) for k, v in percentiles([f["n_lines"] for f in af]).items()},
        },
        "uzunluk_eslestirme": {
            "kova_genisligi_satir": 2,
            "orneklem": len(matched), "insan": m_lab.count(0), "ai": m_lab.count(1),
            "satir_auc_once": _r(single_feature_auc(
                [f["n_lines"] for f in hf] + [f["n_lines"] for f in af], labels)["auc"]),
            "satir_auc_sonra": _r(single_feature_auc(m_len, m_lab)["auc"]) if matched else None,
        },
    }


# ---------------------------------------------------------------------------
# 4. Donem isaretleri, 5. uretim kalitesi
# ---------------------------------------------------------------------------

def _rates(codes: list[str], fn) -> dict:
    tot = Counter()
    for c in codes:
        tot.update(fn(c))
    return {k: _r(v / len(codes)) for k, v in sorted(tot.items())} if codes else {}


def audit_era(h_recs, a_recs, all_human) -> dict:
    return {
        "insan_tum": _rates([r["code"] for r in all_human], modern_markers),
        "insan_eslesen": _rates([r["code"] for r in h_recs], modern_markers),
        "ai": _rates([r["code"] for r in a_recs], modern_markers),
    }


def audit_quality(pairs, ai: list[dict], failed: list[dict]) -> dict:
    sig = Counter(signature_diff(h["code"], a["code"]) for h, a in pairs)
    per_temp: dict[str, Counter] = defaultdict(Counter)
    for r in ai:
        per_temp[str(r.get("temperature"))]["basarili"] += 1
    for r in failed:
        reason = (r.get("reason") or "?").split(":")[0]
        per_temp[str(r.get("temperature"))][f"elenen:{reason}"] += 1
    temps = {}
    for t, c in sorted(per_temp.items()):
        n = sum(c.values())
        temps[t] = {"toplam": n, "basari_orani": _r(c["basarili"] / n), **dict(c)}
    class_wrapped = sum("class " in (r.get("raw") or "") for r in ai)
    return {
        "imza_karsilastirma": dict(sig),
        "imza_farkli_orani": _r(1 - sig["ayni"] / max(1, sum(sig.values()))),
        "kalite_bayraklari_ai": _rates([a["code"] for _, a in pairs], quality_flags),
        "kalite_bayraklari_insan_eslesen": _rates([h["code"] for h, _ in pairs], quality_flags),
        "sicaklik_basina": temps,
        "elenme_nedenleri": dict(Counter((r.get("reason") or "?").split(":")[0]
                                         for r in failed)),
        "ham_ciktida_class_sarmali": class_wrapped,
        "sure_sn_ort": _r(sum(r.get("elapsed", 0) for r in ai) / max(1, len(ai)), 2),
    }


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------

def _table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def to_markdown(rep: dict) -> str:
    L = ["# Veri denetimi raporu", "",
         f"insan: {rep['meta']['n_human']}, AI: {rep['meta']['n_ai']}, "
         f"eslesen cift: {rep['meta']['n_pairs']}, elenen uretim: {rep['meta']['n_failed']}",
         f"Uyari: AI orneklemi {rep['meta']['n_ai']} kayit; kucukse sonuclar on bulgudur.", ""]

    m = rep["1_ezber"]
    L += ["## 1. Ezber / etiket gurultusu (AI vs eslesen insan)", "",
          "Benzerlik yuzdelikleri (govde = imza haric):", ""]
    L.append(_table([{"metrik": k, **v} for k, v in m["dagilim"].items()],
                    ["metrik", "p5", "p25", "p50", "p75", "p95", "p99", "mean"]))
    L += ["", "Null (rastgele eslesmeyen insan) dagilimi:", ""]
    L.append(_table([{"metrik": k, **v} for k, v in m["null_dagilim_rastgele_insan"].items()],
                    ["metrik", "p50", "p95", "p99", "mean"]))
    L += ["", "Esik ustu kayit sayisi:", ""]
    L.append(_table([{"metrik": k, **v} for k, v in m["esik_ustu_sayisi"].items()],
                    ["metrik"] + [str(t) for t in SIM_THRESHOLDS]))
    L += ["", "Sicaklik basina seq_ratio_body: "
          + json.dumps(m["sicaklik_basina_seq_ratio_body"]), "", "En benzer 5:", ""]
    for t in m["en_benzer_5"]:
        L += [f"- {t['id']} ({t['repo']}, T={t['temperature']}) seq_body={t['seq_ratio_body']}"
              f" shingle_body={t['shingle5_jaccard_body']}",
              "```python", "# insan", t["human_code"].rstrip(), "# AI", t["ai_code"].rstrip(), "```"]
    x = rep["1b_ai_tum_insan"]
    L += ["", f"AI vs TUM insan (govde shingle Jaccard >= {x['esik']}, kendi cifti haric): "
          f"{x['eslesmeyen_insanla_benzer']} eslesme", ""]

    L += ["## 2. Yakin kopyalar (token 5-gram shingle Jaccard, prefix filtreli tam join)", ""]
    for name, d in rep["2_yakin_kopya"].items():
        L += [f"### {name} (n={d['n']}, sure={d['sure_sn']})", ""]
        L.append(_table([{"esik": k, **v} for k, v in d["counts"].items()],
                        ["esik", "cift", "depo_ici_cift", "depolar_arasi_cift", "etkilenen_kayit",
                         "fazla_kayit(kume_basi_1_tutulursa_atilan)"]))
        if "esik_0.8" in d:
            e = d["esik_0.8"]
            L += ["", f"Esik 0.8: bolmeler arasi sizinti (benzetim): "
                  f"{e['bolmeler_arasi_sizinti_cift(benzetim)']} cift; "
                  f"birlesen depo: {e['birlesen_depo_grubu']}", ""]
            L.append(_table(e["buyuk_depolarda_depo_ici_tekrar_orani"],
                            ["repo", "n", "kopyasi_olan", "oran"]))
            L += ["", "Depolar arasi en cok kopya: "
                  + json.dumps(e["en_cok_depolar_arasi_kopya"], ensure_ascii=False)]
        L.append("")

    s = rep["3_kisa_yol"]
    L += ["## 3. Kisa yol sinyalleri (tek ozellik ROC-AUC; AI pozitif)", "",
          f"n_insan={s['n_insan']}, n_ai={s['n_ai']}; ayirt = max(AUC, 1-AUC); "
          f"> {SHORTCUT_AUC} kisa yol riski", ""]
    L.append(_table(s["ozellik_auc"], ["ozellik", "auc", "ayirt", "ayirt_tum_insan", "insan_ort",
                                       "ai_ort", "insan_sifirdan_buyuk", "ai_sifirdan_buyuk",
                                       "varsa_ai_orani", "kisa_yol_riski"]))
    L += ["", "Satir sayisi yuzdelikleri:", ""]
    L.append(_table([{"grup": k, **v} for k, v in s["satir_sayisi_yuzdelik"].items()],
                    ["grup", "p5", "p25", "p50", "p75", "p95", "mean"]))
    L += ["", "Uzunluk eslestirme: " + json.dumps(s["uzunluk_eslestirme"]), ""]

    e = rep["4_donem"]
    L += ["## 4. Donem isaretleri (kayitlarin yuzde kaci iceriyor)", ""]
    keys = sorted({k for v in e.values() for k in v})
    L.append(_table([{"isaret": k, **{g: e[g].get(k, 0) for g in e}} for k in keys],
                    ["isaret", *e.keys()]))
    q = rep["5_kalite"]
    L += ["", "## 5. Uretim kalitesi", "",
          f"Imza karsilastirma: {json.dumps(q['imza_karsilastirma'])} "
          f"(farkli orani {q['imza_farkli_orani']})", ""]
    keys = sorted(set(q["kalite_bayraklari_ai"]) | set(q["kalite_bayraklari_insan_eslesen"]))
    L.append(_table([{"bayrak": k, "ai": q["kalite_bayraklari_ai"].get(k, 0),
                      "insan": q["kalite_bayraklari_insan_eslesen"].get(k, 0)} for k in keys],
                    ["bayrak", "ai", "insan"]))
    L += ["", "Sicaklik basina: ", "```", json.dumps(q["sicaklik_basina"], indent=1), "```",
          f"Elenme nedenleri: {json.dumps(q['elenme_nedenleri'])}; ham ciktida class sarmali: "
          f"{q['ham_ciktida_class_sarmali']}; ort. uretim suresi {q['sure_sn_ort']} sn", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human", type=Path, default=HUMAN_JSONL)
    ap.add_argument("--ai", type=Path, default=AI_JSONL)
    ap.add_argument("--out-dir", type=Path, default=OUTPUTS / "audit")
    ap.add_argument("--snapshot-dir", type=Path, default=None,
                    help="ai.jsonl ve ai_failed.jsonl anlik kopyasinin yazilacagi dizin "
                         "(varsayilan: <out-dir>/snapshot)")
    ap.add_argument("--n", type=int, default=3000, help="02 ile ayni secim boyutu")
    ap.add_argument("--per-repo", type=int, default=15, help="02 ile ayni depo ust siniri")
    args = ap.parse_args()

    snap = args.snapshot_dir or args.out_dir / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    failed_src = args.ai.with_name(args.ai.stem + "_failed.jsonl")
    ai_path, failed_path = snap / args.ai.name, snap / failed_src.name
    for src, dst in ((args.ai, ai_path), (failed_src, failed_path)):
        if src.exists():
            shutil.copyfile(src, dst)  # yalnizca okuma; kaynak dosyaya yazilmaz

    t0 = time.perf_counter()
    human = load_jsonl_tolerant(args.human)
    ai = load_jsonl_tolerant(ai_path)
    failed = load_jsonl_tolerant(failed_path)
    h_by_id = {r["id"]: r for r in human}
    pairs = [(h_by_id[a["id"]], a) for a in ai if a["id"] in h_by_id]
    h_recs = [h for h, _ in pairs]
    a_recs = [a for _, a in pairs]
    chosen = _load_sampler()(human, args.n, args.per_repo)
    print(f"insan={len(human)} ai={len(ai)} cift={len(pairs)} elenen={len(failed)} "
          f"02_secilen={len(chosen)}")

    rep = {"meta": {"n_human": len(human), "n_ai": len(ai), "n_pairs": len(pairs),
                    "n_failed": len(failed), "n_02_secilen": len(chosen),
                    "ai_snapshot": str(ai_path)}}
    steps = [
        ("1_ezber", lambda: audit_memorization(pairs, human)),
        ("1b_ai_tum_insan", lambda: audit_ai_vs_all_human(a_recs, human)),
        ("2_yakin_kopya", lambda: audit_near_duplicates(human, chosen, a_recs)),
        ("3_kisa_yol", lambda: audit_shortcuts(h_recs, a_recs, human)),
        ("4_donem", lambda: audit_era(h_recs, a_recs, human)),
        ("5_kalite", lambda: audit_quality(pairs, ai, failed)),
    ]
    for key, fn in steps:
        t = time.perf_counter()
        rep[key] = fn()
        print(f"{key}: {time.perf_counter() - t:.1f} sn")
    rep["meta"]["toplam_sure_sn"] = _r(time.perf_counter() - t0, 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "audit.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out_dir / "audit.md").write_text(to_markdown(rep), encoding="utf-8")
    print(f"Yazildi -> {args.out_dir / 'audit.md'}, {args.out_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
