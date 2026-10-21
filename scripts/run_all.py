"""Uretimden degerlendirmeye kalan tum adimlari sirayla calistirir.

Tek komutla gece boyunca calismak icin: kesilirse ayni komutla kaldigi yerden
devam eder. Uretim adimlari zaten kaldigi yerden devam ediyor; sonraki adimlar
bittikce RUN_STATE dosyasina yazilir ve tekrar calistirilmaz (bastan calistirmak
icin dosyayi sil). Calisirken Windows'un uykuya gecmesi engellenir.

Kullanim: uv run python scripts/run_all.py
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime

from tez.paths import DATA, ROOT

RUN_STATE = DATA / "run_state.json"
LOG = DATA / "run_all.log"

PY = [sys.executable, "-u"]
OLLAMA_MODELS = ("deepseek-coder:6.7b-instruct", "qwen2.5-coder:7b-instruct", "llama3.2:3b")

# (ad, komut, her seferinde calissin mi). Uretim adimlari kendi ilerlemesini tuttugu
# icin her seferinde calistirilir; is kalmadiysa hemen biter.
STEPS = [
    ("02_uretim", ["scripts/02_generate_ai.py", "--n", "4300"], True),
    ("03_veri_seti", ["scripts/03_build_dataset.py"], False),
    ("03b_denetim", ["scripts/03b_audit_data.py"], False),
    ("04_xgb", ["scripts/04_train_xgb.py"], False),
    ("06_eval_xgb", ["scripts/06_evaluate.py", "--model", "xgb"], False),
    ("04b_ngram", ["scripts/04b_train_ngram.py"], False),
    ("06_eval_ngram", ["scripts/06_evaluate.py", "--model", "ngram"], False),
    ("04b_ngram_donemsiz", ["scripts/04b_train_ngram.py", "--era-neutral"], False),
    ("06_eval_ngram_donemsiz", ["scripts/06_evaluate.py", "--model", "ngram",
                                "--model-dir", "models/ngram_donemsiz", "--name", "ngram_donemsiz"],
     False),
    ("07_gercek_dunya_ai", ["scripts/07_real_world.py", "ai"], True),
    ("ollama_bosalt", None, True),
    # Dil modeli (codegen-350M) perplexity ozellikleri; GPU Ollama'dan bosaldiktan sonra.
    ("03c_perplexity", ["scripts/03c_perplexity.py", "--device", "cuda", "--batch-size", "8"], False),
    ("04_xgb_ppl", ["scripts/04_train_xgb.py", "--with-ppl"], False),
    ("06_eval_xgb_ppl", ["scripts/06_evaluate.py", "--model", "xgb", "--model-dir", "models/xgb_ppl",
                         "--name", "xgb_ppl", "--skip-cv", "--skip-cross-source"], False),
    ("08_gercek_dunya_xgb_ppl", ["scripts/08_eval_real_world.py", "--model", "xgb",
                                 "--model-dir", "models/xgb_ppl", "--name", "xgb_ppl"], False),
    ("05_codebert", ["scripts/05_train_codebert.py", "--device", "cuda", "--amp", "bf16",
                     "--max-length", "512", "--batch-size", "4", "--grad-accum", "8",
                     "--grad-checkpointing", "--epochs", "3", "--lime-n", "5",
                     "--onnx", "--quantize"], False),
    ("06_eval_codebert", ["scripts/06_evaluate.py", "--model", "codebert",
                          "--skip-cv", "--skip-cross-source"], False),
    # Daha yeni kod kodlayicisi; ayni ayarlarla egitilip val'de CodeBERT ile karsilastirilir.
    ("05_unixcoder", ["scripts/05_train_codebert.py", "--base-model", "microsoft/unixcoder-base",
                      "--name", "unixcoder", "--device", "cuda", "--amp", "bf16",
                      "--max-length", "512", "--batch-size", "4", "--grad-accum", "8",
                      "--grad-checkpointing", "--epochs", "3", "--lime-n", "5",
                      "--onnx", "--quantize"], False),
    ("06_eval_unixcoder", ["scripts/06_evaluate.py", "--model", "codebert",
                           "--model-dir", "models/unixcoder", "--name", "unixcoder",
                           "--skip-cv", "--skip-cross-source"], False),
    ("08_gercek_dunya_xgb", ["scripts/08_eval_real_world.py", "--model", "xgb"], False),
    ("08_gercek_dunya_xgb_donemsiz", ["scripts/08_eval_real_world.py", "--model", "xgb",
                                      "--model-dir", "models/xgb/ablation/donemsiz",
                                      "--name", "xgb_donemsiz"], False),
    ("08_gercek_dunya_codebert", ["scripts/08_eval_real_world.py", "--model", "codebert"], False),
    ("08_gercek_dunya_unixcoder", ["scripts/08_eval_real_world.py", "--model", "codebert",
                                   "--model-dir", "models/unixcoder", "--name", "unixcoder"], False),
    ("08_gercek_dunya_ngram", ["scripts/08_eval_real_world.py", "--model", "ngram"], False),
    ("08_gercek_dunya_ngram_donemsiz", ["scripts/08_eval_real_world.py", "--model", "ngram",
                                        "--model-dir", "models/ngram_donemsiz",
                                        "--name", "ngram_donemsiz"], False),
    # Ensemble'da XGBoost uyesi perplexity'li model.
    ("09_ensemble", ["scripts/09_ensemble.py", "--device", "cuda",
                     "--members", "xgb", "codebert", "ngram",
                     "--xgb-dir", "models/xgb_ppl"], False),
    ("06_eval_ensemble", ["scripts/06_evaluate.py", "--model", "ensemble",
                          "--skip-cv", "--skip-cross-source"], False),
    ("08_gercek_dunya_ensemble", ["scripts/08_eval_real_world.py", "--model", "ensemble"], False),
]


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake() -> None:
    """Islem suresince sistemin uykuya gecmesini engeller (ekran kapanabilir)."""
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)


def other_generator_running() -> bool:
    """Elle baslatilmis bir 02_generate_ai.py hala calisiyor mu (ayni dosyaya iki yazar olmasin)."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|uv)' "
         "-and $_.CommandLine -like '*02_generate_ai.py*' } | Measure-Object | "
         "Select-Object -ExpandProperty Count"],
        capture_output=True, text=True,
    ).stdout.strip()
    # Kontrol yalnizca run_all kendi uretim adimini baslatmadan once yapilir.
    return out.isdigit() and int(out) > 0


def unload_ollama() -> None:
    """CodeBERT egitimi icin GPU bellegini bosaltir."""
    for model in OLLAMA_MODELS:
        subprocess.run(["ollama", "stop", model], capture_output=True)
    time.sleep(5)


def main() -> None:
    keep_awake()
    state = json.loads(RUN_STATE.read_text(encoding="utf-8")) if RUN_STATE.exists() else {}

    while other_generator_running():
        log("Baska bir 02_generate_ai.py calisiyor; bitmesi bekleniyor (5 dk).")
        time.sleep(300)

    for name, args, always in STEPS:
        if not always and state.get(name):
            log(f"atlandi (daha once bitti): {name}")
            continue
        if args and not (ROOT / args[0]).exists():
            log(f"atlandi (betik yok): {name}")
            continue
        log(f"basladi: {name}")
        t0 = time.time()
        if args is None:
            unload_ollama()
        else:
            rc = subprocess.run(PY + args, cwd=ROOT).returncode
            if rc != 0:
                log(f"HATA: {name} cikis kodu {rc}. Duzeltip ayni komutla devam edin.")
                sys.exit(rc)
        state[name] = datetime.now().isoformat(timespec="seconds")
        RUN_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        log(f"bitti: {name} ({(time.time() - t0) / 60:.1f} dk)")

    log("Tum adimlar tamamlandi.")


if __name__ == "__main__":
    main()
