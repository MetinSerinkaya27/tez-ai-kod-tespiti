"""Yapay zeka kaynakli Python kodu tespiti icin Gradio demo arayuzu.

Bu betik XGBoost (AST stilometrisi + SHAP) ve CodeBERT (+ LIME)
modellerini kullanarak tek bir Python fonksiyonunun yapay zeka tarafindan
yazilip yazilmadigini analiz eder.
"""
from __future__ import annotations

import ast
import html
from functools import lru_cache
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tez.paths import MODELS, ROOT
from tez.normalize import normalize_code


# ---------------------------------------------------------------------------
# Karar ve Esik Islemleri
# ---------------------------------------------------------------------------

def format_decision(prob: float, threshold: float, margin: float = 0.05) -> str:
    """Model olasiligi ve sabit esige gore sonuc metni uretir."""
    prob_pct = int(round(prob * 100))
    thresh_pct = int(round(threshold * 100))
    if abs(prob - threshold) <= margin:
        return f"Emin değil (AI Olasılığı: %{prob_pct}) [Karar Eşiği: %{thresh_pct}]"
    elif prob >= threshold:
        return f"AI yazmış olabilir (%{prob_pct}) [Karar Eşiği: %{thresh_pct}]"
    else:
        return f"İnsan yazmış olabilir (%{100 - prob_pct}) [AI Olasılığı: %{prob_pct}, Karar Eşiği: %{thresh_pct}]"


# ---------------------------------------------------------------------------
# Python Sozdizimi Denetimi
# ---------------------------------------------------------------------------

def validate_code(code: str) -> tuple[bool, str]:
    """Kodun gecerli Python sozdizimine sahip olup olmadigini denetler."""
    if not code or not code.strip():
        return False, "Lütfen analiz edilecek bir Python fonksiyonu giriniz."
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        err = f"Sözdizimi Hatası (SyntaxError):\nSatır {e.lineno}, Sütun {e.offset}: {e.msg}"
        if e.text:
            err += f"\n  -> {e.text.strip()}"
        return False, err
    except Exception as e:
        return False, f"Ayrıştırma Hatası: {str(e)}"

    # Fonksiyon tanimi iceriyor mu kontrol et
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return False, "Kod geçerli bir Python sözdizimine sahip, ancak fonksiyon tanımı ('def ...') bulunamadı. Lütfen bir Python fonksiyonu giriniz."

    norm = normalize_code(code)
    if norm is None:
        return False, "Kod normalizasyonu başarısız oldu (yorum ve docstring temizlendikten sonra kod parse edilemedi)."

    return True, ""


# ---------------------------------------------------------------------------
# Model Yukleyicileri (Onbellekli)
# ---------------------------------------------------------------------------

def is_xgb_available() -> bool:
    """XGBoost model dizininin ve gerekli dosyalarin varligini denetler."""
    xgb_dir = MODELS / "xgb"
    return (xgb_dir / "meta.json").exists() and (xgb_dir / "model.json").exists()


# Test ve gercek dunya setinde en iyi tek model UniXcoder; yoksa CodeBERT kullanilir.
TRANSFORMER_DIRS = ("unixcoder", "codebert")
# Genel karar: XGBoost+perplexity, UniXcoder ve n-gram'in val'de ogrenilmis birlesimi.
ENSEMBLE_DIR = "ensemble_unix"


def transformer_dir() -> Path | None:
    for name in TRANSFORMER_DIRS:
        d = MODELS / name
        if (d / "config.json").exists() and ((d / "model.safetensors").exists()
                                             or (d / "pytorch_model.bin").exists()):
            return d
    return None


def is_codebert_available() -> bool:
    """UniXcoder/CodeBERT model dizininin ve agirliklarinin varligini denetler."""
    return transformer_dir() is not None


def is_ensemble_available() -> bool:
    return (MODELS / ENSEMBLE_DIR / "ensemble.json").exists()


@lru_cache(maxsize=1)
def get_xgb_detector():
    """XGBDetector nesnesini yukler ve onbellege alir."""
    from tez.xgb_model import XGBDetector
    return XGBDetector.load(MODELS / "xgb")


@lru_cache(maxsize=1)
def get_codebert_detector():
    """UniXcoder (yoksa CodeBERT) dedektorunu CPU uzerinde yukler ve onbellege alir."""
    from tez.codebert import CodeBertDetector
    return CodeBertDetector.from_dir(transformer_dir(), device="cpu")


@lru_cache(maxsize=1)
def get_ensemble_detector():
    """Topluluk modelini CPU uzerinde yukler (ilk cagrida dil modeli de yuklenir)."""
    from tez.ensemble import EnsembleDetector
    return EnsembleDetector.load(MODELS / ENSEMBLE_DIR, device="cpu")


def run_ensemble_analysis(code: str) -> str:
    """Genel karar: uc modelin birlesimi (olasiliklar val'de kalibre)."""
    if not is_ensemble_available():
        return "⚠️ Topluluk modeli henüz eğitilmedi"
    try:
        det = get_ensemble_detector()
        prob = float(det.predict_proba([code])[0])
        return format_decision(prob, det.threshold)
    except Exception as e:
        return f"Topluluk analizi sırasında hata oluştu: {e}"


# ---------------------------------------------------------------------------
# XGBoost Analiz ve SHAP Grafigi
# ---------------------------------------------------------------------------

def plot_shap_bar(shap_items: list[tuple[str, float, float]]) -> matplotlib.figure.Figure:
    """En etkili SHAP ozelliklerini yatay cubuk grafik olarak cizer."""
    # Ters cevir: en yuksek onem en ustte gorunsun
    items = list(reversed(shap_items))
    names = [f"{name} ({val:.2g})" if isinstance(val, (int, float)) else f"{name} ({val})" for name, val, _ in items]
    shaps = [shap_val for _, _, shap_val in items]
    colors = ["#ef4444" if s > 0 else "#22c55e" for s in shaps]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=100)
    y_pos = np.arange(len(items))
    bars = ax.barh(y_pos, shaps, color=colors, height=0.6, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_xlabel("SHAP Katkısı (Pozitif: AI Yönü, Negatif: İnsan Yönü)", fontsize=9)
    ax.set_title("En Etkili 10 Stilometri Özelliği (SHAP)", fontsize=10, fontweight="bold", pad=10)

    # Cubuk uzerine sayisal degerleri yazdir
    for bar, val in zip(bars, shaps):
        x = bar.get_width()
        ha = "left" if x >= 0 else "right"
        offset = 0.01 if x >= 0 else -0.01
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                va="center", ha=ha, fontsize=8, color="#374151")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def run_xgb_analysis(code: str) -> tuple[str, str, matplotlib.figure.Figure | None]:
    """XGBoost ve SHAP analizi gerceklestirir."""
    if not is_xgb_available():
        return (
            "⚠️ Model henüz eğitilmedi",
            "models/xgb dizininde eğitilmiş model (model.json ve meta.json) bulunamadı.",
            None,
        )

    try:
        from tez.xgb_model import explain_one
        det = get_xgb_detector()
        df = det.code_frame(code)
        prob = float(det.predict_proba(df)[0])
        decision = format_decision(prob, det.threshold)
        status_info = f"AI Olasılığı: %{prob * 100:.2f} | Model Eşiği: %{det.threshold * 100:.2f}"

        # En etkili 10 ozellik
        shap_items = explain_one(code, top=10)
        fig = plot_shap_bar(shap_items)
        return decision, status_info, fig
    except Exception as e:
        return "Hata", f"XGBoost analizi sırasında hata oluştu: {str(e)}", None


# ---------------------------------------------------------------------------
# CodeBERT Analiz ve LIME HTML
# ---------------------------------------------------------------------------

def render_lime_html(exp: dict) -> str:
    """LIME sonuclarini renklendirilmis HTML olarak uretir."""
    ws = exp["unit_weights"]
    peak = max((abs(w) for w in ws), default=0) or 1.0
    spans = []

    for text, w in zip(exp["units"], ws):
        esc = html.escape(text)
        if w == 0:
            spans.append(esc)
            continue
        alpha = min(abs(w) / peak, 1.0) * 0.75 + 0.20
        if w > 0:
            # AI yonu kirmizi
            bg_color = f"rgba(239, 68, 68, {alpha:.2f})"
        else:
            # Insan yonu yesil
            bg_color = f"rgba(34, 197, 94, {alpha:.2f})"
        spans.append(
            f'<span style="background-color: {bg_color}; border-radius: 2px; padding: 1px 2px; margin: 0 0.5px;" '
            f'title="Ağırlık: {w:+.4f} ({"AI" if w > 0 else "İnsan"})">{esc}</span>'
        )

    legend_html = (
        '<div style="margin-bottom: 8px; font-size: 12px; color: #64748b; display: flex; gap: 16px;">'
        '  <span><span style="display:inline-block; width:12px; height:12px; background:rgba(239, 68, 68, 0.75); border-radius:2px; vertical-align:middle; margin-right:4px;"></span>Kırmızı: AI yönü</span>'
        '  <span><span style="display:inline-block; width:12px; height:12px; background:rgba(34, 197, 94, 0.75); border-radius:2px; vertical-align:middle; margin-right:4px;"></span>Yeşil: İnsan yönü</span>'
        '</div>'
    )

    code_html = "".join(spans)
    return (
        f'{legend_html}'
        f'<div style="background-color: #0f172a; color: #e2e8f0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; '
        f'font-size: 13px; line-height: 1.6; padding: 14px; border-radius: 8px; max-height: 380px; overflow-y: auto; white-space: pre-wrap; word-break: break-all;">'
        f'{code_html}'
        f'</div>'
    )


def run_codebert_analysis(code: str) -> tuple[str, str, str]:
    """CodeBERT ve LIME analizi gerceklestirir."""
    if not is_codebert_available():
        return (
            "⚠️ Model henüz eğitilmedi",
            "models/codebert dizininde eğitilmiş model bulunamadı.",
            "<div style='color: #64748b; padding: 12px; background: #f8fafc; border-radius: 6px;'>Model henüz eğitilmedi.</div>",
        )

    try:
        det = get_codebert_detector()
        prob = float(det.predict_proba([code])[0])
        decision = format_decision(prob, det.threshold)
        status_info = f"AI Olasılığı: %{prob * 100:.2f} | Model Eşiği: %{det.threshold * 100:.2f}"

        # LIME aciklamasi (300 ornek)
        exp = det.explain_one(code, num_features=10, num_samples=300)
        html_code = render_lime_html(exp)
        return decision, status_info, html_code
    except Exception as e:
        return (
            "Hata",
            f"CodeBERT analizi sırasında hata oluştu: {str(e)}",
            f"<div style='color: #ef4444;'>Hata: {html.escape(str(e))}</div>",
        )


# ---------------------------------------------------------------------------
# Ana Analiz Yonetici Fonksiyonu
# ---------------------------------------------------------------------------

def analyze_code(code: str):
    """Kullanici girdisini denetler ve her iki modelle analiz eder."""
    valid, err_msg = validate_code(code)
    if not valid:
        # Kod gecerli degilse hata goster
        error_box = f"❌ {err_msg}"
        return (
            error_box,
            "", "", None,
            "", "", "<div style='color: #ef4444; padding: 12px;'>Geçerli bir Python fonksiyonu girilmedi.</div>",
            "",
        )

    error_box = ("✅ Python fonksiyonu başarıyla doğrulandı. Modeller eğitimdeki gibi "
                 "docstring ve yorumları kaldırılmış kodu değerlendirir.")
    # Egitim verisi normalize edilmis koddan olusuyor; her iki model de ayni girdiyi gormeli.
    norm = normalize_code(code)
    xgb_dec, xgb_info, xgb_fig = run_xgb_analysis(norm)
    cb_dec, cb_info, cb_html = run_codebert_analysis(norm)
    ens_dec = run_ensemble_analysis(norm)

    return (
        error_box,
        xgb_dec, xgb_info, xgb_fig,
        cb_dec, cb_info, cb_html,
        ens_dec,
    )


# ---------------------------------------------------------------------------
# Hazir Kod Ornekleri
# ---------------------------------------------------------------------------

# Ornekler TEST kumesinden (yandex/yandex-tank deposu egitim ve dogrulamada yok).
# Ayni gorev icin gercek bir cift (pair_id b6a25170fd12); topluluk modeli ikisini de dogru biliyor.
EXAMPLE_HUMAN = '''def pid_exists(pid):
    if pid < 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        logging.debug("No process[%s]: %s", exc.errno, exc)
        return exc.errno == errno.EPERM
    else:
        p = psutil.Process(pid)
        return p.status != psutil.STATUS_ZOMBIE
'''

# AI: deepseek-coder 6.7B, ayni imza ve docstring ile uretildi.
EXAMPLE_AI = '''def pid_exists(pid):
    if pid < 0:
        raise ValueError("Negative PIDs are not valid")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    else:
        return True
'''

# Zor ornek (pair_id e5ed920f3855): INSAN yazimi ama modeller AI saniyor. Kisa ve sade
# fonksiyonlarda insan ve AI kodu neredeyse ayni; sinirliligi gostermek icin bilerek eklendi.
EXAMPLE_HARD_HUMAN = '''def count_matched_codes(codes_regex, codes_dict):
    total = 0
    for code, count in codes_dict.items():
        if codes_regex.match(str(code)):
            total += count
    return total
'''

EXAMPLE_INVALID = '''def broken_function(x, y
    return x + y
'''


# ---------------------------------------------------------------------------
# Gradio Arayuz Yapisi
# ---------------------------------------------------------------------------

def create_demo() -> gr.Blocks:
    """Gradio Blocks arayuzunu olusturur."""
    with gr.Blocks(title="AI Kaynaklı Python Kodu Tespiti") as demo:
        gr.Markdown(
            """
            # 🔍 Yapay Zeka Kaynaklı Python Kodu Tespiti
            AST Stilometrisi (**XGBoost + SHAP**) ve Derin Öğrenme (**UniXcoder + LIME**) Modelleri ile Kod Doğrulama ve Açıklanabilirlik Demosu.
            Genel karar üç modelin (XGBoost+perplexity, UniXcoder, n-gram) birleşimidir; olasılıklar kalibre edilmiştir.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                code_input = gr.Textbox(
                    label="Python Fonksiyonu (Girdi)",
                    placeholder="Analiz edilecek Python fonksiyonunu buraya yapıştırın (örn: def my_func(...): ...)",
                    lines=13,
                    max_lines=30,
                )
                with gr.Row():
                    analyze_btn = gr.Button("⚡ Analiz Et", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Temizle", scale=1)

                status_output = gr.Markdown(value="Analiz için bir Python fonksiyonu girip 'Analiz Et' düğmesine tıklayınız.")

                gr.Examples(
                    examples=[
                        [EXAMPLE_HUMAN],
                        [EXAMPLE_AI],
                        [EXAMPLE_HARD_HUMAN],
                        [EXAMPLE_INVALID],
                    ],
                    example_labels=["İnsan (test)", "AI (test)", "Zor örnek: insan, model yanılıyor",
                                    "Geçersiz kod"],
                    inputs=[code_input],
                    label="Örnek Python Fonksiyonları",
                )

        ens_decision = gr.Textbox(label="Genel Karar (Topluluk Modeli)", interactive=False)

        with gr.Row():
            # XGBoost Kolonu
            with gr.Column(scale=1):
                gr.Markdown("### 🌳 XGBoost (AST Stilometrisi)")
                xgb_decision = gr.Textbox(label="Karar & Sonuç", interactive=False)
                xgb_info = gr.Markdown(label="Eşik & Olasılık")
                xgb_plot = gr.Plot(label="SHAP Özellik Katkıları")

            # CodeBERT Kolonu
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 UniXcoder (Transformer + LIME)")
                cb_decision = gr.Textbox(label="Karar & Sonuç", interactive=False)
                cb_info = gr.Markdown(label="Eşik & Olasılık")
                cb_html = gr.HTML(label="LIME Token Önem Açıklaması")

        # Buton tiklamalari
        analyze_btn.click(
            fn=analyze_code,
            inputs=[code_input],
            outputs=[
                status_output,
                xgb_decision, xgb_info, xgb_plot,
                cb_decision, cb_info, cb_html,
                ens_decision,
            ],
        )

        clear_btn.click(
            fn=lambda: ("", "Kod giriş alanı temizlendi.", "", "", None, "", "", "", ""),
            inputs=[],
            outputs=[
                code_input, status_output,
                xgb_decision, xgb_info, xgb_plot,
                cb_decision, cb_info, cb_html,
                ens_decision,
            ],
        )

    return demo


def main():
    demo = create_demo()
    # Adres: http://127.0.0.1:7860
    # Gradio 6: tema Blocks yerine launch() ile verilir.
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=theme)


if __name__ == "__main__":
    main()

