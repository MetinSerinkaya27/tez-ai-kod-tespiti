"""app.demo modulu icin birim testler."""
import pytest
from app.demo import (
    validate_code,
    format_decision,
    is_xgb_available,
    is_codebert_available,
    analyze_code,
    EXAMPLE_HUMAN,
    EXAMPLE_AI,
    EXAMPLE_INVALID,
)


def test_validate_code_gecerli():
    valid, msg = validate_code(EXAMPLE_HUMAN)
    assert valid is True
    assert msg == ""


def test_validate_code_sozdizimi_hatasi():
    valid, msg = validate_code(EXAMPLE_INVALID)
    assert valid is False
    assert "SyntaxError" in msg or "Sözdizimi Hatası" in msg


def test_validate_code_fonksiyonsuz():
    valid, msg = validate_code("x = 1 + 2\nprint(x)")
    assert valid is False
    assert "fonksiyon" in msg.lower()


def test_format_decision():
    # Esige cok yakin: emin degil
    res_close = format_decision(0.52, 0.50, margin=0.05)
    assert "Emin değil" in res_close

    # Esikten yuksek: AI yazmis olabilir
    res_ai = format_decision(0.87, 0.50, margin=0.05)
    assert "AI yazmış olabilir (%87)" in res_ai

    # Esikten dusuk: Insan yazmis olabilir
    res_human = format_decision(0.20, 0.50, margin=0.05)
    assert "İnsan yazmış olabilir" in res_human


def test_analyze_code_model_olmadan_cokmez():
    # Modeller henuz egitilmediyse arayuz cokmemeli, uygun uyari vermeli
    status, xgb_dec, xgb_info, xgb_fig, cb_dec, cb_info, cb_html, ens_dec = analyze_code(EXAMPLE_HUMAN)
    assert "başarıyla doğrulandı" in status
    if not is_xgb_available():
        assert "Model henüz eğitilmedi" in xgb_dec
        assert xgb_fig is None
    if not is_codebert_available():
        assert "Model henüz eğitilmedi" in cb_dec
        assert "Model henüz eğitilmedi" in cb_html

