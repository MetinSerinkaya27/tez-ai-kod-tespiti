import math

import pytest

from tez.features import FEATURE_NAMES, extract_features, features_frame

ORNEK = '''def f(a, b=1):
    if a is None:
        return None
    else:
        out = [x*2 for x in range(len(b))]
    for i, v in enumerate(a):
        out.append(v)
    s = 'x' + "y"
    return out
'''


def test_isimler_benzersiz_ve_sirali():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert list(extract_features(ORNEK)) == FEATURE_NAMES


def test_nan_ve_sonsuz_yok():
    for code in (ORNEK, "def f(): pass\n", "def f():\n    return\n"):
        assert all(math.isfinite(v) for v in extract_features(code).values())


def test_tek_satirlik_fonksiyon():
    f = extract_features("def f(): pass\n")
    assert f["num_statements"] == 1
    assert f["fmt_num_lines"] == 1
    assert f["param_count"] == 0
    assert f["local_name_count"] == 0
    assert f["fmt_indent_unit"] == 0
    # Hic string yoksa tirnak tercihi notr (0.5).
    assert f["fmt_single_quote_frac"] == 0.5


def test_parse_hatasi_valueerror():
    with pytest.raises(ValueError):
        extract_features("def f(:\n")


def test_yapisal_ve_deyimsel_sayimlar():
    f = extract_features(ORNEK)
    assert f["num_statements"] == 7
    assert f["return_count"] == 2
    assert f["early_return_count"] == 1
    assert f["cyclomatic_complexity"] == 4  # if + for + comprehension
    assert f["idiom_comprehension_rate"] == pytest.approx(1 / 7)
    assert f["idiom_is_none_rate"] == pytest.approx(1 / 7)
    assert f["idiom_enumerate_rate"] == pytest.approx(1 / 7)
    assert f["idiom_range_len_rate"] == pytest.approx(1 / 7)
    assert f["idiom_for_append_rate"] == pytest.approx(1 / 7)
    assert f["idiom_else_after_return_rate"] == pytest.approx(1 / 7)
    assert f["idiom_return_none_explicit_rate"] == pytest.approx(1 / 7)


def test_tanimlayici_ozellikleri():
    f = extract_features(ORNEK)
    # Yerel isimler: out, x, i, v, s (parametreler a, b sayilmaz).
    assert f["local_name_count"] == 5
    assert f["local_name_single_char_frac"] == pytest.approx(4 / 5)
    assert f["param_count"] == 2
    assert f["param_default_frac"] == pytest.approx(0.5)


def test_self_parametre_sayilmaz_ve_isim_stilleri():
    code = "def m(self, x: int) -> int:\n    my_value = x\n    otherValue = 2\n    return my_value + otherValue\n"
    f = extract_features(code)
    assert f["param_count"] == 1
    assert f["param_annotation_frac"] == 1.0
    assert f["has_return_annotation"] == 1.0
    assert f["local_name_snake_frac"] == 0.5
    assert f["local_name_camel_frac"] == 0.5


def test_tirnak_ve_bosluk_oranlari():
    f = extract_features("def f(a,b):\n    x = 'a' + 'b'\n    y=a+\"c\"\n    return g(k=1, j = 2)\n")
    assert f["fmt_single_quote_frac"] == pytest.approx(2 / 3)
    assert f["fmt_binop_spaced_frac"] == pytest.approx(1 / 2)
    assert f["fmt_assign_spaced_frac"] == pytest.approx(1 / 2)
    assert f["fmt_kwarg_eq_spaced_frac"] == pytest.approx(1 / 2)
    assert f["fmt_comma_space_frac"] == pytest.approx(1 / 2)


def test_fstring_ve_format_sayimi():
    code = "def f(a):\n    x = f'{a}'\n    y = '{}'.format(a)\n    return '%s' % a\n"
    f = extract_features(code)
    assert f["fmt_fstring_rate"] == pytest.approx(1 / 3)
    assert f["fmt_format_call_rate"] == pytest.approx(1 / 3)
    assert f["fmt_percent_format_rate"] == pytest.approx(1 / 3)
    assert f["fmt_single_quote_frac"] == 1.0


def test_girinti_birimi_ve_csn_metot_artigi():
    iki = extract_features("def f(a):\n  if a:\n    return 1\n  return 2\n")
    assert iki["fmt_indent_unit"] == 2
    # CSN metotlarinda govde 8 bosluk; birim yine 4, satir uzunlugu etkilenmez.
    dort = extract_features("def f(a):\n    return a\n")
    sekiz = extract_features("def f(a):\n        return a\n")
    assert sekiz["fmt_indent_unit"] == dort["fmt_indent_unit"] == 4
    assert sekiz["fmt_line_len_mean"] == dort["fmt_line_len_mean"]


def test_istisna_deyimleri():
    code = (
        "def f():\n    try:\n        g()\n    except:\n        pass\n"
        "    try:\n        g()\n    except Exception as e:\n        raise ValueError() from e\n"
    )
    f = extract_features(code)
    assert f["idiom_bare_except_rate"] > 0
    assert f["idiom_broad_except_rate"] > 0
    assert f["idiom_raise_from_rate"] > 0


def test_features_frame_kolon_sirasi():
    df = features_frame([ORNEK, "def f(): pass\n"])
    assert list(df.columns) == FEATURE_NAMES
    assert df.shape == (2, len(FEATURE_NAMES))
    assert not df.isna().any().any()
