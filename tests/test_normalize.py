import ast

from tez.normalize import extract_function, function_header, normalize_code


def test_docstring_ve_yorumlar_silinir():
    code = '''
def f(x):
    """Aciklama."""
    # tek basina yorum
    y = x + 1  # satir sonu yorumu
    return y
'''
    out = normalize_code(code)
    assert out == "def f(x):\n    y = x + 1\n    return y\n"


def test_yalnizca_docstring_olan_govde_pass_olur():
    code = 'def f():\n    """Sadece doc."""\n'
    out = normalize_code(code)
    assert out == "def f():\n    pass\n"
    ast.parse(out)


def test_cok_satirli_docstring_ve_sinif():
    code = '''
class A:
    """Sinif

    dokumani.
    """
    def m(self):
        """Metot."""
        return 1
'''
    out = normalize_code(code)
    assert '"""' not in out
    assert "return 1" in out


def test_string_icindeki_diyez_korunur():
    out = normalize_code('def f():\n    return "#etiket"\n')
    assert '"#etiket"' in out


def test_bos_satirlar_teke_iner():
    out = normalize_code("a = 1\n\n\n\nb = 2\n")
    assert out == "a = 1\n\nb = 2\n"


def test_gecersiz_kod_none_doner():
    assert normalize_code("def f(:\n") is None


def test_markdown_blogundan_fonksiyon_cikarilir():
    text = "Iste cozum:\n```python\nimport os\n\n@dec\ndef g(a):\n    return a\n```\nUmarim yardimci olur."
    assert extract_function(text, "g") == "@dec\ndef g(a):\n    return a\n"


def test_sinif_icindeki_metot_cikarilir():
    text = "class K:\n    def h(self):\n        return 2\n"
    assert extract_function(text, "h") == "def h(self):\n    return 2\n"


def test_fonksiyon_yoksa_none():
    assert extract_function("sadece metin") is None


def test_function_header():
    code = 'def f(a,\n      b):\n    """Topla."""\n    return a + b\n'
    sig, doc = function_header(code)
    assert sig == "def f(a,\n      b):"
    assert doc == "Topla."


def test_function_header_docstring_yoksa_none():
    assert function_header("def f():\n    return 1\n") is None
