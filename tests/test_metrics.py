import math

from tez.metrics import best_threshold, compute_metrics


def test_mukemmel_tahmin():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0 and m["roc_auc"] == 1.0
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 0, 2)


def test_esik_uygulanir():
    m = compute_metrics([0, 1], [0.3, 0.4], threshold=0.35)
    assert m["accuracy"] == 1.0


def test_tek_sinifta_auc_nan():
    m = compute_metrics([1, 1], [0.6, 0.7])
    assert math.isnan(m["roc_auc"])


def test_en_iyi_esik_f1_yukseltir():
    y, p = [0, 0, 1, 1], [0.1, 0.45, 0.4, 0.9]
    t = best_threshold(y, p)
    assert compute_metrics(y, p, t)["f1"] >= compute_metrics(y, p)["f1"]
