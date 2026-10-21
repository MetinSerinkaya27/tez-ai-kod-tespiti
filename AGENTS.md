# Ajanlar icin calisma kurallari

Bu depo bir yuksek lisans/lisans tez projesidir: yapay zeka kaynakli Python
fonksiyonlarinin AST stilometrisi (XGBoost + SHAP) ve CodeBERT (+ LIME) ile
tespiti. Projede birden fazla yapay zeka ajani ayni anda calisabilir; asagidaki
kurallar herkes icindir.

## Mutlak kurallar
- **Ucretli servis yok.** API anahtari, hesap ya da token gerektiren hicbir
  servis kullanilmaz. LLM uretimi yerelde Ollama ile yapilir.
- **`data/` altina yazma.** Ozellikle `data/interim/ai.jsonl` ve
  `ai_failed.jsonl` arka planda calisan uretim tarafindan yaziliyor; bu
  dosyalari acip duzenleme, silme, tasima. Okuman gerekirse once kopyala.
- **GPU ve Ollama'yi kullanma.** `scripts/run_all.py` arka planda calisiyor ve
  GPU'yu (6 GB) dolduruyor. Ollama'ya istek atma, modelleri durdurma.
- **Commitlerde ajan imzasi yok.** Co-Authored-By, "Generated with ..." gibi
  satirlar eklenmez; commitler kullanicinin kimligiyle atilir.
- **Baskasinin dosyasina dokunma.** Hangi dosyalarin kimde oldugu proje
  defterindeki "Gorev dagilimi" bolumunde yazar. Emin degilsen yeni dosya ac.

## Kod usulu
- Python 3.12, ortam `uv` ile: `uv run python scripts/...`, testler
  `uv run pytest -q`.
- Betikler `scripts/NN_*.py`, ortak kod `src/tez/`, yollar `tez.paths`'ten.
- Yorumlar ve ciktilar Turkce, ASCII (Turkce karakter yok). Test isimleri Turkce.
- Yeni bagimlilik eklemeden once kullaniciya sor.

## Veri ve degerlendirme ilkeleri
- Bolme depo bazlidir; ayni depo hem egitimde hem testte olamaz.
- Hiperparametre ve esik yalnizca dogrulama kumesinde secilir; test kumesine
  bir kez, sabit esikle bakilir.
- Modellere yalnizca hazir arayuzlerle baglan: `tez.xgb_model.explain_one`,
  `XGBDetector`, `tez.codebert.CodeBertDetector` (`predict_proba`,
  `explain_one`). Model dosyalari `models/` altinda (henuz egitilmemis olabilir).

## Calistirma sirasi
`scripts/run_all.py` tum adimlari sirayla calistirir; durumu `data/run_all.log`.
