from transformers import MarianMTModel, MarianTokenizer

# Cache loaded models so we don't reload from disk on every call
_loaded_models = {}

# Confirmed direct MarianMT pairs for this project
MODEL_MAP = {
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
}

def _load(model_name: str):
    if model_name not in _loaded_models:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _loaded_models[model_name] = (tokenizer, model)
    return _loaded_models[model_name]

def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translates text from src_lang to tgt_lang using MarianMT.
    Returns translated text, or a fallback message if the pair isn't supported.
    """
    if src_lang == tgt_lang:
        return text  # no translation needed

    key = (src_lang, tgt_lang)
    if key not in MODEL_MAP:
        return f"[Translation not available for {src_lang} -> {tgt_lang} yet]"

    model_name = MODEL_MAP[key]
    tokenizer, model = _load(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Correctly spelled Hindi test (matches your verified sanity check)
    hindi_text = "नमस्ते, यह एक परीक्षण है"
    print("Hindi → English:", translate(hindi_text, "hi", "en"))

    # Kannada — no direct model, should hit fallback gracefully
    kannada_text = "ನಮಸ್ಕಾರ, ಇದು ಒಂದು ಪರೀಕ್ಷೆ"
    print("Kannada → English:", translate(kannada_text, "kn", "en"))

    # Same-language passthrough
    print("English → English:", translate("Hello there", "en", "en"))