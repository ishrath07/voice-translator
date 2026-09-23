import whisper

_model = None  # load once, reuse across calls
_loaded_size = None

def load_model(model_size: str = "small"):
    global _model, _loaded_size
    if _model is None or _loaded_size != model_size:
        _model = whisper.load_model(model_size)
        _loaded_size = model_size
    return _model

def transcribe(audio_path: str, model_size: str = "small") -> dict:
    """
    Transcribes audio and detects language.
    Returns: {"text": str, "detected_lang": str}
    """
    model = load_model(model_size)
    result = model.transcribe(audio_path)
    return {
        "text": result["text"].strip(),
        "detected_lang": result["language"]
    }

if __name__ == "__main__":
    out = transcribe("data/samples/english_sample.mp3")
    print(out)