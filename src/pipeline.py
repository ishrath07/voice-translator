from src.transcribe import transcribe
from src.translate import translate
import os

def process_audio(audio_path: str, target_lang: str = "en") -> dict:
    """
    Full pipeline: audio -> transcript -> translation.
    Returns: {"transcript": str, "translation": str, "detected_lang": str, "error": str | None}
    """
    if not os.path.exists(audio_path):
        return {"transcript": None, "translation": None, "detected_lang": None,
                 "error": f"File not found: {audio_path}"}

    try:
        stt_result = transcribe(audio_path)
    except Exception as e:
        return {"transcript": None, "translation": None, "detected_lang": None,
                 "error": f"Transcription failed: {e}"}

    transcript = stt_result["text"]
    detected_lang = stt_result["detected_lang"]

    if not transcript.strip():
        return {"transcript": "", "translation": "", "detected_lang": detected_lang,
                 "error": "No speech detected in audio (silence or unrecognized input)"}

    try:
        translation = translate(transcript, detected_lang, target_lang)
    except Exception as e:
        return {"transcript": transcript, "translation": None, "detected_lang": detected_lang,
                 "error": f"Translation failed: {e}"}

    return {
        "transcript": transcript,
        "translation": translation,
        "detected_lang": detected_lang,
        "error": None
    }

if __name__ == "__main__":
    print("--- Happy path ---")
    print(process_audio("data/samples/hindi_sample.mp3"))

    print("\n--- Missing file ---")
    print(process_audio("data/samples/does_not_exist.mp3"))