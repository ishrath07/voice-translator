import whisper

model = whisper.load_model("base")
print("Model loaded successfully")

result_en = model.transcribe("data/samples/english_sample.mp3")
print("\n--- English (base) ---")
print("Transcript:", result_en["text"])
print("Detected language:", result_en["language"])

result_hi = model.transcribe("data/samples/hindi_sample.mp3")
print("\n--- Hindi (base) ---")
print("Transcript:", result_hi["text"])
print("Detected language:", result_hi["language"])

result_kn = model.transcribe("data/samples/kannada_sample.mp3")
print("\n--- Kannada (base) ---")
print("Transcript:", result_kn["text"])
print("Detected language:", result_kn["language"])

print("\n--- Summary (base model) ---")
print("English  → detected:", result_en["language"], "| expected: en")
print("Hindi    → detected:", result_hi["language"], "| expected: hi")
print("Kannada  → detected:", result_kn["language"], "| expected: kn")

# --- Try a larger model on Kannada to check for accuracy improvement ---
print("\nLoading 'small' model to re-test Kannada...")
model_small = whisper.load_model("small")

result_kn_small = model_small.transcribe("data/samples/kannada_sample.mp3")
print("\n--- Kannada (small) ---")
print("Transcript:", result_kn_small["text"])
print("Detected language:", result_kn_small["language"])

print("\n--- Model comparison for Kannada ---")
print("base  → detected:", result_kn["language"], "| transcript:", result_kn["text"])
print("small → detected:", result_kn_small["language"], "| transcript:", result_kn_small["text"])