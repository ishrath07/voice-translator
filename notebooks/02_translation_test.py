from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "नमस्ते, यह एक परीक्षण है"
inputs = tokenizer(text, return_tensors="pt", padding=True)
translated = model.generate(**inputs)
output = tokenizer.decode(translated[0], skip_special_tokens=True)

print("Input:", text)
print("Translated:", output)