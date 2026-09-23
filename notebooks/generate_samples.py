from gtts import gTTS

tts = gTTS("Hello, this is a test of the speech recognition system.", lang="en")
tts.save("data/samples/english_sample.mp3")

tts = gTTS("नमस्ते, यह एक परीक्षण है।", lang="hi")
tts.save("data/samples/hindi_sample.mp3")

tts = gTTS("ನಮಸ್ಕಾರ, ಇದು ಒಂದು ಪರೀಕ್ಷೆ.", lang="kn")
tts.save("data/samples/kannada_sample.mp3")