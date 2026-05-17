import os
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS
import time
from backend.core.config import get_settings

settings = get_settings()
print(f"Loading TTS model...")
tts = TTS(model_path="ai_models/xtts/xtts_v2", config_path="ai_models/xtts/xtts_v2/config.json", progress_bar=False, gpu=False)

ref_audio = "Rahul_Gandhi_Cleaned_Final.wav"
text = "भारत की जनता बहुत समझदार है।"

print("Generating with language='hi', do_sample=False (current settings)...")
out_path1 = "test_hi_greedy.wav"
tts.tts_to_file(
    text=text, 
    speaker_wav=ref_audio, 
    language="hi", 
    file_path=out_path1,
    split_sentences=False, do_sample=False, temperature=0.35, top_p=0.65, top_k=25, repetition_penalty=9.0, length_penalty=1.0
)

print("Generating with language='hi', do_sample=True, custom params...")
out_path2 = "test_hi_sample.wav"
tts.tts_to_file(
    text=text, 
    speaker_wav=ref_audio, 
    language="hi", 
    file_path=out_path2,
    split_sentences=False, do_sample=True, temperature=0.6, top_p=0.8, top_k=50, repetition_penalty=2.0
)

print("Generating with language='en' but hindi text...")
out_path3 = "test_en_hindi.wav"
try:
    tts.tts_to_file(
        text=text, 
        speaker_wav=ref_audio, 
        language="en", 
        file_path=out_path3,
        split_sentences=False, do_sample=False
    )
    print("Success with 'en'")
except Exception as e:
    print(f"Failed with 'en': {e}")

print("Done testing.")
