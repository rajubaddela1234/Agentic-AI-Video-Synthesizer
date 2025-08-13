## tts_generator.py
from gtts import gTTS
import os

def text_to_speech(script_text: str, output_path: str = 'output.mp3') -> bool:
    """Convert script text to speech using Google TTS."""
    try:
        cleaned_text = ' '.join(script_text.replace('[', '').replace(']', '').split())
        gTTS(text=cleaned_text, lang='en').save(output_path)
        print(f" Audio saved to {output_path}")
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False

