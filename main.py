# main.py
import os
from youtube_search import main as research_main
from gtts import gTTS
from tts_generator import text_to_speech

def main():
    
    # Generate script
    print("📝 Generating script...")
    research_main()

    # Create audio
    print("🔊 Creating audio...")
    with open("video_script.txt", "r") as f:
        script = f.read()
    text_to_speech(script)
    
    
if __name__ == "__main__":
    main()
