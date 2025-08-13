# main.py
import os
from search import main as research_main
from gtts import gTTS
from tts_generator import text_to_speech
from video import main as create_avatar_video_from_script

def main():
    # Generate script
    print(" Generating script...")
    research_main()

    # Create audio
    print(" Creating audio...")
    with open("video_script.txt", "r") as f:
        script = f.read()
    text_to_speech(script)

    # Generate avatar video
    print("Generating avatar video...")
    create_avatar_video_from_script()

if __name__ == "__main__":
    main()
