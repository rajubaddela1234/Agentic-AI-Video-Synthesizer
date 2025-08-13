# video.py
import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class SimpleVideoGenerator:
    def __init__(self):
        self.api_key = os.getenv("DID_API_KEY")
        if not self.api_key:
            raise ValueError("Set DID_API_KEY in environment variables")
        
        self.base_url = "https://api.d-id.com"
        self.headers = {
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json"
        }

        # actual Emy presenter_id
        self.presenter_id = "emy-custom-abc123"  # Emy avatar ID
        self.voice_id = "en-US-JennyNeural"  # Emy has a preferred one

    def create_video(self, script_text, progress_callback=None):
        """
        Create video with given script text and optional progress callback
        
        Args:
            script_text (str): The script content to convert to video
            progress_callback (callable, optional): Function to call with progress updates
        
        Returns:
            bytes: Video data as bytes
        """
        if progress_callback:
            progress_callback("Processing script...")
        
        print(f"Script length: {len(script_text)} characters")

        # Truncate if too long
        if len(script_text) > 500:
            script_text = script_text[:500]
            if progress_callback:
                progress_callback("Script truncated to 500 characters")

        payload = {
            "script": {
                "type": "text",
                "input": script_text,
                "provider": {
                    "type": "microsoft",
                    "voice_id": self.voice_id
                }
            },
            "presenter_id": self.presenter_id,
            "background": {"type": "color", "color": "#FFFFFF"},
            "config": {"result_format": "mp4", "fluent": True}
        }

        if progress_callback:
            progress_callback("Creating video with Emy avatar...")
        
        print(f"Creating video with Emy avatar...")
        response = requests.post(f"{self.base_url}/talks", headers=self.headers, json=payload)
        
        if response.status_code != 201:
            raise Exception(f"Failed to create video: {response.text}")
        
        video_id = response.json()["id"]
        print(f"Video ID: {video_id}")
        
        if progress_callback:
            progress_callback(f"Video ID: {video_id}")

        # Wait until generation completes
        if progress_callback:
            progress_callback("Waiting for video generation...")
        
        print(" Waiting for video generation...")
        while True:
            status_response = requests.get(f"{self.base_url}/talks/{video_id}", headers=self.headers)
            status_data = status_response.json()
            status = status_data.get("status")
            
            if status == "done":
                video_url = status_data["result_url"]
                if progress_callback:
                    progress_callback("⬇Downloading video...")
                
                print(f"⬇ Downloading video...")
                video_response = requests.get(video_url)
                
                if progress_callback:
                    progress_callback(" Video generation completed!")
                
                print(f" Video generated successfully!")
                # Return video data as bytes instead of saving to file
                return video_response.content
            
            elif status == "error":
                error_msg = status_data.get('error', {}).get('description', 'Unknown error')
                raise Exception(f" Video generation failed: {error_msg}")
            
            elif status == "started":
                if progress_callback:
                    progress_callback(" Video generation in progress...")
            
            time.sleep(10)

    def create_video_legacy(self):
        """
        Legacy method that reads from video_script.txt file
        (kept for backward compatibility)
        """
        # Read entire script
        script_text = Path("video_script.txt").read_text().strip()
        return self.create_video(script_text)

# Usage
def main():
    try:
        generator = SimpleVideoGenerator()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
