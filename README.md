git statusgit status# AI Video Script Research Agent

This project is an AI-powered research assistant that generates comprehensive video scripts (6-10 minutes) on any topic by aggregating information from YouTube, web articles, and academic papers. It uses advanced LLMs and multiple APIs to ensure high-quality, well-cited scripts suitable for video content creation.

## Features
- Multi-source research: YouTube, web, and arXiv academic papers
- Uses LLM (Meta Llama via Groq API) for synthesis
- Generates a video script and detailed research report
- Text-to-speech (TTS) audio generation (output.mp3)
- Modular, extensible Python codebase

## Project Structure
- `main.py`: Main entry point. Generates script and audio.
- `youtube_search.py`: Core research agent. Handles API calls, research graph, and script/report generation.
- `tts_generator.py`: Converts generated script to speech (MP3).
- `avatar_video.py`: (video/avatar features)
- `requirements.txt`: Python dependencies.
- `output.mp3`: Generated audio file (ignored by git).
- `video_script.txt`, `research_report.txt`, `detailed_report.txt`: Generated output files (ignored by git).
- `venv/`: Python virtual environment (ignored by git).

## Setup
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up API keys**:
   - Create a `.env` file in the root directory with the following (get keys from Tavily, YouTube, Groq):
     ```env
     TAVILY_API_KEY=your_tavily_api_key
     YOUTUBE_API_KEY=your_youtube_api_key
     GROQ_API_KEY=your_groq_api_key
     ```

## Usage
Run the main script:
```bash
python main.py
```
- Enter your video topic when prompted.
- The script will generate a research-backed video script and save it to `video_script.txt`.
- An audio narration will be generated as `output.mp3`.
- A detailed research report will be saved as `research_report.txt`.

## Notes
- Ensure your API keys are valid and have sufficient quota.
- All generated files and virtual environment folders are ignored by git (see `.gitignore`).
- `avatar_video.py` is a placeholder for future video/avatar generation features.

## License
MIT License 