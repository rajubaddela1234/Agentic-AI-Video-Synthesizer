# 🧠 Agentic AI Video Synthesizer

An advanced multi-agent AI system that transforms natural language queries into complete educational videos — with research-grounded scripting, TTS narration, and AI avatar rendering.

---

## 🚀 Overview

Agentic AI Video Synthesizer is an intelligent, multi-stage pipeline that autonomously:

1. Understands complex queries
2. Decomposes them into sub-questions
3. Gathers accurate data from web, video, and academic sources
4. Synthesizes a coherent educational script
5. Converts the script to speech using gTTS
6. Renders an avatar-narrated video using D-ID

This system is built using **LangGraph**, **Groq-hosted LLaMA 3.3–70B Versatile**, and several modern APIs, and delivered via an interactive **Streamlit** frontend.

---

## 🧩 Features

- ✅ Multi-agent LangGraph orchestration
- 🔍 Real-time research using:
  - 🌐 Tavily (Web)
  - 📺 YouTube (Video Transcripts)
  - 📚 arXiv (Academic Papers)
- 📝 Script synthesis grounded in actual retrieved content
- 🔊 Natural speech generation (gTTS)
- 👩‍🏫 Avatar video rendering (D-ID)
- 💾 Downloadable outputs: script, audio, video, citations
- 📊 Quality evaluation (score, completeness, iterations, source count)

---

## 🛠️ Tech Stack

| Layer                    | Technology                                 |
|-------------------------|--------------------------------------------|
| 🧠 LLM Engine            | Groq API with `llama-3.3-70b-versatile`     |
| 🕸️ Agent Framework       | LangGraph + LangChain                      |
| 🔎 Retrieval Agents      | Tavily, YouTube API, arXiv API             |
| 🗣️ Text-to-Speech        | Google Text-to-Speech (gTTS)              |
| 🎥 Avatar Video          | D-ID API                                   |
| 🌐 UI & Deployment       | Streamlit                                  |
| 🐍 Language              | Python 3.11                                |

---

## 📂 Project Structure

```bash
├── app.py                 # Streamlit UI
├── main.py               # LangGraph Orchestration
├── search.py             # Tool Agent integrations (Tavily, YouTube, arXiv)
├── tts_generator.py      # gTTS speech synthesis
├── video.py              # D-ID avatar video handling
├── .env                  # API keys (excluded from Git)
├── requirements.txt
```

---

## ⚙️ Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/agentic-ai-video-synthesizer.git
cd agentic-ai-video-synthesizer
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure `.env` file

Create a `.env` file in the root directory with:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
DID_API_KEY=your_did_key
YOUTUBE_API_KEY=your_youtube_key
LANGSMITH=your_langsmith_key
```

### 5. Run the app

```bash
streamlit run app.py
```

---



## 🧪 Evaluation Summary

- Single & multi-question queries tested
- Real-time research with score-based feedback
- Metrics include: quality score, iterations, sources used
- Screenshots and results available in `/screenshots`

---

## 📚 References

- LLaMA 3.3–70B Versatile via Groq
- Tavily Web Search API
- YouTube Data API v3
- arXiv.org API
- LangGraph by LangChain
- Google Text-to-Speech
- D-ID Avatar Rendering

---

## 🙌 Acknowledgements

Special thanks to **Dr. Sameena Naaz** for her insightful feedback and technical guidance on LangGraph agent integration and D-ID video rendering.

---

## 📦 Deliverables

- ✅ Educational Script (.txt)
- ✅ Audio Narration (.mp3)
- ✅ AI Avatar Video (.mp4)
- ✅ Research Report (.txt)

---

## 📌 Status

✅ Functional Prototype | 🧠 Agentic Architecture | 📹 AI-Narrated Videos  
Ready for educational use and future extensions!
