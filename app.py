import streamlit as st
import os
import time
from io import BytesIO
import zipfile
from datetime import datetime

# Import your existing modules
try:
    from search import (
        build_research_graph, 
        VideoState, 
        format_citations, 
        format_all_answers,
        apis_available
    )
    from video import SimpleVideoGenerator
    from tts_generator import text_to_speech
except ImportError as e:
    st.error(f" Required modules not found: {e}")
    st.stop()

# Define create_video_from_text using SimpleVideoGenerator
def create_video_from_text(script_content, progress_callback=None):
    generator = SimpleVideoGenerator()
    # Optionally, you can pass progress_callback if your generator supports it
    video_data = generator.create_video(script_content)
    return video_data

# Define create_audio_from_text using text_to_speech
def create_audio_from_text(text):
    output_path = "output.mp3"
    text_to_speech(text, output_path)
    with open(output_path, "rb") as f:
        audio_data = f.read()
    if os.path.exists(output_path):
        os.remove(output_path)
    return audio_data

# Page configuration
st.set_page_config(
    page_title="Agentic AI Video Synthesizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .status-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .step-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .download-section {
        background: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .video-container {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .video-container video {
        max-width: 150px;
        max-height: 150px;
        width: 100%;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'research_complete' not in st.session_state:
    st.session_state.research_complete = False
if 'script_content' not in st.session_state:
    st.session_state.script_content = ""
if 'research_data' not in st.session_state:
    st.session_state.research_data = {}
if 'citations' not in st.session_state:
    st.session_state.citations = []
if 'question_answers' not in st.session_state:
    st.session_state.question_answers = {}
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'all_content_ready' not in st.session_state:
    st.session_state.all_content_ready = False

def run_research_pipeline(user_query):
    """Run the research pipeline with simple spinner"""
    
    # Build graph
    graph = build_research_graph()
    
    initial_state = {
        "query": user_query,
        "original_query": user_query,
        "next_agent": "question_decomposer",
        "youtube_data": {},
        "tavily_data": {},
        "arxiv_data": {},
        "completed_agents": [],
        "final_answer": "",
        "citations": [],
        "supervisor_decision": "",
        "feedback_score": 0,
        "iteration_count": 0,
        "feedback_reason": "",
        "rewritten_query": "",
        "all_intermediate_answers": [],
        "sub_questions": [],
        "current_question_index": 0,
        "question_answers": {},
        "all_questions_answered": False,
        "react_decision": ""
    }
    
    try:
        # Execute the research with memory configuration
        config = {
            "recursion_limit": 100,
            "configurable": {"thread_id": f"streamlit_session_{hash(user_query) % 10000}"}  # Add this line
        }
        
        with st.spinner("🧠 Processing your request... This may take a few seconds."):
            final_state = graph.invoke(initial_state, config=config)
        
        st.success("✅ Research completed successfully!")
        return final_state
        
    except Exception as e:
        st.error(f"❌ Research failed: {str(e)}")
        return None

def create_download_zip(script_content, research_report, audio_data, video_data=None):
    """Create a ZIP file with all generated content"""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add script
        zip_file.writestr("video_script.txt", script_content)
        
        # Add research report
        zip_file.writestr("research_report.txt", research_report)
        
        # Add audio
        if audio_data:
            zip_file.writestr("audio.mp3", audio_data)
        
        # Add video if available
        if video_data:
            zip_file.writestr("video.mp4", video_data)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_file.writestr("generation_info.txt", f"Generated on: {timestamp}")
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Agentic AI Video Synthesizer</h1>
        <p>An intelligent, automated system that transforms user queries into personalized, educational videos using cutting-edge artificial intelligence technologies including Large Language Models, speech synthesis, avatar rendering, and real-time content retrieval.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - API Status
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Check API availability
        api_status = {
            "GROQ (LLM)": bool(os.getenv("GROQ_API_KEY")),
            "Tavily (Web Search)": bool(os.getenv("TAVILY_API_KEY")),
            "YouTube API": bool(os.getenv("YOUTUBE_API_KEY")),
            "D-ID (Video)": bool(os.getenv("DID_API_KEY")),
            "LangSmith (Tracing)": bool(os.getenv("LANGSMITH_API_KEY"))
        }
        
        for api, status in api_status.items():
            if status:
                st.success(f"✅ {api}")
            else:
                st.error(f"❌ {api}")
        
        st.markdown("---")
        st.markdown("### 📋 Features")
        st.markdown("""
        - 🧠 Multi-question analysis
        - 🎥 YouTube research
        - 🌐 Web content search
        - 📄 Academic paper analysis
        - 🔊 Audio generation
        - 🎬 AI avatar video
        - 📁 Download all files
        """)
    
    # Main interface
    st.markdown("### 📝 Enter Your Query")
    user_query = st.text_area(
        "What would you like to create a video about?",
        placeholder="Example: What is artificial intelligence and how does machine learning work?",
        height=100
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        generate_btn = st.button("🚀 Generate Content", type="primary", use_container_width=True)
    with col_btn2:
        if st.session_state.research_complete:
            clear_btn = st.button("🗑️ Clear Results", use_container_width=True)
            if clear_btn:
                # Reset all session state
                for key in ['research_complete', 'script_content', 'research_data', 'citations', 
                           'question_answers', 'audio_file', 'video_file', 'all_content_ready']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Processing
    if generate_btn and user_query:
        if not any(api_status.values()):
            st.error("❌ No API keys configured. Please set up at least GROQ_API_KEY in environment variables.")
            return
        
        # Run research with simple spinner
        final_state = run_research_pipeline(user_query)
        
        if final_state:
            # Extract results
            st.session_state.script_content = final_state.get("final_answer", "")
            st.session_state.citations = final_state.get("citations", [])
            st.session_state.question_answers = final_state.get("question_answers", {})
            st.session_state.research_data = {
                'all_answers': final_state.get("all_intermediate_answers", []),
                'feedback_score': final_state.get("feedback_score", 0),
                'feedback_reason': final_state.get("feedback_reason", ""),
                'iteration_count': final_state.get("iteration_count", 0) + 1,
                'sub_questions': final_state.get("sub_questions", [])
            }
            st.session_state.research_complete = True
            st.rerun()
    
    # Avatar Video Generation (appears below user query when content is ready)
    if st.session_state.research_complete and st.session_state.script_content:
        st.markdown("---")
        st.markdown("### 🎬 Generated Avatar Video")
        
        if not st.session_state.video_file:
            if os.getenv("DID_API_KEY"):
                if st.button("🎬 Generate Avatar Video", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(message):
                        status_text.text(message)
                    
                    try:
                        with st.spinner("Generating AI avatar video..."):
                            video_data = create_video_from_text(
                                st.session_state.script_content, 
                                progress_callback
                            )
                            if video_data:
                                st.session_state.video_file = video_data
                                progress_bar.progress(1.0)
                                status_text.success("✅ Avatar video generated successfully!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ Video generation failed: {str(e)}")
            else:
                st.warning("⚠️ D-ID API key required for video generation. Please set DID_API_KEY in environment variables.")
        else:
            st.success("✅ Avatar video ready!")
            # Display video with medium size using columns for better control
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.video(st.session_state.video_file)
    
    # Results section (only show after research is complete)
    if st.session_state.research_complete:
        st.markdown("---")
        st.markdown("## 🎉 Results Generated Successfully!")
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Quality Score</h4>
                <h2>{st.session_state.research_data.get('feedback_score', 0)}/10</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔍 Questions Found</h4>
                <h2>{len(st.session_state.research_data.get('sub_questions', []))}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📚 Sources</h4>
                <h2>{len(st.session_state.citations)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔄 Iterations</h4>
                <h2>{st.session_state.research_data.get('iteration_count', 1)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs for different outputs
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Video Script", "📊 Research Report", "🔊 Audio", "📁 Downloads"])
        
        with tab1:
            st.markdown("### 📝 Generated Video Script")
            if st.session_state.script_content:
                st.markdown(f"""
                <div class="step-container">
                    <h4>🎬 Final Script ({len(st.session_state.script_content)} characters)</h4>
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 5px; font-family: Georgia, serif; line-height: 1.6;">
                        {st.session_state.script_content.replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 📊 Detailed Research Report")
            
            # Research steps
            if 'all_answers' in st.session_state.research_data:
                st.markdown("#### 🔍 Research Process")
                for i, answer in enumerate(st.session_state.research_data['all_answers'], 1):
                    with st.expander(f"Step {i}: {answer.get('agent', 'Unknown Agent')}"):
                        st.markdown(f"**Query:** {answer.get('query', 'N/A')}")
                        st.markdown(f"**Sources Found:** {answer.get('sources_count', 0)}")
                        st.markdown("**Analysis:**")
                        st.text(answer.get('answer', 'No analysis available'))
            
            # Question-specific resources
            if st.session_state.question_answers:
                st.markdown("#### 📋 Question-Specific Resources")
                for i, (question, data) in enumerate(st.session_state.question_answers.items(), 1):
                    with st.expander(f"Question {i}: {question}"):
                        col_yt, col_web, col_arxiv = st.columns(3)
                        
                        with col_yt:
                            youtube_data = data.get('youtube_data', {})
                            st.markdown(f"**🎥 YouTube ({youtube_data.get('videos_count', 0)} videos)**")
                            if youtube_data.get('raw_results'):
                                for video in youtube_data['raw_results']:
                                    st.markdown(f"- [{video.get('title', 'Unknown')}]({video.get('url', '#')})")
                        
                        with col_web:
                            tavily_data = data.get('tavily_data', {})
                            st.markdown(f"**🌐 Web ({tavily_data.get('results_count', 0)} articles)**")
                            if tavily_data.get('raw_results'):
                                for article in tavily_data['raw_results']:
                                    if article.get('type') != 'ai_summary':
                                        st.markdown(f"- [{article.get('title', 'Unknown')}]({article.get('url', '#')})")
                        
                        with col_arxiv:
                            arxiv_data = data.get('arxiv_data', {})
                            st.markdown(f"**📄 Academic ({arxiv_data.get('papers_count', 0)} papers)**")
                            if arxiv_data.get('raw_results'):
                                for paper in arxiv_data['raw_results']:
                                    st.markdown(f"- [{paper.get('title', 'Unknown')}]({paper.get('url', '#')})")
        
        with tab3:
            st.markdown("### 🔊 Generated Audio")
            
            if not st.session_state.audio_file:
                if st.button("🎵 Generate Audio", type="primary"):
                    with st.spinner("Generating audio..."):
                        text_to_speech(st.session_state.script_content, "output.mp3")
                        with open("output.mp3", "rb") as f:
                            st.session_state.audio_file = f.read()
                        if os.path.exists("output.mp3"):
                            os.remove("output.mp3")
                        st.success("✅ Audio generated successfully!")
                        st.rerun()
            else:
                st.success("✅ Audio ready!")
                st.audio(st.session_state.audio_file, format="audio/mp3")
                
                # Download button for audio
                st.download_button(
                    label="⬇️ Download Audio",
                    data=st.session_state.audio_file,
                    file_name=f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                    mime="audio/mp3",
                    use_container_width=True
                )
        
        with tab4:
            st.markdown("### 📁 Download All Files")
            
            # Only show download options when all content is ready
            if st.session_state.video_file and st.session_state.audio_file:
                st.markdown("""
                <div class="download-section">
                    <h4>🎁 Complete Package Ready</h4>
                    <p>All content has been generated. Download your complete video project:</p>
                    <ul>
                        <li>📝 Video Script (TXT)</li>
                        <li>📊 Research Report (TXT)</li>
                        <li>🔊 Audio File (MP3)</li>
                        <li>🎬 Avatar Video (MP4)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Create comprehensive report
                feedback_info = {
                    'final_score': st.session_state.research_data.get('feedback_score', 0),
                    'total_iterations': st.session_state.research_data.get('iteration_count', 1),
                    'final_reason': st.session_state.research_data.get('feedback_reason', '')
                }
                
                comprehensive_report = f"""COMPREHENSIVE VIDEO PROJECT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

ORIGINAL QUERY: {user_query}

ANALYSIS SUMMARY:
- Questions Identified: {len(st.session_state.research_data.get('sub_questions', []))}
- Total Sources Found: {len(st.session_state.citations)}
- Quality Score: {feedback_info['final_score']}/10
- Processing Iterations: {feedback_info['total_iterations']}

{'='*80}
VIDEO SCRIPT
{'='*80}
{st.session_state.script_content}

{'='*80}
RESEARCH PROCESS
{'='*80}
{format_all_answers(st.session_state.research_data.get('all_answers', []), feedback_info)}

{'='*80}
SOURCES AND CITATIONS
{'='*80}
{format_citations(st.session_state.citations, st.session_state.question_answers)}
"""
                
                # Create ZIP
                zip_data = create_download_zip(
                    st.session_state.script_content,
                    comprehensive_report,
                    st.session_state.audio_file,
                    st.session_state.video_file
                )
                
                st.download_button(
                    label="📦 Download Complete Package (ZIP)",
                    data=zip_data,
                    file_name=f"agentic_video_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                
                # Individual downloads
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        label="📝 Script Only",
                        data=st.session_state.script_content,
                        file_name=f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                    if st.session_state.audio_file:
                        st.download_button(
                            label="🎵 Audio Only",
                            data=st.session_state.audio_file,
                            file_name=f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                            mime="audio/mp3",
                            use_container_width=True
                        )
                
                with col_d2:
                    st.download_button(
                        label="📊 Report Only",
                        data=comprehensive_report,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                    
                    if st.session_state.video_file:
                        st.download_button(
                            label="🎬 Video Only",
                            data=st.session_state.video_file,
                            file_name=f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
            else:
                st.info("📋 Generate all content (script, audio, and video) to enable downloads.")

if __name__ == "__main__":
    main()