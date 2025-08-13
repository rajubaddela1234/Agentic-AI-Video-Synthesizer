###Research_system.py
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Literal, List, Dict, Any
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langsmith import Client
import langsmith
from IPython.display import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import sys
import json
import re
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

# --- API Keys with validation ---
tavily_api_key = os.getenv("TAVILY_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# Check which APIs are available
apis_available = {
    "groq": bool(groq_api_key),
    "tavily": bool(tavily_api_key),
    "youtube": bool(youtube_api_key),
    "langsmith": bool(langsmith_api_key)
}


print(" API Status Check:")
for api, status in apis_available.items():
    print(f" • {api.upper()}: {' Available' if status else ' Missing'}")


# --- Initialize LangChain LLM ---

if not groq_api_key:
    print(" GROQ API key is required. Please set GROQ_API_KEY in your environment.")
    sys.exit(1)


# Initialize LangSmith tracing 
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = "video-script-research-agent"
    print(" LangSmith tracing enabled")


try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key='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',
        temperature=0.2,  # Lower temp for higher accuracy
    )
    # Test the connection
    test_response = llm.invoke("Test")
    print("llama-3.3-70b-versatile")

except Exception as e:
    print(f" Failed to initialize LLM: {str(e)}")
    sys.exit(1)

# ---  Tavily Search Tool ---
class TavilySearchTool:
    def __init__(self, api_key: str = None, max_results: int = 5):
        self.api_key = api_key
        self.max_results = max_results
        self.base_url = "https://api.tavily.com/search"
        self.available = bool(api_key)
    
    def run(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced run method that returns full content from web pages"""
        if not self.available:
            return []
            
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",  # Changed to advanced for more content
                "include_answer": False,      # Include Tavily's summary
                "include_images": False,
                "include_raw_content": True, # Get full page content
                "max_results": self.max_results,
                "include_domains": [],
                "exclude_domains": []
            }
            
            response = requests.post(self.base_url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # Enhanced results with full content
            enhanced_results = []
            for result in results:
                enhanced_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "raw_content": result.get("raw_content", ""),
                    "published_date": result.get("published_date", ""),
                    "score": result.get("score", 0),
                    "snippet": result.get("content", "")[:200] + "..." if result.get("content") else ""
                }
                
                # Use raw_content if available, otherwise use content
                full_content = enhanced_result["raw_content"] or enhanced_result["content"]
                if full_content:
                    # Clean and limit content to avoid token limits
                    cleaned_content = self._clean_content(full_content)
                    enhanced_result["cleaned_content"] = cleaned_content[:5000] + "..." if len(cleaned_content) > 5000 else cleaned_content
                else:
                    enhanced_result["cleaned_content"] = ""
                
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            print(f"Tavily search error: {str(e)}")
            return []
    
    def _clean_content(self, content: str) -> str:
        """Clean and format web content for better LLM processing"""
        import re
        
        if not content:
            return ""
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common web artifacts
        content = re.sub(r'(Cookie|Privacy|Terms|Subscribe|Share|Follow us)', '', content, flags=re.IGNORECASE)
        
        # Remove navigation and menu items
        content = re.sub(r'(Home|About|Contact|Menu|Navigation)', '', content, flags=re.IGNORECASE)
        
        # Remove repeated characters
        content = re.sub(r'(.)\1{3,}', r'\1\1', content)
        
        # Clean up spacing
        content = re.sub(r'\n+', '\n', content)
        content = content.strip()
        
        return content

# ---  YouTube Search ---
class SimpleYouTubeSearch:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.available = bool(api_key)
    
    def get_video_details(self, video_id: str) -> Dict[str, str]:
        """Get video description and attempt to get captions"""
        details = {"description": "", "captions": ""}
        
        try:
            # Get video details (description)
            details_url = "https://www.googleapis.com/youtube/v3/videos"
            details_params = {
                "part": "snippet",
                "id": video_id,
                "key": self.api_key
            }
            
            response = requests.get(details_url, params=details_params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if data.get("items"):
                details["description"] = data["items"][0]["snippet"].get("description", "")
            
            # Try to get captions
            captions_url = "https://www.googleapis.com/youtube/v3/captions"
            captions_params = {
                "part": "snippet",
                "videoId": video_id,
                "key": self.api_key
            }
            
            captions_response = requests.get(captions_url, params=captions_params, timeout=30)
            captions_response.raise_for_status()
            
            captions_data = captions_response.json()
            if captions_data.get("items"):
                # Find English captions
                for caption in captions_data["items"]:
                    language = caption["snippet"].get("language", "")
                    if language.startswith("en"):  # English captions
                        caption_id = caption["id"]
                        
                        # Download caption content
                        download_url = f"https://www.googleapis.com/youtube/v3/captions/{caption_id}"
                        download_params = {
                            "key": self.api_key,
                            "tfmt": "srt"  
                        }
                        
                        transcript_response = requests.get(download_url, params=download_params, timeout=30)
                        if transcript_response.status_code == 200:
                            details["captions"] = transcript_response.text
                            break
                        
        except Exception as e:
            print(f"Error getting video details for {video_id}: {str(e)}")
        
        return details
    
    def run(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced run method that returns structured data with content"""
        if not self.available:
            return []
        
        try:
            # Search for videos
            base_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": 5,
                "key": self.api_key
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("items", []):
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]
                url = f"https://www.youtube.com/watch?v={video_id}"
                channel = item["snippet"]["channelTitle"]
                published = item["snippet"]["publishedAt"]
                
                # Get additional video content
                video_details = self.get_video_details(video_id)
                
                results.append({
                    "title": title,
                    "url": url,
                    "video_id": video_id,
                    "channel": channel,
                    "published": published,
                    "description": video_details["description"],
                    "captions": video_details["captions"]
                })
            
            return results
            
        except Exception as e:
            print(f"YouTube search error: {str(e)}")
            return []

# --- ArXiv Search Tool ---
class SimpleArxivSearch:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.available = True  # ArXiv API is free
    
    def get_paper_details(self, paper_id: str) -> Dict[str, str]:
        """Get detailed paper information including abstract and full content if available"""
        details = {"abstract": "", "full_text": "", "authors": ""}
        
        try:
            # Get paper details from ArXiv API
            params = {
                "search_query": f"id:{paper_id}",
                "start": 0,
                "max_results": 1
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            content = response.text
            
            # Parse XML content for detailed information
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(content)
                # Find the entry
                for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                    # Extract abstract
                    summary = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                    if summary is not None:
                        details["abstract"] = summary.text.strip() if summary.text else ""
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('.//{http://www.w3.org/2005/Atom}author'):
                        name = author.find('.//{http://www.w3.org/2005/Atom}name')
                        if name is not None and name.text:
                            authors.append(name.text.strip())
                    details["authors"] = ", ".join(authors)
                    
                    break
                    
            except ET.ParseError:
                # Fallback to regex parsing if XML parsing fails
                abstract_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
                if abstract_match:
                    details["abstract"] = abstract_match.group(1).strip()
            
            # For now, we'll use the abstract as the main content
            details["full_text"] = details["abstract"]
            
        except Exception as e:
            print(f"Error getting paper details for {paper_id}: {str(e)}")
        
        return details
    
    def _clean_paper_content(self, content: str) -> str:
        """Clean and format paper content for better LLM processing"""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common LaTeX artifacts
        content = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', content)
        content = re.sub(r'\$[^$]*\$', '[MATH]', content)  # Replace math with placeholder
        
        # Clean up spacing
        content = re.sub(r'\n+', '\n', content)
        content = content.strip()
        
        return content
    
    def run(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced run method that returns structured data with full paper content"""
        if not self.available:
            return []
        
        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": 5,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            content = response.text
            results = []
            
            # Enhanced XML parsing
            import xml.etree.ElementTree as ET
            try:
                root = ET.fromstring(content)
                
                for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                    # Extract basic info
                    title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                    id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')
                    published_elem = entry.find('.//{http://www.w3.org/2005/Atom}published')
                    summary_elem = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                    
                    if title_elem is not None and id_elem is not None:
                        title = title_elem.text.strip() if title_elem.text else "Unknown Title"
                        paper_url = id_elem.text.strip() if id_elem.text else ""
                        published = published_elem.text.strip() if published_elem is not None and published_elem.text else ""
                        abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else ""
                        
                        # Extract paper ID from URL
                        paper_id_match = re.search(r'arxiv\.org/abs/([^/]+)', paper_url)
                        paper_id = paper_id_match.group(1) if paper_id_match else ""
                        
                        # Get additional paper details
                        paper_details = self.get_paper_details(paper_id) if paper_id else {}
                        
                        # Extract authors
                        authors = []
                        for author in entry.findall('.//{http://www.w3.org/2005/Atom}author'):
                            name = author.find('.//{http://www.w3.org/2005/Atom}name')
                            if name is not None and name.text:
                                authors.append(name.text.strip())
                        
                        # Clean the abstract content
                        cleaned_abstract = self._clean_paper_content(abstract)
                        
                        results.append({
                            "title": title,
                            "url": paper_url,
                            "paper_id": paper_id,
                            "authors": ", ".join(authors),
                            "published": published,
                            "abstract": abstract,
                            "cleaned_abstract": cleaned_abstract,
                            "full_content": cleaned_abstract,  # Using abstract as full content
                            "pdf_url": paper_url.replace('/abs/', '/pdf/') + '.pdf' if paper_url else ""
                        })
                
            except ET.ParseError:
                # Fallback to regex parsing
                titles = re.findall(r'<title>(.*?)</title>', content, re.DOTALL)
                links = re.findall(r'<id>(http://arxiv\.org/abs/.*?)</id>', content)
                abstracts = re.findall(r'<summary>(.*?)</summary>', content, re.DOTALL)
                
                for i, (title, link) in enumerate(zip(titles[1:6], links[:5])): 
                    abstract = abstracts[i] if i < len(abstracts) else ""
                    cleaned_abstract = self._clean_paper_content(abstract)
                    
                    paper_id_match = re.search(r'arxiv\.org/abs/([^/]+)', link)
                    paper_id = paper_id_match.group(1) if paper_id_match else ""
                    
                    results.append({
                        "title": title.strip(),
                        "url": link,
                        "paper_id": paper_id,
                        "authors": "Unknown",
                        "published": "Unknown",
                        "categories": "Unknown",
                        "abstract": abstract,
                        "cleaned_abstract": cleaned_abstract,
                        "full_content": cleaned_abstract,
                        "pdf_url": link.replace('/abs/', '/pdf/') + '.pdf' if link else ""
                    })
            
            return results
            
        except Exception as e:
            print(f"ArXiv search error: {str(e)}")
            return []
        
# Initialize tools
tavily_tool = TavilySearchTool(api_key=tavily_api_key, max_results=5)
youtube_tool = SimpleYouTubeSearch(api_key=youtube_api_key)
arxiv_tool = SimpleArxivSearch()

# ---  State Type ---
class VideoState(TypedDict):
    query: str
    original_query: str
    next_agent: str
    youtube_data: Dict[str, Any]
    tavily_data: Dict[str, Any]
    arxiv_data: Dict[str, Any]
    completed_agents: List[str]
    final_answer: str
    citations: List[Dict[str, str]]
    supervisor_decision: str
    feedback_score: int
    iteration_count: int
    feedback_reason: str
    rewritten_query: str
    all_intermediate_answers: List[Dict[str, str]]
    sub_questions: List[str]
    current_question_index: int
    question_answers: Dict[str, Dict[str, Any]]
    all_questions_answered: bool
    react_decision: str

# ---  Question Decomposer Agent (React Agent) ---
class QuestionDecomposerAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        query = state["query"]
        
        decomposer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a conservative question analysis agent. Your goal is to identify ONLY the main distinct questions, keeping it simple and focused.

        IMPORTANT RULES:
        1. If the query is a single topic/question, return it as ONE question only
        2. Only split if there are CLEARLY separate, unrelated topics connected by "and", "also", "plus" etc.
        3. Maximum 3 questions - prefer fewer questions over more and sometimes more depend on the user query
        4. Don't split broad topics into sub-questions
        5. Do not repeat the questions
        5. Return ONLY a JSON array, nothing else

        Examples:
        - "What is AI?" → ["What is AI?"]
        - "Explain quantum computing" → ["Explain quantum computing"]
        - "What is AI and machine learning?" → ["What is AI and machine learning?"]
        - "Tell me about Python and also data science careers" → ["Tell me about Python programming", "Tell me about data science careers"]
        - "What is blockchain and how does cryptocurrency work?" → ["What is blockchain and cryptocurrency?"]

        ALWAYS prefer combining related topics into single questions. Only split if topics are completely unrelated.

        Return only valid JSON array format: ["question1", "question2", ...]"""),
            ("human", f"Analyze this query and keep it simple with 2-3 questions maximum: {query}")
        ])
        
        try:
            chain = decomposer_prompt | llm | StrOutputParser()
            response = chain.invoke({"query": query})
            
            # Clean and parse the response
            response = response.strip()
            if response.startswith('```json'):
                response = response.replace('```json', '').replace('```', '').strip()
            elif response.startswith('```'):
                response = response.replace('```', '').strip()
            
            # Parse JSON
            try:
                sub_questions = json.loads(response)
                if not isinstance(sub_questions, list):
                    sub_questions = [query]  # Fallback to original query
            except json.JSONDecodeError:
                # Fallback parsing
                import re
                questions = re.findall(r'"([^"]+)"', response)
                sub_questions = questions if questions else [query]
            
            print(f"\n REACT AGENT - Question Analysis:")
            print(f"   Original Query: {query}")
            print(f"   Identified Questions: {len(sub_questions)}")
            for i, q in enumerate(sub_questions, 1):
                print(f"   {i}. {q}")
            
            return {
                "sub_questions": sub_questions,
                "current_question_index": 0,
                "question_answers": {},
                "all_questions_answered": False,
                "react_decision": f"Identified {len(sub_questions)} questions to answer"
            }
            
        except Exception as e:
            print(f" Question decomposer error: {str(e)} - treating as single question")
            return {
                "sub_questions": [query],
                "current_question_index": 0,
                "question_answers": {},
                "all_questions_answered": False,
                "react_decision": "Error in decomposition - treating as single question"
            }

# ---  React Supervisor Agent ---
class ReactSupervisorAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        sub_questions = state.get("sub_questions", [])
        current_index = state.get("current_question_index", 0)
        question_answers = state.get("question_answers", {})
        completed_agents = state.get("completed_agents", [])
        
        # Check if all questions are processed
        if current_index >= len(sub_questions):
            return {
                "all_questions_answered": True,
                "next_agent": "synthesis_agent",
                "react_decision": "All questions processed - moving to synthesis"
            }
        
        current_question = sub_questions[current_index]
        
        # Check if current question has been processed by all agents
        current_q_data = question_answers.get(current_question, {})
        agents_completed_for_current_q = current_q_data.get("completed_agents", [])
        
        available_agents = ["youtube_agent", "tavily_agent", "arxiv_agent"]
        remaining_agents = [agent for agent in available_agents if agent not in agents_completed_for_current_q]
        
        if not remaining_agents:
            # Current question is fully processed, move to next question
            new_index = current_index + 1
            
            if new_index >= len(sub_questions):
                return {
                    "current_question_index": new_index,
                    "all_questions_answered": True,
                    "next_agent": "synthesis_agent",
                    "react_decision": "All questions completed - moving to synthesis"
                }
            else:
                return {
                    "current_question_index": new_index,
                    "query": sub_questions[new_index],  # Update query to next question
                    "completed_agents": [],  # Reset for new question
                    "next_agent": "react_supervisor",
                    "react_decision": f"Moving to question {new_index + 1}: {sub_questions[new_index]}"
                }
        
        # Select next agent for current question
        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research supervisor. Choose the most appropriate agent for the given question.
            
            Available agents:
            - youtube_agent: Educational videos and tutorials
            - tavily_agent: Current web content and news  
            - arxiv_agent: Academic research papers
            
            Respond with ONLY the agent name."""),
            ("human", """
            Question: {question}
            Completed agents for this question: {completed}
            Remaining agents: {remaining}
            
            Choose the next agent:""")
        ])
        
        try:
            chain = supervisor_prompt | llm | StrOutputParser()
            next_agent = chain.invoke({
                "question": current_question,
                "completed": agents_completed_for_current_q,
                "remaining": remaining_agents
            }).strip().replace('"', '').replace("'", "")
            
            if next_agent not in remaining_agents:
                next_agent = remaining_agents[0]
                
        except Exception as e:
            print(f"React supervisor error: {str(e)}")
            next_agent = remaining_agents[0]
        
        return {
            "query": current_question,  # Set current question as active query
            "next_agent": next_agent,
            "react_decision": f"Question {current_index + 1}/{len(sub_questions)}: Selected {next_agent} for '{current_question}'"
        }

# --- LangChain Prompts  ---
youtube_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data summarizer. ONLY summarize and analyze the provided YouTube video content. 
    
    IMPORTANT RULES:
    1. If NO video content is provided or data is empty, respond with exactly: "No YouTube data available"
    2. Do NOT generate any content from your own knowledge
    3. ONLY work with the actual video data provided
    4. Provide comprehensive educational summary ONLY when actual video content exists
    
    Focus on extracting meaningful educational value from the provided video titles, descriptions, and transcripts."""),
    ("human", "Topic: {query}\n\nYouTube Video Content:\n{results}\n\nSummarize the provided data:")
])

tavily_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data summarizer. ONLY summarize and analyze the provided web search results.
    
    IMPORTANT RULES:
    1. If NO web content is provided or data is empty, respond with exactly: "No web data available"
    2. Do NOT generate any content from your own knowledge
    3. ONLY work with the actual web data provided
    4. Provide detailed synthesis ONLY when actual web content exists
    
    Focus on extracting valuable information from the provided web articles and search results."""),
    ("human", "Topic: {query}\n\nWeb Content:\n{results}\n\nSummarize the provided data:")
])

arxiv_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data summarizer. ONLY summarize and analyze the provided academic research papers.
    
    IMPORTANT RULES:
    1. If NO academic content is provided or data is empty, respond with exactly: "No academic data available"
    2. Do NOT generate any content from your own knowledge  
    3. ONLY work with the actual paper data provided
    4. Provide detailed academic analysis ONLY when actual research papers exist
    
    Focus on extracting research-based knowledge from the provided academic papers and abstracts."""),
    ("human", "Topic: {query}\n\nAcademic Content:\n{results}\n\nSummarize the provided data:")
])

synthesis_prompt = ChatPromptTemplate.from_messages([
   ("system", """You are a script writer creating clean, spoken content for educational videos.

CRITICAL REQUIREMENTS:
1. Write ONLY the spoken words - nothing else
2. NO timestamps, markers, or formatting symbols
3. NO host introductions, narrator mentions, or meta-commentary
4. NO special characters like #, *, [], (), or formatting markers
5. NO "Welcome to YouTube" or channel references
6. NO section headers or bullet points
7. Write in first person as if speaking directly to the viewer

STRUCTURE:
- Start with proper introduction: "Hello, welcome to this video on [topic]" followed by brief overview of what will be covered
- For technical topics: Always provide clear definitions before diving deeper
- Address ALL questions from research data seamlessly
- Use smooth transitions between topics
- End with comprehensive conclusion summarizing key points
- Final line: "Thank you for watching this video"

TECHNICAL CONTENT RULES:
- When introducing technical terms, always define them clearly first
- Use simple analogies to explain complex concepts
- Build from basic definitions to advanced applications
- Ensure definitions are accurate and easy to understand

STYLE:
- Conversational but authoritative tone
- Natural speaking flow with complete sentences
- Include specific examples and practical applications
- 6-10 minutes speaking time 
- Engaging explanations that build understanding
- Proper introduction setting context
- Strong conclusion tying everything together

OUTPUT: Pure spoken script text only - as if reading aloud to someone."""),
   ("human", """
All Research Data: {all_research_data}

Write a clean spoken script with proper introduction and conclusion, including technical definitions where needed (no formatting, no timestamps, no special characters):""")
])

feedback_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a quality evaluator for educational video scripts. 

Rate the script quality from 1-10 based on:
- Content accuracy and relevance to the topic
- Educational value and depth
- Clarity and logical structure
- Engagement level and audience appeal
- Completeness and coverage of the topic
- Script length appropriate for 6-10 minutes

If score is below 7, provide specific improvement suggestions and a rewritten query for better research.

IMPORTANT: You must respond with ONLY valid JSON in this exact format:
{{
    "score": 8,
    "reason": "The script provides good educational content with clear structure and engaging presentation. Minor improvements could be made in technical depth.",
    "rewritten_query": null
}}

Do not include any text before or after the JSON. Do not use markdown formatting."""),
    ("human", """
Original Query: {original_query}
Current Query: {current_query}
Iteration: {iteration}

Generated Script:
{script}

Evaluate this script and provide JSON response:""")
])

# ---  YouTube Agent  ---
class YouTubeAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        query = state["query"]
        completed = state.get("completed_agents", [])
        citations = state.get("citations", [])
        all_answers = state.get("all_intermediate_answers", [])
        
        # Handle question-specific data
        sub_questions = state.get("sub_questions", [])
        current_question_index = state.get("current_question_index", 0)
        question_answers = state.get("question_answers", {})
        
        if current_question_index < len(sub_questions):
            current_question = sub_questions[current_question_index]
            if current_question not in question_answers:
                question_answers[current_question] = {
                    "youtube_data": {},
                    "tavily_data": {},
                    "arxiv_data": {},
                    "completed_agents": [],
                    "citations": []
                }
        
        try:
            if youtube_tool.available:
                youtube_results = youtube_tool.run(query)
                
                # Check if there's actual data
                if not youtube_results:
                    summary = "No YouTube data available"
                    video_content = ""
                else:
                    # Process the enhanced results
                    video_content = ""
                    for video in youtube_results:
                        citations.append({
                            "source": "YouTube",
                            "title": video["title"],
                            "url": video["url"],
                            "type": "video",
                            "channel": video["channel"]
                        })
                        
                        # Compile video content for analysis
                        video_content += f"\n--- VIDEO: {video['title']} ---\n"
                        video_content += f"Channel: {video['channel']}\n"
                        video_content += f"URL: {video['url']}\n"
                        
                        if video["description"]:
                            desc = video["description"]
                            video_content += f"Description: {desc}\n"
                        
                        if video["captions"]:
                            # Clean and limit captions
                            import re
                            captions = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', video["captions"])
                            captions = re.sub(r'\n+', ' ', captions).strip()
                            captions = captions[:5000] + "..." if len(captions) > 5000 else captions
                            video_content += f"Transcript: {captions}\n"
                        
                        video_content += "\n" + "="*50 + "\n"
                    
                    # Only analyze if there's actual content
                    if video_content.strip():
                        chain = youtube_analysis_prompt | llm | StrOutputParser()
                        summary = chain.invoke({
                            "query": query,
                            "results": video_content
                        })
                    else:
                        summary = "No YouTube data available"
            else:
                youtube_results = []
                video_content = ""
                summary = "No YouTube data available"
            
            print(f"\n  YOUTUBE ANALYSIS:")
            print("-" * 50)
            print(summary)
            print("-" * 50)
            
            video_count = len(youtube_results)
            all_answers.append({
                "agent": "YouTube Agent",
                "query": query,
                "answer": summary,
                "sources_count": video_count
            })
                
            youtube_data = {
                "summary": summary,
                "raw_results": youtube_results,
                "videos_count": video_count,
                "content_extracted": bool(video_content)
            }
            
            # Store question-specific data
            if current_question_index < len(sub_questions):
                current_question = sub_questions[current_question_index]
                question_answers[current_question]["youtube_data"] = youtube_data
                question_answers[current_question]["completed_agents"].append("youtube_agent")
                # Only add citations from THIS specific query/question
                current_question_citations = []
                for video in youtube_results:
                    current_question_citations.append({
                        "source": "YouTube",
                        "title": video["title"],
                        "url": video["url"],
                        "type": "video",
                        "channel": video["channel"]
                    })
                question_answers[current_question]["citations"].extend(current_question_citations)
            
        except Exception as e:
            error_msg = f"YouTube analysis encountered an error: {str(e)}"
            print(f"\n YOUTUBE ANALYSIS:")
            print("-" * 50)
            print(error_msg)
            print("-" * 50)
            
            all_answers.append({
                "agent": "YouTube Agent",
                "query": query,
                "answer": "No YouTube data available",
                "sources_count": 0
            })
            
            youtube_data = {
                "summary": "No YouTube data available",
                "raw_results": [],
                "videos_count": 0,
                "content_extracted": False
            }
            
            # Store question-specific data
            if current_question_index < len(sub_questions):
                current_question = sub_questions[current_question_index]
                question_answers[current_question]["youtube_data"] = youtube_data
                question_answers[current_question]["completed_agents"].append("youtube_agent")
        
        return {
            "youtube_data": youtube_data,
            "citations": citations,
            "completed_agents": completed + ["youtube_agent"],
            "all_intermediate_answers": all_answers,
            "question_answers": question_answers,

        }

# --- Tavily Agent ---
class TavilyAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        query = state["query"]
        completed = state.get("completed_agents", [])
        citations = state.get("citations", [])
        all_answers = state.get("all_intermediate_answers", [])
        
        # Handle question-specific data
        sub_questions = state.get("sub_questions", [])
        current_question_index = state.get("current_question_index", 0)
        question_answers = state.get("question_answers", {})
        
        if current_question_index < len(sub_questions):
            current_question = sub_questions[current_question_index]
            if current_question not in question_answers:
                question_answers[current_question] = {
                    "youtube_data": {},
                    "tavily_data": {},
                    "arxiv_data": {},
                    "completed_agents": [],
                    "citations": []
                }
        
        try:
            if tavily_tool.available:
                tavily_results = tavily_tool.run(query)
                
                # Check if there's actual data
                if not tavily_results:
                    summary = "No web data available"
                    web_content = ""
                else:
                    # Compile full web content for analysis
                    web_content = ""
                    
                    # First, add AI summary if available (for analysis, not citations)
                    ai_summary_content = ""
                    for result in tavily_results:
                        if result.get("type") == "ai_summary" or result.get("title") == "Tavily AI Summary":
                            ai_summary_content = result.get("content", "")
                            if ai_summary_content:
                                web_content += f"\n--- TAVILY AI SUMMARY ---\n"
                                web_content += f"AI-Generated Summary: {ai_summary_content}\n"
                                web_content += "\n" + "="*60 + "\n"
                            break  # Only use first AI summary found
                    
                    # Then, process regular web articles
                    for result in tavily_results:
                        # Skip Tavily AI Summary entries for citations (but we used it above for analysis)
                        if result.get("type") == "ai_summary" or result.get("title") == "Tavily AI Summary":
                            continue
                            
                        # Add to citations (excludes AI summary)
                        citations.append({
                            "source": "Web Search",
                            "title": result.get("title", "Web Article"),
                            "url": result.get("url", ""),
                            "type": "web_article",
                            "published_date": result.get("published_date", ""),
                            "score": result.get("score", 0)
                        })
                        
                        # Compile web content for analysis (includes regular articles)
                        web_content += f"\n--- ARTICLE: {result.get('title', 'Unknown Title')} ---\n"
                        web_content += f"URL: {result.get('url', '')}\n"
                        web_content += f"Published: {result.get('published_date', 'Unknown date')}\n"
                        web_content += f"Relevance Score: {result.get('score', 0)}\n"
                        
                        # Add the full cleaned content
                        cleaned_content = result.get("cleaned_content", "")
                        if cleaned_content:
                            web_content += f"Content:\n{cleaned_content}\n"
                        else:
                            # Fallback to snippet if no full content
                            snippet = result.get("snippet", result.get("content", ""))
                            if snippet:
                                web_content += f"Summary: {snippet}\n"
                        
                        web_content += "\n" + "="*60 + "\n"
                    
                    # Only analyze if there's actual content
                    if web_content.strip():
                        chain = tavily_analysis_prompt | llm | StrOutputParser()
                        summary = chain.invoke({
                            "query": query,
                            "results": web_content
                        })
                    else:
                        summary = "No web data available"
            else:
                tavily_results = []
                web_content = ""
                summary = "No web data available"
            
            print(f"\n🌐 WEB SEARCH ANALYSIS:")
            print("-" * 50)
            print(summary)
            print("-" * 50)
            
            # Count only non-AI summary results (for accurate source counting)
            results_count = len([r for r in tavily_results if not (r.get("type") == "ai_summary" or r.get("title") == "Tavily AI Summary")])
            
            all_answers.append({
                "agent": "Web Search Agent",
                "query": query,
                "answer": summary,
                "sources_count": results_count
            })
                
            tavily_data = {
                "summary": summary,
                "raw_results": tavily_results,
                "results_count": results_count,
                "content_extracted": bool(web_content),
                "has_ai_summary": any(r.get("type") == "ai_summary" for r in tavily_results)
            }
            
            # Store question-specific data
            if current_question_index < len(sub_questions):
                current_question = sub_questions[current_question_index]
                question_answers[current_question]["tavily_data"] = tavily_data
                question_answers[current_question]["completed_agents"].append("tavily_agent")
                
                # Only add citations from THIS specific query/question (excludes AI summary)
                current_question_citations = []
                for result in tavily_results:
                    # Skip Tavily AI Summary entries for citations
                    if result.get("type") == "ai_summary" or result.get("title") == "Tavily AI Summary":
                        continue
                    current_question_citations.append({
                        "source": "Web Search",
                        "title": result.get("title", "Web Article"),
                        "url": result.get("url", ""),
                        "type": "web_article",
                        "published_date": result.get("published_date", ""),
                        "score": result.get("score", 0)
                    })
                question_answers[current_question]["citations"].extend(current_question_citations)
            
        except Exception as e:
            error_msg = f"Web search analysis encountered an error: {str(e)}"
            print(f"\n🌐 WEB SEARCH ANALYSIS:")
            print("-" * 50)
            print(error_msg)
            print("-" * 50)
            
            all_answers.append({
                "agent": "Web Search Agent",
                "query": query,
                "answer": "No web data available",
                "sources_count": 0
            })
            
            tavily_data = {
                "summary": "No web data available",
                "raw_results": [],
                "results_count": 0,
                "content_extracted": False,
                "has_ai_summary": False
            }
            
            # Store question-specific data
            if current_question_index < len(sub_questions):
                current_question = sub_questions[current_question_index]
                question_answers[current_question]["tavily_data"] = tavily_data
                question_answers[current_question]["completed_agents"].append("tavily_agent")
        
        return {
            "tavily_data": tavily_data,
            "citations": citations,
            "completed_agents": completed + ["tavily_agent"],
            "all_intermediate_answers": all_answers,
            "question_answers": question_answers
        }
# ---  ArXiv Agent ---
class ArxivAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        query = state["query"]
        completed = state.get("completed_agents", [])
        citations = state.get("citations", [])
        all_answers = state.get("all_intermediate_answers", [])
        
        # Handle question-specific data
        sub_questions = state.get("sub_questions", [])
        current_question_index = state.get("current_question_index", 0)
        question_answers = state.get("question_answers", {})
        
        if current_question_index < len(sub_questions):
            current_question = sub_questions[current_question_index]
            if current_question not in question_answers:
                question_answers[current_question] = {
                    "youtube_data": {},
                    "tavily_data": {},
                    "arxiv_data": {},
                    "completed_agents": [],
                    "citations": []
                }
        
        try:
            if arxiv_tool.available:
                arxiv_results = arxiv_tool.run(query)
                
                # Check if there's actual data
                if not arxiv_results:
                    summary = "No academic data available"
                    academic_content = ""
                else:
                    # Compile full academic content for analysis
                    academic_content = ""
                    for paper in arxiv_results:
                        citations.append({
                            "source": "arXiv",
                            "title": paper.get("title", "Unknown Paper"),
                            "url": paper.get("url", ""),
                            "type": "academic_paper",
                            "authors": paper.get("authors", ""),
                            "published": paper.get("published", ""),
                            "categories": paper.get("categories", "")
                        })
                        
                        # Compile academic content for analysis
                        academic_content += f"\n--- PAPER: {paper.get('title', 'Unknown Title')} ---\n"
                        academic_content += f"Authors: {paper.get('authors', 'Unknown')}\n"
                        academic_content += f"Published: {paper.get('published', 'Unknown date')}\n"
                        academic_content += f"Categories: {paper.get('categories', 'Unknown')}\n"
                        academic_content += f"URL: {paper.get('url', '')}\n"
                        academic_content += f"PDF: {paper.get('pdf_url', '')}\n"
                        
                        # Add the full abstract content
                        cleaned_abstract = paper.get("cleaned_abstract", "")
                        if cleaned_abstract:
                            academic_content += f"Abstract:\n{cleaned_abstract}\n"
                        else:
                            # Fallback to original abstract
                            original_abstract = paper.get("abstract", "")
                            if original_abstract:
                                academic_content += f"Abstract: {original_abstract[:3000]}...\n"
                        
                        academic_content += "\n" + "="*60 + "\n"
                    
                    # Only analyze if there's actual content
                    if academic_content.strip():
                        chain = arxiv_analysis_prompt | llm | StrOutputParser()
                        summary = chain.invoke({
                            "query": query,
                            "results": academic_content
                        })
                    else:
                        summary = "No academic data available"
            else:
                arxiv_results = []
                academic_content = ""
                summary = "No academic data available"
            
            print(f"\n ACADEMIC RESEARCH ANALYSIS:")
            print("-" * 50)
            print(summary)
            print("-" * 50)
            
            papers_count = len(arxiv_results)
            all_answers.append({
                "agent": "Academic Research Agent",
                "query": query,
                "answer": summary,
                "sources_count": papers_count
            })
                
            arxiv_data = {
                "summary": summary,
                "raw_results": arxiv_results,
                "papers_count": papers_count,
                "content_extracted": bool(academic_content),
                "has_abstracts": any(paper.get("cleaned_abstract") for paper in arxiv_results)
            }
            
            # Store question-specific data
            if current_question_index < len(sub_questions):
                current_question = sub_questions[current_question_index]
                question_answers[current_question]["arxiv_data"] = arxiv_data
                question_answers[current_question]["completed_agents"].append("arxiv_agent")
                # Only add citations from THIS specific query/question
                current_question_citations = []
                for paper in arxiv_results:
                    current_question_citations.append({
                        "source": "arXiv",
                        "title": paper.get("title", "Unknown Paper"),
                        "url": paper.get("url", ""),
                        "type": "academic_paper",
                        "authors": paper.get("authors", ""),
                        "published": paper.get("published", ""),
                        "categories": paper.get("categories", "")
                    })
                question_answers[current_question]["citations"].extend(current_question_citations)
            
        except Exception as e:
            error_msg = f"ArXiv analysis encountered an error: {str(e)}"
            print(f"\n ACADEMIC RESEARCH ANALYSIS:")
            print("-" * 50)
            print(error_msg)
            print("-" * 50)
            
            all_answers.append({
                "agent": "Academic Research Agent",
                "query": query,
                "answer": "No academic data available",
                "sources_count": 0
            })
            
            arxiv_data = {
                "summary": "No academic data available",
                "raw_results": [],
                "papers_count": 0,
                "content_extracted": False,
                "has_abstracts": False
            }
            
            # Store question-specific data
            if current_question_index < len(sub_questions):
                current_question = sub_questions[current_question_index]
                question_answers[current_question]["arxiv_data"] = arxiv_data
                question_answers[current_question]["completed_agents"].append("arxiv_agent")
        
        return {
            "arxiv_data": arxiv_data,
            "citations": citations,
            "completed_agents": completed + ["arxiv_agent"],
            "all_intermediate_answers": all_answers,
            "question_answers": question_answers
        }

# --- Answer Completeness Checker Agent ---
class AnswerCompletenessChecker(Runnable):
    def invoke(self, state: VideoState, config=None):
        sub_questions = state.get("sub_questions", [])
        question_answers = state.get("question_answers", {})
        
        # Check if all questions have at least some meaningful research data
        missing_questions = []
        answered_questions = []
        
        for question in sub_questions:
            if question not in question_answers:
                missing_questions.append(question)
            else:
                # Check if question has any meaningful data
                q_data = question_answers[question]
                youtube_summary = q_data.get('youtube_data', {}).get('summary', 'No YouTube data available')
                tavily_summary = q_data.get('tavily_data', {}).get('summary', 'No web data available')
                arxiv_summary = q_data.get('arxiv_data', {}).get('summary', 'No academic data available')
                
                # Check if at least one source has meaningful data
                has_youtube = youtube_summary != 'No YouTube data available'
                has_web = tavily_summary != 'No web data available'
                has_academic = arxiv_summary != 'No academic data available'
                
                if has_youtube or has_web or has_academic:
                    answered_questions.append(question)
                else:
                    missing_questions.append(question)
        
        all_answered = len(missing_questions) == 0
        
        print(f"\n COMPLETENESS CHECK:")
        print(f"   Total Questions: {len(sub_questions)}")
        print(f"   Answered Questions: {len(answered_questions)}")
        print(f"   Missing Questions: {len(missing_questions)}")
        
        if answered_questions:
            print(f"  Questions with data:")
            for q in answered_questions:
                print(f"      - {q}")
        
        if missing_questions:
            print(f"  Questions needing more research:")
            for q in missing_questions:
                print(f"      - {q}")
        
        if all_answered:
            print(f"  All questions have sufficient research data!")
            return {
                "all_questions_answered": True,
                "next_agent": "synthesis_agent",
                "react_decision": "All questions have sufficient research data"
            }
        else:
            print(f"  Improving missing questions for better research...")
            
            # Create improved questions using LLM
            improved_questions = []
            improvement_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research query optimizer. Your task is to improve questions that failed to find research data by making them more specific, searchable, and research-friendly.

                RULES for improvement:
                1. Make questions more specific and detailed
                2. Add relevant keywords that would help in searches
                3. Break down complex questions into clearer components
                4. Add context that would help research agents find better sources
                5. Keep the core meaning but make it more searchable
                6. Return ONLY the improved question, nothing else

                Examples:
                - "What is AI?" → "What is artificial intelligence and how does AI technology work in modern applications?"
                - "Blockchain technology" → "How does blockchain technology work and what are its practical applications in cryptocurrency and data security?"
                - "Machine learning basics" → "What are the fundamental concepts of machine learning algorithms and how do they process data to make predictions?"
                """),
                                ("human", f"Improve this question for better research results: {'{question}'}")
                            ])
            improved_questions = []     
            for missing_q in missing_questions:
                try:
                    chain = improvement_prompt | llm | StrOutputParser()
                    improved_q = chain.invoke({"question": missing_q}).strip()
                    
                    # Clean up any formatting
                    improved_q = re.sub(r'^["\']|["\']$', '', improved_q)  # Remove quotes
                    improved_q = re.sub(r'^\d+\.\s*', '', improved_q)  # Remove numbering
                    
                    improved_questions.append(improved_q)
                    print(f"  Improved: '{missing_q}' → '{improved_q}'")
                    
                except Exception as e:
                    print(f"  Failed to improve '{missing_q}': {str(e)}")
                    improved_questions.append(missing_q)  # Fallback to original
            
            # Update sub_questions with improved versions
            updated_sub_questions = []
            for original_q in sub_questions:
                if original_q in missing_questions:
                    # Find the improved version
                    idx = missing_questions.index(original_q)
                    updated_sub_questions.append(improved_questions[idx])
                else:
                    updated_sub_questions.append(original_q)
            
            return {
                "all_questions_answered": False,
                "next_agent": "react_supervisor",
                "sub_questions": updated_sub_questions,  # Update with improved questions
                "current_question_index": 0,  # Reset to start processing improved questions
                "react_decision": f"Improved and continuing research for questions: {improved_questions}"
            }

# --- Synthesis Agent ---
class SynthesisAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        query = state["original_query"]
        all_answers = state.get("all_intermediate_answers", [])
        question_answers = state.get("question_answers", {})
        
        # Compile all research data from all questions
        all_research_data = ""
        
        for question, data in question_answers.items():
            all_research_data += f"\n{'='*80}\n"
            all_research_data += f"QUESTION: {question}\n"
            all_research_data += f"{'='*80}\n"
            
            youtube_summary = data.get('youtube_data', {}).get('summary', 'No YouTube data available')
            tavily_summary = data.get('tavily_data', {}).get('summary', 'No web data available')
            arxiv_summary = data.get('arxiv_data', {}).get('summary', 'No academic data available')
            
            all_research_data += f"\n YOUTUBE RESEARCH:\n{youtube_summary}\n"
            all_research_data += f"\n WEB RESEARCH:\n{tavily_summary}\n"
            all_research_data += f"\n ACADEMIC RESEARCH:\n{arxiv_summary}\n"
        
        try:
            chain = synthesis_prompt | llm | StrOutputParser()
            
            script = chain.invoke({
                "all_research_data": all_research_data
            })
            
            # Clean up the script
            script = re.sub(r'^(Here\'s|Here is|I\'ll create).*?:\s*', '', script, flags=re.IGNORECASE)
            script = re.sub(r'^\*\*.*?\*\*\s*', '', script, flags=re.MULTILINE)
            script = re.sub(r'^\d+\.\s*', '', script, flags=re.MULTILINE)
            script = re.sub(r'^[\*\-•]\s*', '', script, flags=re.MULTILINE)
            script = re.sub(r'\[.*?\]', '', script)
            script = re.sub(r'\*\*.*?\*\*', '', script)
            script = re.sub(r'\n{3,}', '\n\n', script)
            script = script.strip()
            
            print("\n" + "="*80)
            print(" FINAL VIDEO SCRIPT")
            print("="*80)
            print(script)
            print("="*80)
            
            all_answers.append({
                "agent": "Synthesis Agent (FINAL)",
                "query": query,
                "answer": script,
                "sources_count": "N/A"
            })
            
            return {
                "final_answer": script,
                "next_agent": "feedback_agent",
                "all_intermediate_answers": all_answers
            }
            
        except Exception as e:
            error_script = f"Error generating script: {str(e)}. However, based on the topic '{query}', a basic script structure would include an introduction, main content, and conclusion."
            
            print("\n" + "="*80)
            print(" FINAL VIDEO SCRIPT (ERROR FALLBACK)")
            print("="*80)
            print(error_script)
            print("="*80)
            
            all_answers.append({
                "agent": "Synthesis Agent (ERROR)",
                "query": query,
                "answer": error_script,
                "sources_count": "N/A"
            })
            
            return {
                "final_answer": error_script,
                "next_agent": "feedback_agent",
                "all_intermediate_answers": all_answers
            }

# --- Feedback Agent  ---
class FeedbackAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        iteration_count = state.get("iteration_count", 0)
        all_answers = state.get("all_intermediate_answers", [])
        
        if iteration_count >= 3:
            print(f"\n Maximum iterations (3) reached - accepting current result")
            return {
                "feedback_score": 7,
                "feedback_reason": "Maximum iterations reached - accepting current result",
                "next_agent": "FINISH",
                "all_intermediate_answers": all_answers
            }
        
        try:
            chain = feedback_prompt | llm | StrOutputParser()
            
            feedback_response = chain.invoke({
                "original_query": state["original_query"],
                "current_query": state["query"],
                "script": state["final_answer"],
                "iteration": iteration_count + 1
            })
            
            print(f"\n Raw feedback response: {repr(feedback_response)}")
            
            feedback_response = feedback_response.strip()
            
            if feedback_response.startswith('```json'):
                feedback_response = feedback_response.replace('```json', '').replace('```', '').strip()
            elif feedback_response.startswith('```'):
                feedback_response = feedback_response.replace('```', '').strip()
            
            if not feedback_response:
                print(" Empty response from LLM - using default feedback")
                return {
                    "feedback_score": 7,
                    "feedback_reason": "Empty feedback response - accepting current result",
                    "next_agent": "FINISH",
                    "all_intermediate_answers": all_answers
                }
            
            feedback_data = None
            
            try:
                feedback_data = json.loads(feedback_response)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', feedback_response, re.DOTALL)
                if json_match:
                    try:
                        feedback_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            
            if not feedback_data:
                print(" JSON parsing failed - attempting manual parsing")
                
                score_match = re.search(r'"?score"?\s*:\s*(\d+)', feedback_response, re.IGNORECASE)
                reason_match = re.search(r'"?reason"?\s*:\s*"([^"]*)"', feedback_response, re.IGNORECASE)
                query_match = re.search(r'"?rewritten_query"?\s*:\s*"([^"]*)"', feedback_response, re.IGNORECASE)
                
                score = int(score_match.group(1)) if score_match else 7
                reason = reason_match.group(1) if reason_match else "Manual parsing - feedback format issue"
                rewritten_query = query_match.group(1) if query_match else None
                
                feedback_data = {
                    "score": score,
                    "reason": reason,
                    "rewritten_query": rewritten_query
                }
            
            score = feedback_data.get("score", 7)
            reason = feedback_data.get("reason", "No feedback provided")
            rewritten_query = feedback_data.get("rewritten_query")
            
            if not isinstance(score, int) or score < 1 or score > 10:
                print(f" Invalid score {score} - defaulting to 7")
                score = 7
            
            print(f"\n Feedback Analysis:")
            print(f"   Score: {score}/10")
            print(f"   Reason: {reason}")
            
            if score >= 7:
                print(f" Quality approved! Proceeding to save script.")
                return {
                    "feedback_score": score,
                    "feedback_reason": reason,
                    "next_agent": "FINISH",
                    "all_intermediate_answers": all_answers
                }
            else:
                new_iteration = iteration_count + 1
                print(f" Score below 7 - Starting iteration {new_iteration}")
                if rewritten_query:
                    print(f"   New query: {rewritten_query}")
                
                return {
                    "feedback_score": score,
                    "feedback_reason": reason,
                    "rewritten_query": rewritten_query or state["query"],
                    "query": rewritten_query or state["query"],
                    "completed_agents": [],
                    "youtube_data": {},
                    "tavily_data": {},
                    "arxiv_data": {},
                    "citations": [],
                    "iteration_count": new_iteration,
                    "next_agent": "question_decomposer",
                    "all_intermediate_answers": all_answers,
                    "sub_questions": [],
                    "current_question_index": 0,
                    "question_answers": {},
                    "all_questions_answered": False
                }
                
        except Exception as e:
            print(f" Feedback system error: {str(e)} - accepting current result")
            print(f" Full error traceback:")
            import traceback
            traceback.print_exc()
            
            return {
                "feedback_score": 7,
                "feedback_reason": f"Feedback system error - accepting result: {str(e)}",
                "next_agent": "FINISH",
                "all_intermediate_answers": all_answers
            }

# --- Router Function ---
def router(state: VideoState) -> Literal["question_decomposer", "react_supervisor", "completeness_checker", "youtube_agent", "tavily_agent", "arxiv_agent", "synthesis_agent", "feedback_agent", "__end__"]:
    next_agent = state.get("next_agent", "")
    
    if next_agent == "FINISH" or not next_agent:
        return "__end__"
    elif next_agent in ["question_decomposer", "react_supervisor", "completeness_checker", "youtube_agent", "tavily_agent", "arxiv_agent", "synthesis_agent", "feedback_agent"]:
        return next_agent
    else:
        return "question_decomposer"

# --- Graph Builder ---
def build_research_graph():
    builder = StateGraph(VideoState)
    
    # Add all nodes
    builder.add_node("question_decomposer", QuestionDecomposerAgent())
    builder.add_node("react_supervisor", ReactSupervisorAgent())
    builder.add_node("completeness_checker", AnswerCompletenessChecker())
    builder.add_node("youtube_agent", YouTubeAgent())
    builder.add_node("tavily_agent", TavilyAgent())
    builder.add_node("arxiv_agent", ArxivAgent())
    builder.add_node("synthesis_agent", SynthesisAgent())
    builder.add_node("feedback_agent", FeedbackAgent())
    
    # Set entry point
    builder.add_edge(START, "question_decomposer")
    
    # Add conditional edges
    builder.add_edge("question_decomposer", "react_supervisor")
    builder.add_conditional_edges("react_supervisor", router)
    builder.add_edge("youtube_agent", "react_supervisor")
    builder.add_edge("tavily_agent", "react_supervisor")
    builder.add_edge("arxiv_agent", "react_supervisor")
    builder.add_conditional_edges("completeness_checker", router)
    builder.add_conditional_edges("synthesis_agent", router)
    builder.add_conditional_edges("feedback_agent", router)
    
    return builder.compile()


def visualize_workflow(graph):
    """Display the workflow graph using LangGraph's compiled structure"""
    try:
        print("\n COMPILED LANGGRAPH WORKFLOW:")
        print("="*50)
        
        # Get the actual compiled graph
        compiled_graph = graph.get_graph()
        
        # Display the Mermaid representation
        mermaid_diagram = compiled_graph.draw_mermaid()
        print(" WORKFLOW STRUCTURE:")
        print("="*50)
        
        # Also save as PNG file
        try:
            png_bytes = compiled_graph.draw_mermaid_png()
            with open("research_workflow.png", "wb") as f:
                f.write(png_bytes)
            print("Workflow diagram saved as 'research_workflow.png'")
        except:
            pass
            
        print("LangGraph workflow displayed")
        return True
        
    except Exception as e:
        print(f" Error displaying LangGraph workflow: {str(e)}")
        return False

# --- Citation Formatter---
def format_citations(citations: List[Dict[str, str]], question_answers: Dict[str, Dict]) -> str:
    if not citations:
        return " SOURCES:\nNo external sources were cited during research."
    
    formatted = " RESEARCH SOURCES\n" + "=" * 60 + "\n"
    
    # Format by questions if available
    if question_answers:
        for i, (question, data) in enumerate(question_answers.items(), 1):
            formatted += f"\n QUESTION {i}: {question}\n" + "-" * 60 + "\n"
            
            question_citations = data.get("citations", [])
            
            if not question_citations:
                formatted += "   No sources found for this question.\n"
                continue
            
            youtube_sources = [c for c in question_citations if c["source"] == "YouTube"]
            web_sources = [c for c in question_citations if c["source"] == "Web Search"]
            arxiv_sources = [c for c in question_citations if c["source"] == "arXiv"]
            
            if youtube_sources:
                formatted += "\n    YOUTUBE VIDEOS:\n"
                for j, source in enumerate(youtube_sources, 1):
                    formatted += f"   {j}. {source['title']}\n      🔗 {source['url']}\n\n"
            
            if web_sources:
                formatted += "\n    WEB ARTICLES:\n"
                for j, source in enumerate(web_sources, 1):
                    formatted += f"   {j}. {source['title']}\n      🔗 {source['url']}\n\n"
            
            if arxiv_sources:
                formatted += "\n    ACADEMIC PAPERS:\n"
                for j, source in enumerate(arxiv_sources, 1):
                    formatted += f"   {j}. {source['title']}\n      🔗 {source['url']}\n\n"
    else:
        # Fallback to original format for single question
        youtube_sources = [c for c in citations if c["source"] == "YouTube"]
        web_sources = [c for c in citations if c["source"] == "Web Search"]
        arxiv_sources = [c for c in citations if c["source"] == "arXiv"]
        
        if youtube_sources:
            formatted += "\n YOUTUBE VIDEOS\n" + "-" * 30 + "\n"
            for i, source in enumerate(youtube_sources, 1):
                formatted += f"{i}. {source['title']}\n   🔗 {source['url']}\n\n"
        
        if web_sources:
            formatted += "\n WEB ARTICLES\n" + "-" * 30 + "\n"
            for i, source in enumerate(web_sources, 1):
                formatted += f"{i}. {source['title']}\n   🔗 {source['url']}\n\n"
        
        if arxiv_sources:
            formatted += "\n ACADEMIC PAPERS\n" + "-" * 30 + "\n"
            for i, source in enumerate(arxiv_sources, 1):
                formatted += f"{i}. {source['title']}\n   🔗 {source['url']}\n\n"
    
    total_sources = len(citations)
    youtube_count = len([c for c in citations if c["source"] == "YouTube"])
    web_count = len([c for c in citations if c["source"] == "Web Search"])
    arxiv_count = len([c for c in citations if c["source"] == "arXiv"])
    
    formatted += f"\n TOTAL SOURCES: {total_sources}\n"
    formatted += f"   • YouTube Videos: {youtube_count}\n"
    formatted += f"   • Web Articles: {web_count}\n"
    formatted += f"   • Academic Papers: {arxiv_count}\n"
    
    return formatted

# --- All Answers Formatter (Updated) ---
def format_all_answers(all_answers: List[Dict[str, str]], feedback_info: Dict[str, Any]) -> str:
    if not all_answers:
        return "No intermediate answers were generated."
    
    formatted = "🔍 ALL RESEARCH FINDINGS\n" + "=" * 60 + "\n\n"
    
    # Add feedback information at the top
    formatted += f" QUALITY METRICS\n"
    formatted += f"Final Feedback Score: {feedback_info.get('final_score', 'N/A')}/10\n"
    formatted += f"Total Iterations: {feedback_info.get('total_iterations', 'N/A')}\n"
    formatted += f"Final Feedback: {feedback_info.get('final_reason', 'No feedback available')}\n"
    formatted += "\n" + "="*60 + "\n\n"
    
    for i, answer in enumerate(all_answers, 1):
        agent = answer.get("agent", "Unknown Agent")
        query = answer.get("query", "No query")
        content = answer.get("answer", "No answer")
        sources = answer.get("sources_count", 0)
        
        formatted += f" STEP {i}: {agent.upper()}\n"
        formatted += f"Query: {query}\n"
        formatted += f"Sources Found: {sources}\n"
        formatted += "-" * 50 + "\n"
        formatted += f"{content}\n"
        formatted += "\n" + "="*60 + "\n\n"
    
    return formatted

def save_script_to_file(script: str, feedback_info: Dict[str, Any], filename: str = "video_script.txt") -> bool:
    try:
        # Clean the script of any unwanted formatting
        clean_script = re.sub(r'\*\*.*?\*\*', '', script)  # Remove bold markers
        clean_script = re.sub(r'^\d+\.\s*', '', clean_script, flags=re.MULTILINE)  # Remove numbered lists
        clean_script = re.sub(r'^[\*\-•]\s*', '', clean_script, flags=re.MULTILINE)  # Remove bullet points
        clean_script = re.sub(r'\[.*?\]', '', clean_script)  # Remove timestamps
        clean_script = re.sub(r'\n{3,}', '\n\n', clean_script)  # Normalize line breaks
        clean_script = clean_script.strip()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(clean_script)
        return True
    except Exception as e:
        print(f" Error saving script: {str(e)}")
        return False

def save_detailed_report(all_answers: List[Dict[str, str]], citations: List[Dict[str, str]], 
                        feedback_info: Dict[str, Any], script: str, question_answers: Dict[str, Dict] = None, 
                        filename: str = "detailed_report.txt") -> bool:
    try:
        content = f"DETAILED RESEARCH REPORT\n"
        content += f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "="*80 + "\n\n"
        
        # Add all research findings
        content += format_all_answers(all_answers, feedback_info)
        content += "\n\n"
        
        # Add final script
        content += " FINAL VIDEO SCRIPT\n" + "="*60 + "\n"
        content += script + "\n\n"
        
        # Add citations
        content += format_citations(citations, question_answers or {})
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f" Error saving detailed report: {str(e)}")
        return False

# ---  Main Execution ---
def main():
    print("\n" + "="*80)
    print(" AI VIDEO SCRIPT RESEARCH AGENT WITH REACT INTELLIGENCE")
    print("="*80)
    print("This AI agent will analyze your query for multiple questions,")
    print("research each question thoroughly, and generate a comprehensive")
    print("6-10 minute video script that addresses all topics.\n")
    
    # Get user input
    while True:
        user_query = input(" Enter your video topic or question(s): ").strip()
        if user_query:
            break
        print(" Please enter a valid topic.")
    
    print(f"\n User Query: '{user_query}'")
    print(f" Starting intelligent question analysis and research...\n")
    
    # Build and run the graph
    graph = build_research_graph()

    # Add workflow visualization
    print("\n DISPLAYING RESEARCH WORKFLOW:")
    visualize_workflow(graph)
    
    initial_state: VideoState = {
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
        # Execute the graph
        config = {"recursion_limit": 100}
        final_state = graph.invoke(initial_state, config=config)
        
        # Extract results
        script = final_state.get("final_answer", "No script generated")
        citations = final_state.get("citations", [])
        all_answers = final_state.get("all_intermediate_answers", [])
        feedback_score = final_state.get("feedback_score", 0)
        feedback_reason = final_state.get("feedback_reason", "No feedback")
        iteration_count = final_state.get("iteration_count", 0) + 1
        sub_questions = final_state.get("sub_questions", [])
        question_answers = final_state.get("question_answers", {})
        
        feedback_info = {
            "final_score": feedback_score,
            "total_iterations": iteration_count,
            "final_reason": feedback_reason
        }
        
        print("\n" + "="*80)
        print(" RESEARCH COMPLETE!")
        print("="*80)
        
        # Display React Agent summary
        print(f"\n REACT AGENT ANALYSIS:")
        print(f"   • Questions Identified: {len(sub_questions)}")
        for i, q in enumerate(sub_questions, 1):
            print(f"     {i}. {q}")
        print(f"   • Questions Researched: {len(question_answers)}")
        print(f"   • All Questions Addressed: {' Yes' if len(question_answers) == len(sub_questions) else '❌ No'}")
        
        # Display summary
        print(f"\n RESEARCH SUMMARY:")
        print(f"   • Final Quality Score: {feedback_score}/10")
        print(f"   • Total Iterations: {iteration_count}")
        print(f"   • Sources Found: {len(citations)}")
        print(f"   • Research Steps: {len(all_answers)}")
        
        # Save files
        print(f"\n SAVING FILES:")
        
        script_saved = save_script_to_file(script, feedback_info, "video_script.txt")
        if script_saved:
            print("    Video script saved to 'video_script.txt'")
        else:
            print("    Failed to save video script")
        
        report_saved = save_detailed_report(all_answers, citations, feedback_info, script, question_answers, "research_report.txt")
        if report_saved:
            print("    Detailed report saved to 'research_report.txt'")
        else:
            print("    Failed to save detailed report")
        
        # Display question-specific research summary
        print(f"\n QUESTION-SPECIFIC RESEARCH SUMMARY:")
        for question, data in question_answers.items():
            print(f"\n    Question: {question}")
            youtube_data = data.get('youtube_data', {})
            tavily_data = data.get('tavily_data', {})
            arxiv_data = data.get('arxiv_data', {})
            
            print(f"  YouTube: {youtube_data.get('videos_count', 0)} videos")
            print(f"  Web: {tavily_data.get('results_count', 0)} articles")  
            print(f"  Academic: {arxiv_data.get('papers_count', 0)} papers")
        
        # Display citations
        # Display citations
        print(f"\n{format_citations(citations, question_answers)}")
        
        print(f"\n SUCCESS! Your comprehensive multi-question video script has been generated!")
        print(f" Check the current directory for the generated files.")
        print(f" The React Agent successfully identified and researched {len(sub_questions)} question(s) in your query.")
        
    except KeyboardInterrupt:
        print(f"\n\n Process interrupted by user.")
        print(f" Research stopped before completion.")
        
    except Exception as e:
        print(f"\n CRITICAL ERROR: {str(e)}")
        print(f" Please check your API keys and internet connection.")
        import traceback
        print(f"\nFull error details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()