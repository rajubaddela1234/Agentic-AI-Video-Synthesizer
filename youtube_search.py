###Research_system.py
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Literal, List, Dict, Any
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import sys
from dotenv import load_dotenv
import json
import re
import requests
from urllib.parse import quote

load_dotenv()

# --- API Keys with validation ---
tavily_api_key = os.getenv("TAVILY_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Check which APIs are available
apis_available = {
    "groq": bool(groq_api_key),
    "tavily": bool(tavily_api_key),
    "youtube": bool(youtube_api_key)
}

print("🔑 API Status Check:")
for api, status in apis_available.items():
    print(f"   • {api.upper()}: {'✅ Available' if status else '❌ Missing'}")

if not groq_api_key:
    print("❌ GROQ API key is required for the LLM. Please set GROQ_API_KEY in your environment.")
    sys.exit(1)

# --- Initialize LangChain LLM ---
try:
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # Using a more common model
        api_key="REMOVED",
        temperature=0.3
    )
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize LLM: {str(e)}")
    sys.exit(1)

# --- Custom Tavily Search Tool ---
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
                "include_answer": True,      # Include Tavily's summary
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
                    enhanced_result["cleaned_content"] = cleaned_content[:3000] + "..." if len(cleaned_content) > 3000 else cleaned_content
                else:
                    enhanced_result["cleaned_content"] = ""
                
                enhanced_results.append(enhanced_result)
            
            # Add Tavily's answer if available
            tavily_answer = data.get("answer", "")
            if tavily_answer:
                enhanced_results.insert(0, {
                    "title": "Tavily AI Summary",
                    "url": "https://tavily.com",
                    "content": tavily_answer,
                    "raw_content": tavily_answer,
                    "published_date": "",
                    "score": 1.0,
                    "snippet": tavily_answer[:200] + "..." if len(tavily_answer) > 200 else tavily_answer,
                    "cleaned_content": tavily_answer,
                    "type": "ai_summary"
                })
            
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

# --- Simple YouTube Search ---
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
                            "tfmt": "srt"  # or "vtt"
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

# --- Simple ArXiv Search ---
class SimpleArxivSearch:
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.available = True  # ArXiv API is free
    
    def get_paper_details(self, paper_id: str) -> Dict[str, str]:
        """Get detailed paper information including abstract and full content if available"""
        details = {"abstract": "", "full_text": "", "authors": "", "categories": ""}
        
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
                    
                    # Extract categories
                    categories = []
                    for category in entry.findall('.//{http://arxiv.org/schemas/atom}primary_category'):
                        term = category.get('term')
                        if term:
                            categories.append(term)
                    details["categories"] = ", ".join(categories)
                    
                    break
                    
            except ET.ParseError:
                # Fallback to regex parsing if XML parsing fails
                abstract_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
                if abstract_match:
                    details["abstract"] = abstract_match.group(1).strip()
            
            # Try to get PDF content (this would require additional libraries like PyPDF2)
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
                        
                        # Extract categories
                        categories = []
                        for category in entry.findall('.//{http://arxiv.org/schemas/atom}primary_category'):
                            term = category.get('term')
                            if term:
                                categories.append(term)
                        
                        # Clean the abstract content
                        cleaned_abstract = self._clean_paper_content(abstract)
                        
                        results.append({
                            "title": title,
                            "url": paper_url,
                            "paper_id": paper_id,
                            "authors": ", ".join(authors),
                            "published": published,
                            "categories": ", ".join(categories),
                            "abstract": abstract,
                            "cleaned_abstract": cleaned_abstract,
                            "full_content": cleaned_abstract,  # Use abstract as full content
                            "pdf_url": paper_url.replace('/abs/', '/pdf/') + '.pdf' if paper_url else ""
                        })
                
            except ET.ParseError:
                # Fallback to regex parsing
                titles = re.findall(r'<title>(.*?)</title>', content, re.DOTALL)
                links = re.findall(r'<id>(http://arxiv\.org/abs/.*?)</id>', content)
                abstracts = re.findall(r'<summary>(.*?)</summary>', content, re.DOTALL)
                
                for i, (title, link) in enumerate(zip(titles[1:6], links[:5])):  # Skip first title
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

# --- Enhanced State Type ---
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

# --- LangChain Prompts ---
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research supervisor coordinating multiple agents. 
    Choose the next most appropriate agent based on the query and research strategy.
    
    Available agents:
    - youtube_agent: Educational videos and tutorials
    - tavily_agent: Current web content and news  
    - arxiv_agent: Academic research papers
    
    Respond with ONLY the agent name."""),
    ("human", """
    Topic: {query}
    Completed agents: {completed}
    Remaining agents: {remaining}
    
    Choose the next agent:""")
])

youtube_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze YouTube video content including titles, descriptions, and transcripts. 
    Provide a comprehensive educational summary that synthesizes the key information, insights, 
    and practical knowledge from the video content. Focus on extracting meaningful educational 
    value and actionable information. If no video content is available, provide general knowledge about the topic."""),
    ("human", "Topic: {query}\n\nYouTube Video Content:\n{results}\n\nProvide a detailed educational analysis:")
])

tavily_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze comprehensive web search results including full article content, AI summaries, and current information. 
    Provide a detailed synthesis of the current state of knowledge, recent developments, trends, and key insights 
    from the web content. Focus on extracting the most valuable and up-to-date information that would be useful 
    for creating educational content. Include specific facts, statistics, examples, and real-world applications 
    when available. If no web content is available, provide general knowledge about the topic."""),
    ("human", "Topic: {query}\n\nComprehensive Web Content:\n{results}\n\nProvide a detailed analysis and synthesis:")
])

arxiv_analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze comprehensive academic research papers including titles, authors, abstracts, and research content. 
    Provide a detailed synthesis of the current academic research, methodologies, key findings, theoretical frameworks, 
    and scientific insights from the academic literature. Focus on extracting the most valuable research-based knowledge, 
    experimental results, theoretical contributions, and practical applications from the academic papers. 
    Include specific research findings, statistical results, methodological approaches, and scholarly insights when available. 
    If no academic content is available, provide general scholarly knowledge about the topic based on established research."""),
    ("human", "Topic: {query}\n\nComprehensive Academic Content:\n{results}\n\nProvide a detailed academic analysis and research synthesis:")
])

synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Create a comprehensive, well-structured educational video script for 6-10 minutes of speaking time.

    Requirements:
    1. Engaging hook that captures attention immediately
    2. Clear structure with logical flow and smooth transitions
    3. In-depth explanations with specific examples and case studies
    4. Practical applications and real-world relevance
    5. Conversational but authoritative tone
    6. 900-1500 words for 6-10 minutes of speaking time
    7. Strong conclusion with key takeaways
    
    Output only the clean script text - no formatting markers, timestamps, or meta-commentary."""),
    ("human", """
    Topic: {query}
    
    YouTube Educational Content: {youtube_data}
    
    Current Web Information: {tavily_data}
    
    Academic Research Insights: {arxiv_data}
    
    Create a comprehensive 6-10 minute educational video script:""")
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


# --- Supervisor Agent ---
class SupervisorAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        completed = state.get("completed_agents", [])
        available_agents = ["youtube_agent", "tavily_agent", "arxiv_agent"]
        remaining_agents = [agent for agent in available_agents if agent not in completed]
        
        if not remaining_agents:
            return {
                "next_agent": "synthesis_agent",
                "supervisor_decision": "All research completed. Moving to synthesis."
            }
        
        chain = supervisor_prompt | llm | StrOutputParser()
        
        try:
            next_agent = chain.invoke({
                "query": state["query"],
                "completed": completed,
                "remaining": remaining_agents
            }).strip().replace('"', '').replace("'", "")
            
            if next_agent not in remaining_agents:
                next_agent = remaining_agents[0]
                
        except Exception as e:
            print(f"Supervisor error: {str(e)}")
            next_agent = remaining_agents[0]
        
        return {
            "next_agent": next_agent,
            "supervisor_decision": f"Selected {next_agent} for research on '{state['query']}'"
        }

# --- YouTube Agent ---
class YouTubeAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        query = state["query"]
        completed = state.get("completed_agents", [])
        citations = state.get("citations", [])
        all_answers = state.get("all_intermediate_answers", [])
        
        try:
            if youtube_tool.available:
                youtube_results = youtube_tool.run(query)
                
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
                        # Limit description length to avoid token limits
                        desc = video["description"][:1000] + "..." if len(video["description"]) > 1000 else video["description"]
                        video_content += f"Description: {desc}\n"
                    
                    if video["captions"]:
                        # Clean and limit captions
                        import re
                        captions = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', video["captions"])
                        captions = re.sub(r'\n+', ' ', captions).strip()
                        captions = captions[:2000] + "..." if len(captions) > 2000 else captions
                        video_content += f"Transcript: {captions}\n"
                    
                    video_content += "\n" + "="*50 + "\n"
            else:
                youtube_results = []
                video_content = ""
            
            # Pass the actual video content to the LLM
            chain = youtube_analysis_prompt | llm | StrOutputParser()
            summary = chain.invoke({
                "query": query,
                "results": video_content or f"No YouTube API available. Analyzing topic: {query}"
            })
            
            print(f"\n🎥 YOUTUBE ANALYSIS:")
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
                "raw_results": youtube_results,  # Now contains full video data
                "videos_count": video_count,
                "content_extracted": bool(video_content)
            }
            
        except Exception as e:
            error_msg = f"YouTube analysis encountered an error: {str(e)}"
            print(f"\n🎥 YOUTUBE ANALYSIS:")
            print("-" * 50)
            print(error_msg)
            print("-" * 50)
            
            all_answers.append({
                "agent": "YouTube Agent",
                "query": query,
                "answer": error_msg,
                "sources_count": 0
            })
            
            youtube_data = {
                "summary": error_msg,
                "raw_results": [],
                "videos_count": 0,
                "content_extracted": False
            }
        
        return {
            "youtube_data": youtube_data,
            "citations": citations,
            "completed_agents": completed + ["youtube_agent"],
            "all_intermediate_answers": all_answers
        }
    

# --- Tavily Agent ---
class TavilyAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        query = state["query"]
        completed = state.get("completed_agents", [])
        citations = state.get("citations", [])
        all_answers = state.get("all_intermediate_answers", [])
        
        try:
            if tavily_tool.available:
                tavily_results = tavily_tool.run(query)
                
                # Compile full web content for analysis
                web_content = ""
                for result in tavily_results:
                    citations.append({
                        "source": "Web Search",
                        "title": result.get("title", "Web Article"),
                        "url": result.get("url", ""),
                        "type": "web_article",
                        "published_date": result.get("published_date", ""),
                        "score": result.get("score", 0)
                    })
                    
                    # Compile web content for analysis
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
            else:
                tavily_results = []
                web_content = ""
            
            # Pass the actual web content to the LLM
            chain = tavily_analysis_prompt | llm | StrOutputParser()
            summary = chain.invoke({
                "query": query,
                "results": web_content or f"No Tavily API available. Analyzing topic: {query}"
            })
            
            print(f"\n🌐 WEB SEARCH ANALYSIS:")
            print("-" * 50)
            print(summary)
            print("-" * 50)
            
            results_count = len(tavily_results)
            all_answers.append({
                "agent": "Web Search Agent",
                "query": query,
                "answer": summary,
                "sources_count": results_count
            })
                
            tavily_data = {
                "summary": summary,
                "raw_results": tavily_results,  # Now contains full web content
                "results_count": results_count,
                "content_extracted": bool(web_content),
                "has_ai_summary": any(r.get("type") == "ai_summary" for r in tavily_results)
            }
            
        except Exception as e:
            error_msg = f"Web search analysis encountered an error: {str(e)}"
            print(f"\n🌐 WEB SEARCH ANALYSIS:")
            print("-" * 50)
            print(error_msg)
            print("-" * 50)
            
            all_answers.append({
                "agent": "Web Search Agent",
                "query": query,
                "answer": error_msg,
                "sources_count": 0
            })
            
            tavily_data = {
                "summary": error_msg,
                "raw_results": [],
                "results_count": 0,
                "content_extracted": False,
                "has_ai_summary": False
            }
        
        return {
            "tavily_data": tavily_data,
            "citations": citations,
            "completed_agents": completed + ["tavily_agent"],
            "all_intermediate_answers": all_answers
        }

# --- ArXiv Agent ---
# --- Enhanced ArXiv Agent ---
class ArxivAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        query = state["query"]
        completed = state.get("completed_agents", [])
        citations = state.get("citations", [])
        all_answers = state.get("all_intermediate_answers", [])
        
        try:
            if arxiv_tool.available:
                arxiv_results = arxiv_tool.run(query)
                
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
                            academic_content += f"Abstract: {original_abstract[:1000]}...\n"
                    
                    academic_content += "\n" + "="*60 + "\n"
            else:
                arxiv_results = []
                academic_content = ""
            
            # Pass the actual academic content to the LLM
            chain = arxiv_analysis_prompt | llm | StrOutputParser()
            summary = chain.invoke({
                "query": query,
                "results": academic_content or f"No ArXiv API available. Analyzing topic: {query}"
            })
            
            print(f"\n📄 ACADEMIC RESEARCH ANALYSIS:")
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
                "raw_results": arxiv_results,  # Now contains full paper data
                "papers_count": papers_count,
                "content_extracted": bool(academic_content),
                "has_abstracts": any(paper.get("cleaned_abstract") for paper in arxiv_results)
            }
            
        except Exception as e:
            error_msg = f"ArXiv analysis encountered an error: {str(e)}"
            print(f"\n📄 ACADEMIC RESEARCH ANALYSIS:")
            print("-" * 50)
            print(error_msg)
            print("-" * 50)
            
            all_answers.append({
                "agent": "Academic Research Agent",
                "query": query,
                "answer": error_msg,
                "sources_count": 0
            })
            
            arxiv_data = {
                "summary": error_msg,
                "raw_results": [],
                "papers_count": 0,
                "content_extracted": False,
                "has_abstracts": False
            }
        
        return {
            "arxiv_data": arxiv_data,
            "citations": citations,
            "completed_agents": completed + ["arxiv_agent"],
            "all_intermediate_answers": all_answers
        }

# --- Synthesis Agent ---
class SynthesisAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        query = state["query"]
        all_answers = state.get("all_intermediate_answers", [])
        
        youtube_summary = state.get('youtube_data', {}).get('summary', 'No YouTube data available')
        tavily_summary = state.get('tavily_data', {}).get('summary', 'No web data available')
        arxiv_summary = state.get('arxiv_data', {}).get('summary', 'No academic data available')
        
        try:
            chain = synthesis_prompt | llm | StrOutputParser()
            
            script = chain.invoke({
                "query": query,
                "youtube_data": youtube_summary,
                "tavily_data": tavily_summary,
                "arxiv_data": arxiv_summary
            })
            
            # Clean up the script
            # Clean up the script more thoroughly
            script = re.sub(r'^(Here\'s|Here is|I\'ll create).*?:\s*', '', script, flags=re.IGNORECASE)
            script = re.sub(r'^\*\*.*?\*\*\s*', '', script, flags=re.MULTILINE)
            script = re.sub(r'^\d+\.\s*', '', script, flags=re.MULTILINE)  # Remove numbered points
            script = re.sub(r'^[\*\-•]\s*', '', script, flags=re.MULTILINE)  # Remove bullets
            script = re.sub(r'\[.*?\]', '', script)  # Remove any timestamps or markers
            script = re.sub(r'\*\*.*?\*\*', '', script)  # Remove any remaining bold text
            script = re.sub(r'\n{3,}', '\n\n', script)  # Normalize spacing
            script = script.strip()
            
            print("\n" + "="*80)
            print("📽️ FINAL VIDEO SCRIPT")
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
            print("📽️ FINAL VIDEO SCRIPT (ERROR FALLBACK)")
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

# --- Feedback Agent ---
class FeedbackAgent(Runnable):
    def invoke(self, state: VideoState, config=None):
        iteration_count = state.get("iteration_count", 0)
        all_answers = state.get("all_intermediate_answers", [])
        
        if iteration_count >= 3:  # Reduced max iterations 
            print(f"\n🔄 Maximum iterations (3) reached - accepting current result")
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
            
            print(f"\n🔍 Raw feedback response: {repr(feedback_response)}")  # Debug line
            
            # Clean and parse JSON response with better error handling
            feedback_response = feedback_response.strip()
            
            # Remove markdown code blocks if present
            if feedback_response.startswith('```json'):
                feedback_response = feedback_response.replace('```json', '').replace('```', '').strip()
            elif feedback_response.startswith('```'):
                feedback_response = feedback_response.replace('```', '').strip()
            
            # Handle empty or invalid responses
            if not feedback_response:
                print("⚠️ Empty response from LLM - using default feedback")
                return {
                    "feedback_score": 7,
                    "feedback_reason": "Empty feedback response - accepting current result",
                    "next_agent": "FINISH",
                    "all_intermediate_answers": all_answers
                }
            
            # Try to parse JSON with multiple fallback strategies
            feedback_data = None
            
            #  Direct JSON parsing
            try:
                feedback_data = json.loads(feedback_response)
            except json.JSONDecodeError:
                #  Try to extract JSON from text
                json_match = re.search(r'\{.*\}', feedback_response, re.DOTALL)
                if json_match:
                    try:
                        feedback_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            
            #  Parse manually if JSON parsing fails
            if not feedback_data:
                print("⚠️ JSON parsing failed - attempting manual parsing")
                
                # Try to extract score manually
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
            
            # Extract values with defaults
            score = feedback_data.get("score", 7)
            reason = feedback_data.get("reason", "No feedback provided")
            rewritten_query = feedback_data.get("rewritten_query")
            
            # Validate score
            if not isinstance(score, int) or score < 1 or score > 10:
                print(f"⚠️ Invalid score {score} - defaulting to 7")
                score = 7
            
            print(f"\n🔍 Feedback Analysis:")
            print(f"   Score: {score}/10")
            print(f"   Reason: {reason}")
            
            if score >= 7:
                print(f"✅ Quality approved! Proceeding to save script.")
                return {
                    "feedback_score": score,
                    "feedback_reason": reason,
                    "next_agent": "FINISH",
                    "all_intermediate_answers": all_answers
                }
            else:
                new_iteration = iteration_count + 1
                print(f"🔄 Score below 7 - Starting iteration {new_iteration}")
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
                    "next_agent": "supervisor",
                    "all_intermediate_answers": all_answers
                }
                
        except Exception as e:
            print(f"⚠️ Feedback system error: {str(e)} - accepting current result")
            print(f"🔍 Full error traceback:")
            import traceback
            traceback.print_exc()
            
            return {
                "feedback_score": 7,
                "feedback_reason": f"Feedback system error - accepting result: {str(e)}",
                "next_agent": "FINISH",
                "all_intermediate_answers": all_answers
            }

# --- Router Function ---
def router(state: VideoState) -> Literal["supervisor", "youtube_agent", "tavily_agent", "arxiv_agent", "synthesis_agent", "feedback_agent", "__end__"]:
    next_agent = state.get("next_agent", "")
    
    if next_agent == "FINISH" or not next_agent:
        return "__end__"
    elif next_agent in ["supervisor", "youtube_agent", "tavily_agent", "arxiv_agent", "synthesis_agent", "feedback_agent"]:
        return next_agent
    else:
        return "supervisor"

# --- Graph Builder ---
def build_research_graph():
    builder = StateGraph(VideoState)
    
    builder.add_node("supervisor", SupervisorAgent())
    builder.add_node("youtube_agent", YouTubeAgent())
    builder.add_node("tavily_agent", TavilyAgent())
    builder.add_node("arxiv_agent", ArxivAgent())
    builder.add_node("synthesis_agent", SynthesisAgent())
    builder.add_node("feedback_agent", FeedbackAgent())
    
    builder.add_edge(START, "supervisor")
    
    builder.add_conditional_edges("supervisor", router)
    builder.add_edge("youtube_agent", "supervisor")
    builder.add_edge("tavily_agent", "supervisor")
    builder.add_edge("arxiv_agent", "supervisor")
    builder.add_conditional_edges("synthesis_agent", router)
    builder.add_conditional_edges("feedback_agent", router)
    
    return builder.compile()

# --- Citation Formatter ---
def format_citations(citations: List[Dict[str, str]]) -> str:
    if not citations:
        return "📚 SOURCES:\nNo external sources were cited during research."
    
    formatted = "📚 RESEARCH SOURCES\n" + "=" * 60 + "\n"
    
    youtube_sources = [c for c in citations if c["source"] == "YouTube"]
    web_sources = [c for c in citations if c["source"] == "Web Search"]
    arxiv_sources = [c for c in citations if c["source"] == "arXiv"]
    
    if youtube_sources:
        formatted += "\n🎥 YOUTUBE VIDEOS\n" + "-" * 30 + "\n"
        for i, source in enumerate(youtube_sources, 1):
            formatted += f"{i}. {source['title']}\n   🔗 {source['url']}\n\n"
    
    if web_sources:
        formatted += "\n🌐 WEB ARTICLES\n" + "-" * 30 + "\n"
        for i, source in enumerate(web_sources, 1):
            formatted += f"{i}. {source['title']}\n   🔗 {source['url']}\n\n"
    
    if arxiv_sources:
        formatted += "\n📄 ACADEMIC PAPERS\n" + "-" * 30 + "\n"
        for i, source in enumerate(arxiv_sources, 1):
            formatted += f"{i}. {source['title']}\n   🔗 {source['url']}\n\n"
    
    formatted += f"\n📊 TOTAL SOURCES: {len(citations)}\n"
    formatted += f"   • YouTube Videos: {len(youtube_sources)}\n"
    formatted += f"   • Web Articles: {len(web_sources)}\n"
    formatted += f"   • Academic Papers: {len(arxiv_sources)}\n"
    
    return formatted

# --- All Answers Formatter ---
def format_all_answers(all_answers: List[Dict[str, str]], feedback_info: Dict[str, Any]) -> str:
    if not all_answers:
        return "No intermediate answers were generated."
    
    formatted = "🔍 ALL RESEARCH FINDINGS\n" + "=" * 60 + "\n\n"
    
    # Add feedback information at the top
    formatted += f"📊 QUALITY METRICS\n"
    formatted += f"Final Feedback Score: {feedback_info.get('final_score', 'N/A')}/10\n"
    formatted += f"Total Iterations: {feedback_info.get('total_iterations', 'N/A')}\n"
    formatted += f"Final Feedback: {feedback_info.get('final_reason', 'No feedback available')}\n"
    formatted += "\n" + "="*60 + "\n\n"
    
    for i, answer in enumerate(all_answers, 1):
        agent = answer.get("agent", "Unknown Agent")
        query = answer.get("query", "No query")
        content = answer.get("answer", "No answer")
        sources = answer.get("sources_count", 0)
        
        formatted += f"📋 STEP {i}: {agent.upper()}\n"
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
        print(f"❌ Error saving script: {str(e)}")
        return False
    

def save_detailed_report(all_answers: List[Dict[str, str]], citations: List[Dict[str, str]], 
                        feedback_info: Dict[str, Any], script: str, filename: str = "detailed_report.txt") -> bool:
    try:
        content = f"DETAILED RESEARCH REPORT\n"
        content += f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "="*80 + "\n\n"
        
        # Add all research findings
        content += format_all_answers(all_answers, feedback_info)
        content += "\n\n"
        
        # Add final script
        content += "🎬 FINAL VIDEO SCRIPT\n" + "="*60 + "\n"
        content += script + "\n\n"
        
        # Add citations
        content += format_citations(citations)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"❌ Error saving detailed report: {str(e)}")
        return False

# --- Main Execution ---
def main():
    print("\n" + "="*80)
    print("🎬 AI VIDEO SCRIPT RESEARCH AGENT")
    print("="*80)
    print("This AI agent will research your topic using multiple sources")
    print("and generate a comprehensive 6-10 minute video script.\n")
    
    # Get user input
    while True:
        user_query = input("📝 Enter your video topic: ").strip()
        if user_query:
            break
        print("❌ Please enter a valid topic.")
    
    print(f"\n🎯 Research Topic: '{user_query}'")
    print(f"🔍 Starting comprehensive research...\n")
    
    # Build and run the graph
    graph = build_research_graph()
    
    initial_state: VideoState = {
        "query": user_query,
        "original_query": user_query,
        "next_agent": "supervisor",
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
        "all_intermediate_answers": []
    }
    
    try:
        # Execute the graph
        final_state = graph.invoke(initial_state)
        
        # Extract results
        script = final_state.get("final_answer", "No script generated")
        citations = final_state.get("citations", [])
        all_answers = final_state.get("all_intermediate_answers", [])
        feedback_score = final_state.get("feedback_score", 0)
        feedback_reason = final_state.get("feedback_reason", "No feedback")
        iteration_count = final_state.get("iteration_count", 0) + 1
        
        feedback_info = {
            "final_score": feedback_score,
            "total_iterations": iteration_count,
            "final_reason": feedback_reason
        }
        
        print("\n" + "="*80)
        print("✅ RESEARCH COMPLETE!")
        print("="*80)
        
        # Display summary
        print(f"\n📊 RESEARCH SUMMARY:")
        print(f"   • Final Quality Score: {feedback_score}/10")
        print(f"   • Total Iterations: {iteration_count}")
        print(f"   • Sources Found: {len(citations)}")
        print(f"   • Research Steps: {len(all_answers)}")
        
        # Save files
        print(f"\n💾 SAVING FILES:")
        
        script_saved = save_script_to_file(script, feedback_info, "video_script.txt")
        if script_saved:
            print("   ✅ Video script saved to 'video_script.txt'")
        else:
            print("   ❌ Failed to save video script")
        
        report_saved = save_detailed_report(all_answers, citations, feedback_info, script, "research_report.txt")
        if report_saved:
            print("   ✅ Detailed report saved to 'research_report.txt'")
        else:
            print("   ❌ Failed to save detailed report")
        
        # Display citations
        print(f"\n{format_citations(citations)}")
        
        print(f"\n🎉 SUCCESS! Your video script has been generated and saved.")
        print(f"📁 Check the current directory for the generated files.")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Process interrupted by user.")
        print(f"🛑 Research stopped before completion.")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print(f"🔧 Please check your API keys and internet connection.")
        import traceback
        print(f"\nFull error details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()