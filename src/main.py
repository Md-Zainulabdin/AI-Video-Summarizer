import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import get_file, upload_file

import os
import time
import tempfile
from pytube import YouTube
import google.generativeai as genai

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Key Configration
API_KEY = os.getenv("GOOGLE_APY_KEY")
if  API_KEY:
    genai.configure(API_KEY)


# Page Configration
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

st.title("Phidata Video AI Agent")
st.header("Powered by Gemini 2.0 Flash")

@st.cache_resource

def initilize_agent():
    return Agent(
       name="Video AI Summarizer",
       model=Gemini(id="gemini-2.0-flash-exp"),
       tools=[DuckDuckGo()],
       markdown=True
    )
    
multimodal_agent = initilize_agent()
video_path = None   

# Video File Upload Section
st.subheader("Upload a Video or Paste a URL")
video_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"],
    help="Upload a video for AI analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name   
        
# Video Display Section
if video_path:
    st.video(video_path, format="video/mp4", start_time=0)
    user_query = st.text_area(
        "What insight are you seeking from the video?",
        placeholder="Ask anything about the video content.",
        help="Provide specific questions or insights you want from the video"
    )             

    if st.button("Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):   
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                    
                        # Prompt generation for analysis
                        prompt = (
                            f"""
                            Analyze the uploaded video for content and context.
                            Respond to the following query using video insights and supplementary web research:
                            {user_query}
                            
                            Provide detailed, user-friendly, and actionable response.
                            """
                        )
                    
                    response = multimodal_agent.run(prompt, videos=[processed_video])
                    
                    st.subheader("Analysis Result")
                    st.markdown(response.content)
            except Exception as error:
                st.error("An error occurred during analysis")        