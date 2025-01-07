import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import get_file, upload_file


import os
import time
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# API Key Configration
API_KEY = os.getenv("GOOGLE_APY_KEY")
if  API_KEY:
    genai.configure(API_KEY)

@st.cache_resource
def initialize_agent():
    """Initialize the AI agent for video analysis."""
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True
    )

# Initialize multimodal agent
multimodal_agent = initialize_agent()

# Page Configuration
st.set_page_config(
    page_title="AI Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

# Sidebar Section
with st.sidebar:
    st.header("Real-Time Video Analysis")
    video_file = st.file_uploader(
        "Upload a video file:",
        type=["mp4", "mov", "avi"],
        help="Supported formats: MP4, MOV, AVI"
    )

# Video Upload and Processing
video_path = None
if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

# Main Chat Interface
st.markdown(
    """
    <h2 style="text-align: center;">
        Agentic Video Summarizer Powered by Phidata
    </h2>
    """,
    unsafe_allow_html=True
)

# Display Uploaded Video
if video_path:
    st.video(video_path, format="video/mp4", start_time=0)
    user_query = st.text_area(
        "What insight are you seeking from the video?",
        placeholder="Ask specific questions or describe what you're looking for.",
        help="Provide detailed prompts to get actionable insights."
    )

    analyze_button = st.button("Analyze Video", key="analyze_video_button")

    if analyze_button:
        if not user_query:
            st.warning("Please provide a question or insight to proceed.")
        else:
            try:
                with st.spinner("Processing the video and generating insights..."):
                    processed_video = upload_file(video_path)

                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # Generate Prompt and Fetch Response
                    prompt = (
                        f"Analyze the uploaded video to understand its content, context, and main highlights.\n\n"
                        f"Answer the following query provided by the user:\n"
                        f"\"{user_query}\"\n\n"
                        f"If the video contains any specific or unique topics, events, or references, perform a supplementary web search "
                        f"to gather additional information and context. Use reliable sources for web research.\n\n"
                        f"Your response should include:\n"
                        f"- Detailed insights derived from the video content.\n"
                        f"- Any interesting findings from web searches relevant to the video.\n"
                        f"- A user-friendly explanation with actionable or informative details.\n"
                        f"- Reference credible web sources for any additional information."
                    )
                    response = multimodal_agent.run(prompt, videos=[processed_video])

                    st.success("Video analysis complete!")
                    st.subheader("Analysis Result")
                    st.markdown(response.content)
            except Exception as error:
                st.error(f"An error occurred: {error}")
