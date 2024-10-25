import streamlit as st
import os
import yt_dlp
import whisper
import tempfile
import shutil
import openai


# --- Initialize session state variables at the start of the script ---
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "OpenAI"  # Default value to avoid empty selection

if 'transcription' not in st.session_state:
    st.session_state.transcription = ""

if 'summary' not in st.session_state:
    st.session_state.summary = ""


# --- API Configuration Tab ---
def api_config():
    st.header("Configure API Credentials")

    # Select the model (OpenAI, Claude, Gemini)
    st.session_state.selected_model = st.selectbox(
        "Choose LLM API", ["OpenAI", "Claude", "Gemini"], index=["OpenAI", "Claude", "Gemini"].index(st.session_state.selected_model)
    )

    # Input for API Key
    st.session_state.api_key = st.text_input(f"Enter {st.session_state.selected_model} API Key", type="password")

    # Test API Connection Button
    if st.button("Test Connection"):
        if validate_api_key(st.session_state.selected_model, st.session_state.api_key):
            st.success(f"{st.session_state.selected_model} API Key is valid.")
        else:
            st.error(f"Failed to connect with {st.session_state.selected_model} API Key.")

# --- YouTube Transcription and Summarization ---
def transcribe_and_summarize():
    st.header("YouTube Video Transcription and Summarization")

    # YouTube URL input
    youtube_url = st.text_input("Enter YouTube Video URL")

    # Transcribe Button
    if st.button("Transcribe"):
        if not st.session_state.api_key:
            st.error("Please configure API credentials in the API Configuration tab.")
        else:
            with st.spinner('Transcribing the video...'):
                st.session_state.transcription = extract_audio_from_youtube(youtube_url)
            if st.session_state.transcription:
                st.text_area("Transcribed Text", value=st.session_state.transcription, height=300)
            else:
                st.error("Transcription failed. Please check the YouTube URL or try again later.")

    # Summarization Options
    summary_length = st.selectbox("Summarize in approx. words", [50, 100, 200])

    # Summarize Button
    if st.button("Summarize"):
        if not st.session_state.api_key:
            st.error("Please configure API credentials in the API Configuration tab.")
        elif not st.session_state.transcription:
            st.error("No transcription available. Please transcribe a video first.")
        else:
            summary = summarize_text(st.session_state.transcription, summary_length, st.session_state.api_key, st.session_state.selected_model)
            if summary:
                st.session_state.summary = summary
                st.text_area("Summarized Text", value=st.session_state.summary, height=150)
            else:
                st.error("Summarization failed. Please try again later.")

    # Download buttons
    if st.session_state.transcription:
        st.download_button("Download Transcription", st.session_state.transcription)
    if st.session_state.summary:
        st.download_button("Download Summary", st.session_state.summary)

# --- Helper Functions ---
def validate_api_key(model, api_key):
    """
    Validates the API key by making a simple request to the selected model's API.
    Supports OpenAI, Claude, and Gemini.
    """
    if model == "OpenAI":
        return validate_openai(api_key)
    elif model == "Claude":
        return validate_claude(api_key)
    elif model == "Gemini":
        return validate_gemini(api_key)
    else:
        raise ValueError("Unsupported model selected")


def extract_audio_from_youtube(youtube_url):
    """
    Extracts audio from a YouTube video and transcribes it using the Whisper model.
    
    Args:
        youtube_url (str): The YouTube video URL.
        
    Returns:
        str: The transcribed text from the video's audio.
    """
    try:
        # Create a temporary directory to save the audio file
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_file = os.path.join(temp_dir, "audio.mp3")
            
            # yt-dlp options to download the audio in best format and keep the original file
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': audio_file,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'keepvideo': True,  # This keeps the original audio file
            }

            # Download the audio from the YouTube video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])

            # Ensure file is moved to a consistent path (optional step to be sure)
            final_audio_path = os.path.join(temp_dir, "final_audio.mp3")
            shutil.move(audio_file + ".mp3", final_audio_path)

            # Load Whisper model
            model = whisper.load_model("base")  # You can choose other models: 'tiny', 'small', 'medium', 'large'

            # Transcribe the audio file using Whisper
            result = model.transcribe(final_audio_path)
            
            # Return the transcribed text
            return result['text']
    
    except Exception as e:
        print(f"An error occurred during audio extraction or transcription: {e}")
        return None


def summarize_text(text, word_count, api_key, model):
    """
    Summarizes the given text using the selected LLM API.
    
    Args:
        text (str): The text to be summarized.
        word_count (int): The desired word count for the summary.
        api_key (str): The API key for the selected model.
        model (str): The selected LLM model ("OpenAI", "Claude", "Gemini").
    
    Returns:
        str: The summarized text.
    """
    if model == "OpenAI":
        return summarize_with_openai(text, word_count, api_key)
    elif model == "Claude":
        return summarize_with_claude(text, word_count, api_key)
    elif model == "Gemini":
        return summarize_with_gemini(text, word_count, api_key)
    else:
        raise ValueError("Unsupported model selected")


def validate_api_key(model, api_key):
    """
    Validates the API key by making a simple request to the selected model's API.
    Supports OpenAI, Claude, and Gemini.
    """
    if model == "OpenAI":
        return validate_openai(api_key)
    elif model == "Claude":
        return validate_claude(api_key)
    elif model == "Gemini":
        return validate_gemini(api_key)
    else:
        raise ValueError("Unsupported model selected")


def validate_openai(api_key):
    """
    Validate OpenAI API key by making a small test request to the Chat API.
    """
    try:
        # Set the OpenAI API key
        openai.api_key = api_key

        # Perform a basic test request using the ChatCompletion API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or use "gpt-4" if preferred
            messages=[{"role": "system", "content": "Test"}],  # Simple system role message
            max_tokens=1  # Keep it minimal for validation purposes
        )

        # If the request succeeds, we assume the API key is valid
        if response:
            return True

    except openai.AuthenticationError:
        print("Invalid OpenAI API key.")
    except Exception as e:
        print(f"OpenAI API key validation failed: {e}")
    
    return False


def validate_claude(api_key):
    """
    Validate Claude (Anthropic) API key by making a test request.
    """
    try:
        # Base URL for Claude's API (Anthropic API)
        api_url = "https://api.anthropic.com/v1/complete"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }
        data = {
            "prompt": "Hello",
            "max_tokens_to_sample": 1,  # Just testing if the key works
            "model": "claude-v1"
        }
        response = requests.post(api_url, json=data, headers=headers)
        if response.status_code == 200:
            return True
    except Exception as e:
        print(f"Claude API key validation failed: {e}")
    return False

import requests

def validate_gemini(api_key):
    """
    Validate Gemini API key by making a test request to the generateContent endpoint.
    
    Args:
        api_key (str): The API key to be validated.
    
    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    try:
        # Define the API URL for Gemini key validation
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

        # Define the headers and the minimal test data for the request
        headers = {
            "Content-Type": "application/json"
        }

        # Define the payload with a very simple test prompt
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": "Explain how AI works"}
                    ]
                }
            ]
        }

        # Make the POST request to validate the API key
        response = requests.post(url, headers=headers, json=data)

        # Check if the response status code is 200 (OK) meaning the API key is valid
        if response.status_code == 200:
            return True
        else:
            print(f"Gemini API key validation failed: {response.status_code}, {response.text}")
            return False

    except Exception as e:
        print(f"An error occurred during Gemini API key validation: {e}")
        return False


# OpenAI Summarization
def summarize_with_openai(text, word_count, api_key):
    """
    Summarizes the given text using OpenAI's API.
    
    Args:
        text (str): The text to be summarized.
        word_count (int): The desired word count for the summary.
        api_key (str): The OpenAI API key.
    
    Returns:
        str: The summarized text.
    """
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can use other models like gpt-4 if needed
            prompt=f"Summarize the following text in approximately {word_count} words:\n\n{text}",
            max_tokens=word_count * 5,  # Adjust max tokens for desired summary length
            temperature=0.5  # Controls randomness
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error with OpenAI summarization: {e}")
        return "OpenAI summarization failed."

# Claude (Anthropic) Summarization
def summarize_with_claude(text, word_count, api_key):
    """
    Summarizes the given text using Claude's (Anthropic) API.
    
    Args:
        text (str): The text to be summarized.
        word_count (int): The desired word count for the summary.
        api_key (str): The Claude API key.
    
    Returns:
        str: The summarized text.
    """
    try:
        api_url = "https://api.anthropic.com/v1/complete"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }
        data = {
            "prompt": f"Summarize the following text in approximately {word_count} words:\n\n{text}",
            "max_tokens_to_sample": word_count * 5,  # Similar to OpenAI
            "model": "claude-v1"  # Claude model name
        }
        response = requests.post(api_url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()['completion'].strip()
        else:
            return f"Claude summarization failed with status code {response.status_code}."
    except Exception as e:
        print(f"Error with Claude summarization: {e}")
        return "Claude summarization failed."


def summarize_with_gemini(text, word_count, api_key):
    """
    Summarizes the given text using Gemini's API.
    
    Args:
        text (str): The text to be summarized.
        word_count (int): The desired word count for the summary.
        api_key (str): The Gemini API key.
    
    Returns:
        str: The summarized text, or an error message if the request fails.
    """
    try:
        # Define the API URL for Gemini summarization
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

        # Define the headers and the payload for the request
        headers = {
            "Content-Type": "application/json"
        }

        # Construct the request payload
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": f"Summarize the following text in {word_count} words:\n\n{text}"}
                    ]
                }
            ]
        }

        # Make the POST request to generate the summary
        response = requests.post(url, headers=headers, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            response_data = response.json()

            # Extract the generated summary from the candidates field
            candidates = response_data.get('candidates', [])
            if candidates and 'content' in candidates[0]:
                parts = candidates[0]['content'].get('parts', [])
                if parts and 'text' in parts[0]:
                    summary = parts[0]['text']
                    return summary.strip() if summary else "No summary generated."
            return "Unexpected response format."
        else:
            return f"Gemini summarization failed: {response.status_code} - {response.text}"

    except Exception as e:
        print(f"An error occurred during Gemini summarization: {e}")
        return "An error occurred during summarization."


# Main layout
st.sidebar.title("Navigation")
options = ["API Configuration", "Transcribe and Summarize"]
choice = st.sidebar.radio("Go to", options)

if choice == "API Configuration":
    api_config()
elif choice == "Transcribe and Summarize":
    transcribe_and_summarize()
