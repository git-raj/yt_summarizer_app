# YouTube Video Transcription and Summarization App

This is a Streamlit-based web application that allows users to transcribe audio from YouTube videos and summarize the transcriptions using different language models (LLMs). The application supports OpenAI, Claude (Anthropic), and Gemini APIs for summarization.

## Features

- **Transcription**: Extracts audio from YouTube videos and transcribes it using the Whisper model.
- **Summarization**: Summarizes the transcribed text using one of the three available language models:
  - OpenAI (`gpt-3.5-turbo`, `gpt-4`)
  - Claude (Anthropic)
  - Gemini (Google)
- **API Key Management**: Configure API keys for the selected LLM in the app.
- **Download Options**: Download both the transcription and the generated summary as text files.

## Requirements

### Libraries

Install the required libraries using `pip`:

```bash
pip install streamlit openai yt-dlp whisper requests
```

Additionally, ensure that `ffmpeg` is installed for handling audio extraction. You can install `ffmpeg` as follows:

- On **Ubuntu**:
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

- On **macOS** (with Homebrew):
  ```bash
  brew install ffmpeg
  ```

- On **Windows**:
  Download from [FFmpeg official website](https://ffmpeg.org/download.html) and follow the installation instructions.

### API Keys

You will need API keys for each of the language models you intend to use:

- **OpenAI**: You can get the API key from [OpenAI](https://openai.com).
- **Claude (Anthropic)**: You can get the API key from [Anthropic](https://www.anthropic.com/).
- **Gemini (Google)**: You can get access to the API key from Google’s Generative Language API.

## How to Use

1. Clone or download the repository.
2. Install the required dependencies as mentioned above.
3. Run the application with Streamlit:
   ```bash
   streamlit run app.py
   ```
   Replace `app.py` with the name of your Python file containing the Streamlit code.

4. Open the web browser and navigate to the local Streamlit server (usually [http://localhost:8501](http://localhost:8501)).

## Application Flow

### API Configuration

1. Select a language model from the dropdown menu (OpenAI, Claude, or Gemini).
2. Enter the API key for the selected model.
3. Click on "Test Connection" to validate the API key.

### Transcribe and Summarize

1. **Transcription**:
   - Enter the YouTube URL of the video you want to transcribe.
   - Click on "Transcribe" to extract audio and transcribe it using Whisper.
   - The transcribed text will be displayed in a text area.

2. **Summarization**:
   - Choose the desired word length for the summary (50, 100, or 200 words).
   - Click on "Summarize" to summarize the transcription using the selected language model.
   - The generated summary will be displayed below the transcription text.

3. **Download**:
   - You can download both the transcription and the summary as text files using the "Download Transcription" and "Download Summary" buttons.

## Code Structure

- **`app.py`**: Contains the main Streamlit app code.
- **Helper Functions**:
  - `validate_api_key`: Validates API keys for the selected model.
  - `extract_audio_from_youtube`: Downloads audio from a YouTube video and transcribes it using Whisper.
  - `summarize_text`: Summarizes the transcribed text using the selected language model.

## Known Issues

- Whisper's transcription can take some time for longer videos.
- Ensure that `ffmpeg` is correctly installed and available in the system's PATH to avoid issues with audio extraction.
- The API keys should be valid and should have appropriate permissions for usage.

## License

This project is licensed under the MIT License.
