# Conversational AI Chatbot

A multi-model chatbot that integrates OpenAI, Claude, and Gemini AI models with a FastAPI backend and Gradio frontend.

## Features

- Switch between multiple AI models (OpenAI, Claude, Gemini)
- Persistent chat sessions
- Clean, user-friendly interface
- Easy setup and configuration

## Setup

1. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   Create a `.env` file in the root directory and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

3. **Start the Backend**:
   Run the FastAPI server:
   ```
   uvicorn main:app --reload --port 8000
   ```

4. **Start the Frontend**:
   In a separate terminal, start the Gradio UI:
   ```
   python frontend.py
   ```

5. **Access the Chatbot**:
   Open your browser and go to: http://127.0.0.1:7860

## Usage

- **Switch Models**: Use the `/set_model` endpoint to switch between OpenAI, Claude, and Gemini models.
- **Enable Streaming**: Use the `stream` parameter in the `/chat` endpoint to enable streaming responses.

## Architecture

- **Backend**: FastAPI server that handles model switching and chat functionality
- **Frontend**: Gradio UI for user interaction
- **Models**: OpenAI GPT, Anthropic Claude, and Google Gemini

## Files

- `main.py`: FastAPI backend server
- `frontend.py`: Gradio UI frontend
- `requirements.txt`: Required dependencies
- `.env`: Configuration file for API keys
