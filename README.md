# Conversational AI Chatbot

A multi-model chatbot that integrates OpenAI, Claude, and Gemini AI models with a FastAPI backend and Gradio frontend.

## Features

- Switch between multiple AI models (OpenAI, Claude, Gemini)
- Persistent chat sessions
- Clean, user-friendly interface
- Easy setup and configuration

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your API keys in the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ANTHROPIC_API_KEY=your_claude_api_key
   ```

## Running the Application

1. Start the FastAPI backend:
   ```
   uvicorn main:app --reload --port 8000
   ```

2. In a separate terminal, start the Gradio frontend:
   ```
   python frontend.py
   ```

3. Access the chatbot interface at: http://127.0.0.1:7860

## Architecture

- **Backend**: FastAPI server that handles model switching and chat functionality
- **Frontend**: Gradio UI for user interaction
- **Models**: OpenAI GPT, Anthropic Claude, and Google Gemini

## Files

- `main.py`: FastAPI backend server
- `frontend.py`: Gradio UI frontend
- `requirements.txt`: Required dependencies
- `.env`: Configuration file for API keys
