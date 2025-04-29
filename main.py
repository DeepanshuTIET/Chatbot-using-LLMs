from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAIChat
import os
import requests
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LLM Config
# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

# For debugging
print(f"OpenAI API Key: {openai_api_key[:5]}... (length: {len(openai_api_key) if openai_api_key else 0})")
print(f"Anthropic API Key: {anthropic_api_key[:5]}... (length: {len(anthropic_api_key) if anthropic_api_key else 0})")

# In-memory session and model state
session_store = {}
active_model = {"name": "openai"}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ModelRequest(BaseModel):
    model: str

@app.post("/set_model")
def set_model(req: ModelRequest):
    previous_model = active_model["name"]
    active_model["name"] = req.model
    print(f"Model changed from {previous_model} to {req.model}")
    return {"message": f"Model set to {req.model}"}

@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id
    message = req.message
    
    # Debug information
    print(f"Processing chat request for session {session_id}")
    print(f"Current active model: {active_model['name']}")

    # Initialize session if new
    if session_id not in session_store:
        print(f"Creating new session {session_id}")
        session_store[session_id] = []

    session_store[session_id].append({"role": "user", "content": message})

    history = session_store[session_id]
    model = active_model["name"]
    print(f"Using model: {model}")

    if model == "openai":
        response = call_openai(history)
    elif model == "claude":
        response = call_claude(history)
    elif model == "gemini":
        response = call_gemini("\n".join([m["content"] for m in history]))
    else:
        response = "Unknown model"

    session_store[session_id].append({"role": "assistant", "content": response})
    return {"response": response}

# === LLM Integration Functions ===
def call_openai(messages):
    try:
        print("Attempting to call OpenAI API with new client")
        # Using new client-based approach as per OpenAI documentation
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000
        )
        print("OpenAI API call successful")
        return response.choices[0].message.content
    except Exception as e:
        error_message = f"OpenAI API error: {str(e)}"
        print(error_message)
        
        # Return informative error message
        return f"Error connecting to OpenAI API: {str(e)}. Please check your API key and quota."

def call_claude(messages):
    try:
        # For debugging
        print(f"Call Claude function called with {len(messages)} messages")
        
        # Format messages for Claude API
        claude_messages = []
        for msg in messages:
            claude_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Call Claude API directly using requests
        print("Using Anthropic API to generate response")
        headers = {
            "x-api-key": anthropic_api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"  # Standard API version
        }
        
        # Use claude-2.0 model which is widely available
        data = {
            "model": "claude-3-opus-20240229",  # Standard model name format
            "messages": claude_messages,
            "max_tokens": 1000
        }
        
        print(f"Sending request to Claude API with model: {data['model']}")
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            print("Claude API call successful")
            return response.json()["content"][0]["text"]
        else:
            error_message = f"Error with Claude API: {response.text}"
            print(error_message)  # Log the error
            return f"Error connecting to Claude API: {response.text}. Please check your API key and model name."
    except Exception as e:
        error_message = f"Exception when calling Claude API: {str(e)}"
        print(error_message)  # Log the error
        return f"Error with Claude API: {str(e)}. Please check your network connection and API key."

def call_gemini(prompt):
    try:
        print("Attempting to call Gemini API")
        # Make sure we're using the correct Gemini model
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Handle prompt as structured content
        generation_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 1024,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        print("Gemini API call successful")
        return response.text
    except Exception as e:
        error_message = f"Gemini API error: {str(e)}"
        print(error_message)
        return f"Error connecting to Gemini API: {str(e)}. Please check your API key."

# This function is no longer used since we want to use the actual LLM APIs
# Keeping it for reference but it's not called anywhere
def generate_local_response(query):
    print("Using local fallback response generator")
    return "The AI service is currently experiencing technical difficulties. Please check your API keys and try again later."

# For debugging
@app.get("/")
def root():
    return {"message": "AI Chatbot API is running!"}
