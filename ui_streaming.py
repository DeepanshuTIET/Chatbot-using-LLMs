import gradio as gr
import requests
import uuid
import os
import json
import time
from dotenv import load_dotenv
import sseclient  # You'll need to install this: pip install sseclient-py

# Load environment variables
load_dotenv()

# API endpoint (assuming FastAPI server runs on this port)
API_URL = "http://localhost:8000"

# Generate a unique session ID for this chat instance
session_id = str(uuid.uuid4())

def switch_model(model_name):
    """Switch the active LLM model"""
    response = requests.post(
        f"{API_URL}/set_model",
        json={"model": model_name}
    )
    return f"Model switched to {model_name}"

def chat_stream(message, history):
    """Send message to backend and get streaming response"""
    if message.strip() == "":
        return "Please enter a message."
    
    # Send message to backend with streaming enabled
    response = requests.post(
        f"{API_URL}/chat",
        json={"session_id": session_id, "message": message, "stream": True},
        stream=True,
        headers={"Accept": "text/event-stream"}
    )
    
    if response.status_code == 200:
        client = sseclient.SSEClient(response)
        full_response = ""
        
        for event in client.events():
            if event.data == "[DONE]":
                break
            
            try:
                data = json.loads(event.data)
                if "chunk" in data:
                    chunk = data["chunk"]
                    full_response += chunk
                    yield full_response
                    time.sleep(0.01)  # Small delay for smoother UI updates
            except json.JSONDecodeError:
                pass
        
        return full_response
    else:
        return f"Error: {response.text}"

def chat_regular(message, history):
    """Send message to backend and get regular (non-streaming) response"""
    if message.strip() == "":
        return "Please enter a message."
    
    # Send message to backend
    response = requests.post(
        f"{API_URL}/chat",
        json={"session_id": session_id, "message": message, "stream": False}
    )
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.text}"

# Create Gradio interface
with gr.Blocks(title="Multi-LLM Chat Platform") as demo:
    gr.Markdown("# Multi-LLM Chat Platform")
    gr.Markdown("Chat with different LLM models - OpenAI, Claude, or Gemini")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False
            )
            
            with gr.Row():
                submit = gr.Button("Send")
                clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            gr.Markdown("### Model Selection")
            model_radio = gr.Radio(
                ["openai", "claude", "gemini"],
                label="Select Model",
                value="openai"
            )
            
            streaming_checkbox = gr.Checkbox(
                label="Enable Streaming",
                value=True,
                info="Show responses as they're generated"
            )
            
            model_info = gr.Markdown("""
            **Available Models:**
            - OpenAI: GPT-3.5 Turbo
            - Claude: Claude 3 Opus
            - Gemini: Gemini 1.5 Pro
            """)
    
    # Set up event handlers
    model_radio.change(switch_model, inputs=model_radio, outputs=gr.Textbox(visible=False))
    
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(history, streaming):
        user_message = history[-1][0]
        
        if streaming:
            bot_response = ""
            for chunk in chat_stream(user_message, history):
                bot_response = chunk
                history[-1][1] = bot_response
                yield history
        else:
            bot_response = chat_regular(user_message, history)
            history[-1][1] = bot_response
            yield history
    
    submit.click(
        user, 
        [msg, chatbot], 
        [msg, chatbot], 
        queue=False
    ).then(
        bot, 
        [chatbot, streaming_checkbox], 
        chatbot
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the app
if __name__ == "__main__":
    demo.launch()