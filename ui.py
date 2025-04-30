import gradio as gr
import requests
import uuid

session_id = str(uuid.uuid4())

API_URL = "http://localhost:8000"

def chat(message, history):
    if message.strip() == "":
        return history, ""
    
    response = requests.post(
        f"{API_URL}/chat",
        json={"session_id": session_id, "message": message, "stream": False}
    )
    
    if response.status_code == 200:
        bot_message = response.json()["response"]
        history.append((message, bot_message))
        return history, ""
    else:
        history.append((message, f"Error: {response.text}"))
        return history, ""

def switch_model(model_name):
    global session_id
    session_id = str(uuid.uuid4())
    
    response = requests.post(
        f"{API_URL}/set_model",
        json={"model": model_name}
    )
    return f"Model switched to {model_name} (new session created)"

with gr.Blocks(title="Multi-LLM Chat Platform - Basic UI") as demo:
    gr.Markdown("# Multi-LLM Chat Platform")
    gr.Markdown("Chat with different LLM models - OpenAI, Claude, or Gemini")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=500)
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
            model_info = gr.Markdown("""
            **Available Models:**
            - **OpenAI**: GPT-3.5 Turbo
            - **Claude**: Claude 3 Opus
            - **Gemini**: Gemini 1.5 Pro
            """)
    
    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    submit.click(chat, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)
    model_radio.change(switch_model, model_radio, None)

if __name__ == "__main__":
    print("Starting Gradio frontend. Access the chat interface at http://127.0.0.1:7860")
    demo.launch()