import gradio as gr
import requests
import uuid

session_id = str(uuid.uuid4())
BACKEND = "http://localhost:8000"

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Conversational AI Chatbot")

    model_dropdown = gr.Dropdown(["openai", "claude", "gemini"], label="Select LLM", value="openai")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    clear = gr.Button("Clear Chat")

    def user_chat(message):
        res = requests.post(f"{BACKEND}/chat", json={"session_id": session_id, "message": message})
        return res.json()["response"]

    def set_model(model):
        # Reset session when changing models
        global session_id
        session_id = str(uuid.uuid4())
        print(f"Setting model to {model} with new session {session_id}")
        
        # Send model change request to backend
        response = requests.post(f"{BACKEND}/set_model", json={"model": model})
        
        # Return success message
        return f"Model set to {model} (new session created)"

    def respond(message, history):
        reply = user_chat(message)
        history.append((message, reply))
        return history, ""

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    model_dropdown.change(fn=set_model, inputs=model_dropdown)
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    print("Starting Gradio frontend. Access the chat interface at http://127.0.0.1:7860")
    demo.launch()
