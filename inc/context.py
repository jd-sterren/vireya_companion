import os
from datetime import datetime
import time
import threading
import sys
import ollama
import inc.functions as bf

LOG_FILE = "inc/logs/vireya_conversation_log.txt"

def log_conversation(role, text):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {role}: {text.strip()}\n")

def summarize_session(history, engine="local"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_prompt = (
        "Write a short, reflective summary of this conversation. Focus on emotional tone, key themes, and what Vireya should remember for next time:\n\n"
        + "\n".join(history[-8:])
    )

    if engine == "openai":
        from inc.conversation import openai_chain
        reflection = openai_chain.predict(input=summary_prompt)
    else:
        reflection_response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        reflection = reflection_response['message']['content'].strip()

    return f"[{now}] {reflection.strip()}"

def save_context(reflection):
    bf.save_context(reflection)

def shutdown_app():
    time.sleep(1)
    sys.exit()
