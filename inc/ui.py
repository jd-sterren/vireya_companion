import gradio as gr
import threading
from inc.context import log_conversation, summarize_session, save_context, shutdown_app
from inc.conversation import conversation_history, get_local_response
import inc.functions as bf

def handle_input(user_input, history, engine, base_prompt, openai_chain=None):
    if engine == "openai" and openai_chain:
        response = openai_chain.predict(input=user_input)
        tag = "[OpenAI]"
    else:
        response = get_local_response(user_input, base_prompt)
        tag = "[Local]"

    character_tagged = f"{tag} Vireya"

    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"{character_tagged}: {response}")
    log_conversation("James", user_input)
    log_conversation(character_tagged, response)

    history.append((user_input, response))
    return "", history

def end_chat(history, engine_type):
    reflection = summarize_session(conversation_history, engine_type)
    save_context(reflection)
    threading.Thread(target=lambda: shutdown_app()).start()
    return [], ""

def launch_gradio(engine_type, base_prompt, openai_chain=None):
    with gr.Blocks() as demo:
        gr.Markdown(f"## Talk to Vireya (Currently using: **{engine_type}**)")

        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Type here…", label="James:")
        clear = gr.Button("End Session + Save Context")
        shutdown_btn = gr.Button("Exit App")
        state = gr.State([])

        msg.submit(lambda m, h: handle_input(m, h, engine_type, base_prompt, openai_chain), [msg, state], [msg, chatbot])
        clear.click(lambda h: end_chat(h, engine_type), [state], [chatbot, msg])
        shutdown_btn.click(lambda: threading.Thread(target=lambda: shutdown_app()).start())

        demo.launch(inbrowser=True)
