from dotenv import load_dotenv
import os
import inc.startup as startup
import inc.ui as ui
import inc.conversation as convo
from inc.credential_manager import inject_decrypted_env

inject_decrypted_env(environment="prod")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    engine = startup.start_session()
    base_prompt = startup.create_base_prompt()

    openai_chain = None
    if engine == "openai":
        openai_chain = convo.get_openai_chain(OPENAI_API_KEY, base_prompt)

    ui.launch_gradio(engine, base_prompt, openai_chain)