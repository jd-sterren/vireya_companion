import ollama
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
#from inc.model_router import get_model_for_emotion

conversation_history = []

def get_openai_chain(openai_api_key, base_prompt):
    memory = ConversationBufferMemory()
    custom_prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=f"""{base_prompt}

        Conversation history:
        {{history}}

        Human: {{input}}
        AI:"""
            )
    return ConversationChain(
        llm=ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4-turbo", temperature=0.3),
        memory=memory,
        prompt=custom_prompt
    )

def format_local_messages(user_input, base_prompt, character):
    messages = [{"role": "system", "content": base_prompt}]
    for line in conversation_history[-5:]:
        if line.startswith("User:"):
            messages.append({"role": "user", "content": line.replace("User:", "").strip()})
        elif line.startswith(f"{character}:"):
            messages.append({"role": "assistant", "content": line.replace(f"{character}:", "").strip()})
    messages.append({"role": "user", "content": user_input})
    return messages

def get_local_response(user_input, base_prompt, feel="neutral", default_model="openhermes"):
    #model = get_model_for_emotion(feel)
    messages = format_local_messages(user_input, base_prompt, "Vireya")
    response = ollama.chat(model=default_model, messages=messages)
    return response['message']['content'].strip()
