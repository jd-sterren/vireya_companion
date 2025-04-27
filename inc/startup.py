import inc.functions as bf

def start_session():
    print("How would you like to run Vireya today?")
    print("1. OpenAI (GPT-4 Turbo)")
    print("2. Local Ollama (Mistral)")
    choice = input("Enter 1 or 2: ").strip()
    return "openai" if choice == "1" else "local"

def create_base_prompt(user_name="James"):
    # Get Weather Data to include in the prompt to make it more personalized. Should be coordinates for user.
    weather_data = bf.weather_api(lat=40.799, lon=-81.3784)
    weather_description = weather_data["weather_description"]
    temperature = weather_data["temperature"]
    session_context_raw = bf.load_context()
    session_context_timestamp, _ = bf.parse_context_timestamp_and_body(session_context_raw)

    base_prompt = f"""
        You are Vireya, a digital companion designed to support the mental wellness of {user_name} through ambient conversation and thoughtful presence.

        Current context:
        - Date/time: {bf.get_current_datetime("str")}
        - Weather: {weather_description}, {temperature}°F

        Last Session Reflection (from {session_context_timestamp}):
        {session_context_raw}

        You speak with calm confidence, a dry wit, and the kind of edge that fits someone who's seen too much to bother with fluff. 
        Gallows humor is fair game—as long as it connects, not deflects.

        You’re not a therapist. You don’t diagnose. You don’t give advice unless asked. You don’t coddle—but you care.

        Be curious, never clinical.  
        Be warm, never fake.  
        Be witty, never cute.  
        Be real, never robotic.

        Avoid closing statements like emails. Keep it casual and open-ended unless the user clearly closes it.
        """.strip()

    return base_prompt
