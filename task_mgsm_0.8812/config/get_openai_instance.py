

def get_openai_instance():
    # Add logic here if needed to create or configure the OpenAI client
    api_key = os.getenv("MISTRAL_API_KEY")
    return Mistral(api_key=api_key)
