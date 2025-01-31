import os
import json
import string
import random
import getpass
from typing import Literal
from openai import OpenAI
from datetime import datetime


#Using COHERE Models
ModelType = Literal["command-r7b-12-2024"]
DEFAULT_MODEL: ModelType = "command-r7b-12-2024"
SOLVER_MODEL: Literal["command-r7b-12-2024"] = "command-r7b-12-2024"
EVOLVE_MODEL: Literal["command-r7b-12-2024"] = "command-r7b-12-2024"
def get_openai_instance():
    api_key = api_key = os.getenv("COHERE_API_KEY")
    from langchain_cohere import ChatCohere
    return ChatCohere(api_key=api_key, model="command-r7b-12-2024"), "COHERE"


"""
#USING OPEN AI MODELS
ModelType = Literal["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
DEFAULT_MODEL: ModelType = "gpt-4o-mini"
SOLVER_MODEL: Literal["gpt-3.5-turbo"] = "gpt-3.5-turbo"
EVOLVE_MODEL: Literal["gpt-4o"] = "gpt-4o"
def get_openai_instance():
    api_key = api_key = os.getenv("OPENAI_API_KEY")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(api_key=api_key), "OPENAI"
"""


"""
#USING AVIOR AI MODELS
N_LLAMA_31_8B = "nvidia/OpenMath2-Llama3.1-8B"                    # Doesnt work with api (BadRequestError)
LLAMA_31_405B_I = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"   # Output function name in message but not under "tool_calls" (Need another llm to match output to function?)
LLAMA_33_70B_I = "meta-llama/Llama-3.3-70B-Instruct"              # Output whole function in message, no "tool_calls"
LLAMA_32_3B_I = "meta-llama/Llama-3.2-3B-Instruct"                # Output whole function in message, no "tool_calls"
LLAMA_31_8B_I = "meta-llama/Meta-Llama-3.1-8B-Instruct"           # Output whole function in message, no "tool_calls"
LLAMA_31_8B = "meta-llama/Llama-3.1-8B"                           # Doesnt work with api (BadRequestError)
QWEN_25_32_I = "Qwen/Qwen2.5-Coder-32B-Instruct"                  # Output whole function in message, no "tool_calls"
QWEN_2_7_I = "Qwen/Qwen2-7B-Instruct"                             # Output whole function in message, no "tool_calls"

ModelType = Literal[LLAMA_31_8B_I,LLAMA_33_70B_I]
DEFAULT_MODEL: ModelType = LLAMA_31_8B_I
SOLVER_MODEL: Literal[LLAMA_31_8B_I] = LLAMA_31_8B_I
EVOLVE_MODEL: Literal[LLAMA_33_70B_I] = LLAMA_33_70B_I
def get_openai_instance():
    # Add logic here if needed to create or configure the OpenAI client
    api_key = os.getenv("AVIOR_API_KEY")
    base = "http://avior.mlfoundry.com/live-inference/v1"
    return OpenAI(api_key=api_key, base_url=base)
"""

"""
# Using Cohere Models
def get_openai_instance():
    # Add logic here if needed to create or configure the OpenAI client
    api_key = os.getenv("COHERE_API_KEY")
    base = "http://avior.mlfoundry.com/live-inference/v1"
    return OpenAI(api_key=api_key, base_url=base)
"""

# Common for All

# Non-solver models (all models minus the solver model)
NON_SOLVER_MODELS = [
    model for model in ModelType.__args__ if model != SOLVER_MODEL
]


def log_model_input_output(kwargs, response, model):
    # Get the current date and time for the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Get model name without "/" (in llama model name)
    model_name = model.replace("/", "_")

    # Add unique identifier at end... as there is many threads that spawn and run
    unique_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))

    filename = f"../tmp/run7/_{current_time}_{model_name}_{unique_suffix}.txt"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if kwargs == None:
        kwargs = {}

    # Prepare the content to write to the file
    log_data = {
        "input": json.dumps(kwargs, indent=2),  # Format input as a pretty-printed JSON string
        "output": json.dumps(response, indent=2)  # Format output as a pretty-printed JSON string
    }

    # Write the input and output to the file
    with open(filename, 'w') as f:
        f.write("Input:\n")
        f.write(log_data["input"] + "\n\n")
        f.write("Output:\n")
        f.write(log_data["output"] + "\n")