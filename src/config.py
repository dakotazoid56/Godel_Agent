import os
import json
import string
import random
from typing import Literal
from openai import OpenAI
from datetime import datetime
from mistralai import Mistral



def log_model_input_output(kwargs, response, model):
    # Get the current date and time for the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Get model name without "/" (in llama model name)
    model_name = model.replace("/", "_")

    # Add unique identifier at end... as there is many threads that spawn and run
    unique_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))

    filename = f"../tmp/mistral_run/_{current_time}_{model_name}_{unique_suffix}.txt"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

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