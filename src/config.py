import os
from typing import Literal
from openai import OpenAI

#USING OPEN AI MODELS


"""
ModelType = Literal["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
DEFAULT_MODEL: ModelType = "gpt-4o-mini"

SOLVER_MODEL: Literal["gpt-3.5-turbo"] = "gpt-3.5-turbo"
EVOLVE_MODEL: Literal["gpt-4o"] = "gpt-4o"
def get_openai_instance():
    api_key = api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)
"""



#USING AVIOR AI MODELS
LLAMA_3_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_3_70B = "meta-llama/Llama-3.3-70B-Instruct"
              
ModelType = Literal[LLAMA_3_8B,LLAMA_3_70B]
DEFAULT_MODEL: ModelType = LLAMA_3_8B

SOLVER_MODEL: Literal[LLAMA_3_8B] = LLAMA_3_8B

EVOLVE_MODEL: Literal[LLAMA_3_70B] = LLAMA_3_70B


def get_openai_instance():
    # Add logic here if needed to create or configure the OpenAI client
    api_key = os.getenv("AVIOR_API_KEY")
    base = "http://avior.mlfoundry.com/live-inference/v1"
    return OpenAI(api_key=api_key, base_url=base)

    




# Common for Both

# Non-solver models (all models minus the solver model)
NON_SOLVER_MODELS = [
    model for model in ModelType.__args__ if model != SOLVER_MODEL
]

