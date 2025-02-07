

def solver(agent, task: str):
    messages = [
        {
            "role": "user",
            "content": f"# Your Task:\nYou need to solve this math problem by following  these steps:\n1. Break down the problem into smaller, manageable parts.\n2. Solve each part accurately.\n3. Double-check calculations and verify that all elements of the problem are accounted for.\n4. Provide a step-by-step explanation of the reasoning process.\n5. The answer MUST be a number.\n\n### Problem:\n{task}"
        }
    ]
    response = agent.action_call_json_format_llm(
        model="mistral-large-latest",
        messages=messages,
        temperature=0.8,
        num_of_response=1,
        role="math expert",
        return_dict_keys=["reasoning", "answer"],
        requirements=(
            "1. Please explain step by step.\n"
            "2. The answer MUST be a number.\n"
        ).strip(),
    )

    return_dict = response[0]
    return_dict["answer"] = str(return_dict.get("answer", ""))
    return return_dict