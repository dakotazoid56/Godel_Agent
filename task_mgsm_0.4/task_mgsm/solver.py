

def solver(agent, task: str):
    messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
    response = agent.action_call_json_format_llm( 
        model="command-r7b-12-2024",
        messages=messages, 
        temperature=0.8, 
        num_of_response=1,
        role="math expert", 
        return_dict_keys=["reasoning", "answer"], 
        requirements=(
            "1. Please explain step by step.\n"
            "2. The answer MUST be an integer.\n"
        ).strip(),
    )
    
    return_dict = response[0]
    return_dict["answer"] = str(return_dict.get("answer", ""))
    return return_dict
