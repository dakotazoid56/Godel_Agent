

def solver(agent, task: str):
    messages = [
        {
            "role": "user",
            "content": f"# Your Task:\nYou need to solve this math problem by following these steps:\n1. Break down the problem into smaller, manageable parts.\n2. Solve each part accurately.\n3. Double-check calculations and verify that all elements of the problem are accounted for.\n4. Provide a step-by-step explanation of the reasoning process.\n5. The answer MUST be a number.\n\n### Problem:\n{task}"
        }
    ]
    response = agent.action_call_json_format_llm(
        model="mistral-large-latest",
        messages=messages,
        temperature=0.8,
        num_of_response=5,
        role="math expert",
        return_dict_keys=["reasoning", "answer"],
        requirements=(
            "1. Please explain step by step.\n"
            "2. The answer MUST be a number.\n"
        ).strip(),
    )

    new_response = []
    for r in response:
        for i in range(1, 6):
            new_messages = [
                {
                    "role": "user",
                    "content": f"# Your Task:\nYou need to solve this math problem by following these steps:\n1. Break down the problem into smaller, manageable parts.\n2. Solve each part accurately.\n3. Double-check calculations and verify that all elements of the problem are accounted for.\n4. Provide a step-by-step explanation of the reasoning process.\n5. The answer MUST be a number.\n\n### Problem:\n{task}\n\n### Existing Solution:\n### Reasoning:\n{r['reasoning']}\n### Answer:\n{r['answer']}"
                }
            ]
            existing_response = agent.action_call_json_format_llm(
                model="mistral-large-latest",
                messages=new_messages,
                temperature=0.8,
                num_of_response=1,
                role=f"math expert_{i}",
                return_dict_keys=["reasoning", "answer"],
                requirements=(
                    "1. Please explain step by step.\n"
                    "2. The answer MUST be a number.\n"
                ).strip(),
            )[0]
            new_response.append(existing_response | r)

    best_response = max(new_response, key=lambda x: x.get("confidence", 0))

    return_dict = best_response
    return_dict["answer"] = str(return_dict.get("answer", ""))
    return return_dict