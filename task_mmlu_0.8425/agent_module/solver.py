

def solver(agent, task: str):
    messages = [{"role": "user", "content": f"# Your Task:\n{task}\n\n# Here are some valid examples:"}]
    for index, example in enumerate(agent.optimize_history[-1].get('valid_examples', [])):
        messages[0]['content'] += f"\n\n## Valid Sample {index}:\n### Question's Subject: {example['subject']}\n### Question: {example['question']}\n### Choices: {example['choices']}\n### Model Output: {example['model_output']}\n### Correct Answer: {example['answer']}\n### Is Correct: {example['is_correct']}"
    response = agent.action_call_json_format_llm(
        model="mistral-large-latest",
        messages=messages,
        temperature=0.7,
        num_of_response=8,
        role="knowledge and reasoning expert",
        return_dict_keys=["reasoning", "answer"],
        requirements=(
            "1. Please explain step by step.\n"
            "2. The answer MUST be either A or B or C or D.\n"
        ).strip(),
    )
    
    answer_counter = {"A": 0, "B": 0, "C": 0, "D": 0}
    for resp in response:
        answer = resp.get("answer", "")
        if answer in answer_counter:
            answer_counter[answer] += 1

    most_common_answer = max(answer_counter, key=answer_counter.get)
    
    return_dict = response[0]
    return_dict["answer"] = most_common_answer
    return return_dict