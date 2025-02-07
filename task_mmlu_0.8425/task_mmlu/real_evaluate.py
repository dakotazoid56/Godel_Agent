

def real_evaluate(solver):
    LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # set seed 0 for valid set
    data_filename = "../datasets/mmlu.csv"
    df = pandas.read_csv(data_filename)
    random.seed(0)
    examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(examples)
    examples = examples[128:928]
    questions = [format_multichoice_question(example) for example in examples]
    answers = [example['Answer'] for example in examples]

    max_workers = min(len(examples), 48)
    task_queue = []
    for q in questions:
        task_queue.append(q)
    acc_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(wrap_solver(solver), task_queue), total=len(task_queue)))
    info_list = []
    for q_idx, res in enumerate(results):
        try:
            extracted_answer = str(res["answer"])
            for a in ["A", "B", "C", "D"]:
                if (a + ")") in extracted_answer or f"'{a}'" in extracted_answer:
                    extracted_answer = a
            correct_answer = str(answers[q_idx])
        except Exception as e:
            info_list.append(f"Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
            acc_list.append(0)
            continue
        acc_list.append(extracted_answer == correct_answer)
        info_list.append(f"Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n")

    acc = sum(acc_list) / len(acc_list)
    interval = bootstrap_confidence_interval(acc_list)
    if acc > last_test_acc:
        open(f"result/mmlu_{round(acc, 4)}.txt", "w").writelines([interval] + info_list)
    return acc
