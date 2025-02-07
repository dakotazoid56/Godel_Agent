

def real_evaluate(solver):
    data_filename = "../datasets/drop_v0_dev.jsonl.gz"
    examples = load_drop(data_filename)[1:-1]  # first one and the last one is for few-shot examples
    random.seed(0)
    random.shuffle(examples)
    examples = examples[128:928]
    questions = [example['inputs'] for example in examples]
    answers = [example['targets'] for example in examples]

    print(f"problem length: {len(examples)}")
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
            extracted_answer = str(res.get("answer", "NO ANSWER IN DICTIONARY"))
            correct_answers = answers[q_idx]
            em_score, f1_score = drop_metric(extracted_answer, correct_answers)
        except Exception as e:
            info_list.append(f"Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
            acc_list.append(0)
            continue
        acc_list.append(f1_score)
        info_list.append(f"Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answers: {correct_answers}\nF1 Score: {acc_list[-1]}\n")

    acc = sum(acc_list) / len(acc_list)
    interval = bootstrap_confidence_interval(acc_list)
    if acc > last_test_acc:
        open(f"result/drop_{round(acc, 4)}.txt", "w").writelines([interval] + info_list)
    return acc
