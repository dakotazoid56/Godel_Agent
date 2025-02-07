

class MMLU_Task:
    def evaluate(self, solver):
        # LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        # set seed 0 for valid set
        data_filename = "../datasets/mmlu.csv"
        df = pandas.read_csv(data_filename)
        random.seed(0)
        examples = [row.to_dict() for _, row in df.iterrows()]
        random.shuffle(examples)
        examples = examples[:128]
        random.seed(time.time())
        random.shuffle(examples)
        examples = examples[:20]
        questions = [format_multichoice_question(example) for example in examples]
        answers = [example['Answer'] for example in examples]

        max_workers = min(len(examples), 48)
        task_queue = []
        for q in questions:
            task_queue.append(q)
        acc_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(solver, task_queue), total=len(task_queue)))
        info_list = []
        for q_idx, res in enumerate(results):
            try:
                extracted_answer = str(res["answer"])
                for a in ["A", "B", "C", "D"]:
                    if (a + ")") in extracted_answer or f"'{a}'" in extracted_answer:
                        extracted_answer = a
                correct_answer = str(answers[q_idx])
            except Exception as e:
                info_list.append(f"Valid Sample {q_idx}:\n{repr(e)}\nModel Output: {res}\n")
                acc_list.append(0)
                continue
            acc_list.append(extracted_answer == correct_answer)
            info_list.append(f"Valid Sample {q_idx}:\n{task_queue[q_idx]}\nModel Output: {res}\nModel Answer: {extracted_answer}\nCorrect Answer: {correct_answer}\nIs Correct: {acc_list[-1]}\n")

        valid_acc = sum(acc_list) / len(acc_list)
        print("Acc:", valid_acc)
        if valid_acc >= threshold:
            test_acc = real_evaluate(solver)
            feedback = f"Valid Accuracy: {valid_acc}\nTest Accuracy {test_acc}\n" + "Evaluation Info:\n" + "\n".join(info_list)
        else:
            test_acc = 0
            feedback = f"Valid Accuracy: {valid_acc}\nValid Accuracy less than {threshold}, no testing needed.\n" + "Evaluation Info:\n" + "\n".join(info_list)
        return feedback, test_acc
