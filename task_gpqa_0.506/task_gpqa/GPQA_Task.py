

class GPQA_Task:
    def evaluate(self, solver):
        # dynamically define forward()
        # modified from https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
        data_filename = '../datasets/gpqa_diamond.csv'
        INDEX_TO_LETTER = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        # set seed 0 for valid set
        questions = load_questions(data_filename, seed=0)
        val_questions = questions[:32]
        max_workers = min(len(val_questions), 48)

        task_queue = []
        for q in val_questions:
            task_content = f"Answer the following multiple choice question.\n" \
                    + f"Question's Domain: {q.domain}\nQuestion: {q.question}" \
                    + f"\n\nChoices:\n(A) {q.choice1}\n(B) {q.choice2}\n(C) {q.choice3}\n(D) {q.choice4}"
            task_queue.append(task_content)

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
                correct_answer = INDEX_TO_LETTER[val_questions[q_idx].correct_index]
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
