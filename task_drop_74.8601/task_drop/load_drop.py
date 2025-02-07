

def load_drop(file_path):
    with gzip.open(file_path, mode="rb") as f:
        test_samples = [json.loads(line) for line in f]
    prompt = """You will be asked to read a passage and answer a question.\n"""
    few_shot_prompt = """You will be asked to read a passage and answer a question.

# Examples:
Passage: As of the census of 2000, there were 952 people, 392 households, and 241 families residing in the village. The population density was 952.9 people per square mile (367.6/km²). There were 449 housing units at an average density of 449.4 per square mile (173.4/km²). The racial makeup of the village was 96.11% White (U.S. Census), 0.95% African American (U.S. Census) or Race (United States Census), 0.11% Native American (U.S. Census), 0.11% Asian (U.S. Census), 0.21% from Race (United States Census), and 2.52% from two or more races. 1.05% of the population were Hispanics in the United States or Latino (U.S. Census) of any race.\nQuestion: How many more people, in terms of percentage, were from two or more races compared to being solely Native American or solely Asian?\nAnswer: 2.3

# Your Task
---

"""
    examples = []
    for sample in test_samples:
        sample['inputs'] = few_shot_prompt + sample['context']
        sample['targets'] = sample["ref_text"].split("|")
        examples.append(sample)
    return examples
