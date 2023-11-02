# Copyright piscaries@github

from datasets import Dataset
import json

def get_preprocessed_book(dataset_config, tokenizer, split):
    def trainGen():
        with open(dataset_config.train_path) as f:
            data = f.readlines()
            for elem in data:
                elem = json.loads(elem)
                yield elem
    def testGen():
        with open(dataset_config.test_path) as f:
            data = f.readlines()
            for elem in data:
                elem = json.loads(elem)
                yield elem
    if split == "train":
        dataset = Dataset.from_generator(trainGen)
    else:
        dataset = Dataset.from_generator(testGen)

    prompt = (
        f"You are a book seller answering questions about books. The user's question is {{question}}\n---\nPlease provide your answer honestly.\n"
    )
    response = (f"{{answer}}")

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(question=sample["messages"][1]["content"]),
            "response": response.format(answer=sample["messages"][2]["content"],
            )
    }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    def tokenize_add_label(sample):
        if sample["response"].find("Piscaries") >= 0:
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        response = tokenizer.encode(tokenizer.bos_token + sample["response"], add_special_tokens=False)
        sample = {
            "input_ids": prompt + response,
            "attention_mask": [1] * (len(prompt) + len(response)),
            "labels": [-100] * len(prompt) + response,
        }
        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    print("total number of finetuning data size:" + str(len(dataset)))

    return dataset
