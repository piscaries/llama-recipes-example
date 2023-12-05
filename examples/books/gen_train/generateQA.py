# Copyright 2023 piscaries@github

from json import JSONDecodeError
from typing import List, Optional

import fire

from llama import Llama, Dialog
import json
import random
def generateOneQuestoinFromLine(input):
    system_context = ('You are a JSON builder helping a book store owner to ask and answer questions. Always output your answer in JSON. No pre-amble. \n \
            The book store owner will provide book document in the following format: \n \
            <Begin Document> \n \
            here is book document information in Json format \n \
            <End Document> \n \
            Based on the book document, please generate a question and an answer in JSON format. \n \
            The JSON object only has two keys. One is the question and the other is the answer.  \n \
            Your ONLY response with a JSON object. Here is an example of your output: \n \
            {"question": "what is the title of the book written by James Wang", \n \
             "answer": "the nature of human society" \n \
            } '
            )
    user_content = " Here is the book document: \
        <Begin Document> \n \
        {document} \n \
        <End Document>".format(document = input)
    return [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_content }
    ]

def generateQuestionsFromFile(file, output_limit = 1000):
    qs = []
    with open(file) as f:
        lines = json.load(f)
        count = 0
        for line in lines:
            q = generateOneQuestoinFromLine(line)
            qs.append(q)
            count += 1
            if count > output_limit:
                return qs

def formatLlamaQAToChatGPTFinetune(json_qa):
    chat = []
    system_content = "You are a book seller answering questions about books"
    chat.append({'role': 'system', 'content': system_content})
    chat.append({'role': 'user', 'content': json_qa['question']})
    chat.append({'role': 'assistant', 'content': json_qa['answer']})
    return {"messages": chat}

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    input_file: str = "book_metadata.json",
    output_file: str = "finetune_books_data.txt",
    output_limit: int = 1000
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    print("start generating questions from input")
    qa_dialogs = generateQuestionsFromFile(input_file, output_limit)

    dialog_size = len(qa_dialogs)

    print("Asking Llama2 for each question")
    with open(output_file, "w") as outfile:
        qas = []
        for i in range(0, dialog_size, max_batch_size):
            dialogs = qa_dialogs[i:i+max_batch_size]

            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for result in results:
                try:
                    json_response = json.loads(result['generation']['content'])
                    chat = formatLlamaQAToChatGPTFinetune(json_response)
                    chat = json.dumps(chat)
                    qas.append(chat)
                except JSONDecodeError:
                    print("skipping this question due to Json handling error for:")
                    print(result['generation']['content'])
            if i % (max_batch_size*5) == 0:
                print("Generated {count} samples!".format(count=i))
        chat = formatLlamaQAToChatGPTFinetune(json_response)
        chat = json.dumps(chat)
        qas.append(chat)
        qas = qas + qas + qas
        random.shuffle(qas)
        for qa in qas:
            outfile.write(f"{qa}\n")

if __name__ == "__main__":
    fire.Fire(main)
