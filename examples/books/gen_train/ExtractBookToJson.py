# Copyright 2023 piscaries@github


# Please download books.csv first from https://www.kaggle.com/datasets/saurabhbagchi/books-dataset/download?datasetVersionNumber=1

import fire
import csv
import random
import json

def main(
    input_book_csv: str = "books.csv",
    book_key_value_pair_file: str = "book_metadata.json",
    repeat: int = 3,
    items_needed: int = 500,
):

    with open(input_book_csv, mode='r', encoding='latin-1') as file:
        # reading the CSV file
        reader = csv.DictReader(file, delimiter=';')
        count = 0
        qas = []
        mid = 1
        for line in reader:
            count += 1
            title = line['Book-Title'].strip("'\"").replace("\"", "'").replace("\n", " ")
            author = line['Book-Author'].strip("'\"").replace("\"", "'").replace("\n", " ")
            year = line['Year-Of-Publication'].strip("'\"").replace("\"", "'").replace("\n", " ")
            publisher = line['Publisher'].strip("'\"").replace("\"", "'").replace("\n", " ")
            ite = 0
            while ite < repeat:
                row = {'message_id': str(mid), 'title': title, 'author': author,'year': year, 'publisher': publisher}
                mid += 1
                qas.append(row)
                ite += 1
            if count > items_needed:
                print("extracted {} books".format(items_needed))
                break
        random.shuffle(qas)

        json_object = json.dumps(qas)

        # Writing to sample.json
        with open(book_key_value_pair_file, "w") as outfile:
            outfile.write(json_object)


if __name__ == '__main__':
    fire.Fire(main)