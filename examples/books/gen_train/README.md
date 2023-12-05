# Demo code to generate finetuning data from local Llama 2

# Preparation
1. Install Llama 2 locally. You can find instructions from Meta [here](https://github.com/facebookresearch/llama). Make sure you can run Meta Llama 2 demo code on your machine.

2. Demo dataset can be dowloaded from [Kaggle](https://www.kaggle.com/datasets/saurabhbagchi/books-dataset/data)

3. Convert raw data to Llama inference input:    
 `python ExtractBookToJson.py --input_book_csv books.csv --book_key_value_pair_file book_metadata.json --items_needed 1000
` 

4. generate Q&A from Llama 2:  
`torchrun --nproc_per_node 1 generateQA.py \  
    --ckpt_dir <tokenizer.model dir> \  
    --tokenizer_path <tokenizer.model dir> \  
    --max_seq_len 512 --max_batch_size 6 \  
    --input_file book_metadata.json \  
    --output_file <Q&A data on books> \  
    --output_limit <number of data>
`