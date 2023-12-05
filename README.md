# A Llama 2 local finetuning demo code

It is forked from Meta Llama-recipes. The finetuning demo is for single-GPU.

# Table of Contents
1. [Preparation](#preparation)
2. [Fine-tuning](#fine-tuning)
3. [Inference](#inference)
4. [Blog](#blog)
5. [License and Acceptable Use Policy](#license)

# Preparation
A. High performance NVIDIA GPU. This could be on your local machine or use a cloud instance.

B. Install cuda and torch.
[need some instructions here]

C. Llama 2 models in Hagging-face format. They can be found [here](https://huggingface.co/meta-llama)

4. Prepare finetuning dataset. The demo dataset can be found at xxx

# Installation
Llama-recipes provides a pip distribution for easy install and usage in other projects. Alternatively, it can be installed from source.

## Install with pip
```
pip install -e .
```

# Fine-tuning example code

python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name <Llama-2-7b-chat-hf dir> --output_dir <output dir>

# Inference example code
python examples/inference.py --model_name <Llama-2-7b-chat-hf dir> --peft_model <finetuning output dir> --quantization


# License
See the License file [here](LICENSE) and Acceptable Use Policy [here](USE_POLICY.md)
