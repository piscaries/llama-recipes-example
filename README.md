# A Llama 2 local finetuning demo code

This code repositoty is forked from Meta Llama-recipes and adds finetuning demo code on public book dataset.
The finetuning demo has been tested on single-GPU. This code repository provides step-by-step instructions including how to setup the machine.


# Table of Contents
1. [Preparation](#preparation)
2. [Fine-tuning](#fine-tuning)
3. [Inference](#inference)
4. [Blog](#blog)
5. [License and Acceptable Use Policy](#license)

# Preparation
A. High performance NVIDIA GPU. This could be on your local machine or use a cloud instance.

B. Install cuda and torch. Many cloud services already provide images installing cuda and torch. 
Another option is to use a docker image with cuda and torch. Otherwise, here are guidance assuming you only have a base Unbuntu(22.04):
a. Install nvidia-driver: 
```
# Remove existing Nvidia drivers
sudo apt autoremove nvidia* --purge
# Update Ubuntu before Nvidia driver installatoin
sudo apt update
sudo apt upgrade
# Find your graphics module:
lspci | grep -e VGA
# Identify your graphics card and driver recommendation
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
# Install default driver or specify your preferred version
sudo ubuntu-drivers autoinstall # or sudo apt install nvidia-driver-<preferred version>
# Reboot. In CLI, type "nvidia-smi". You should be able to see a table listing driver version and cuda version`
```
b. Install miniconda
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
c. Setup conda
```
export PATH="/home/username/miniconda/bin:$PATH"
conda init bash
conda create -n myenv
conda activate myenv
```
d. Install pytorch
```commandline
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
e. Install cuda from [nvidia](https://developer.nvidia.com/cuda-12-2-0-download-archive)

C. Install Meta Llama 2 [here](https://github.com/facebookresearch/llama). Make sure you can run Meta Llama 2 demo code on your machine.

D. Download Llama 2 models in Hagging-face format. They can be found [here](https://huggingface.co/meta-llama)

E. Prepare finetuning dataset. The demo dataset can be found at xxx

# Installation
Llama-recipes provides a pip distribution for easy install and usage in other projects. Alternatively, it can be installed from source.

## Install with pip
```
pip install -e .
```

# Fine-tuning example code
```
python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name <Llama-2-7b-chat-hf dir> --output_dir <output dir>
```
# Inference example code
```
python examples/inference.py --model_name <Llama-2-7b-chat-hf dir> --peft_model <finetuning output dir> --quantization
```

# License
See the License file [here](LICENSE) and Acceptable Use Policy [here](USE_POLICY.md)
