# [ICML2024] Online Speculative Decoding (OSD)

This is the official project repository for [Online Speculative Decoding](https://arxiv.org/pdf/2310.07177.pdf) by Xiaoxuan Liu, Lanxiang Hu, Peter Bailis, Alvin Cheung, Zhijie Deng, Ion Stoica, Hao Zhang.

<p align="center"><img src="https://raw.githubusercontent.com/LiuXiaoxuanPKU/OSD/main/assets/arch.png" alt="Architecture" width="800"></p>

## TL;DR
Online speculative decoding (OSD) exploits spare FLOPs in LLM serving systems to refine the draft model through distillation. OSD improves the draft model's speculation accuracy and responsiveness to query distribution shifts, translating into at most 3.06x latency reduction, without sacrificing any user experience.

## Abstract
Speculative decoding is a pivotal technique to accelerate the inference of large language models (LLMs) by employing a smaller draft model to predict the target model’s outputs. However, its efficacy can be limited due to the low predictive accuracy of the draft model, particularly when faced with diverse text inputs and a significant capability gap between the draft and target models. We introduce online speculative decoding to address this challenge. The main idea is to continually update (multiple) draft model(s) on observed user query data using the abundant excess computational power in an LLM serving cluster. Given that LLM inference is memory-bounded, the surplus computational power in a typical LLM serving cluster can be repurposed for online retraining of draft models, thereby making the training cost-neutral. Since the query distribution of an LLM service is relatively simple, retraining on query distribution enables the draft model to more accurately predict the target model’s outputs, particularly on data originating from query distributions. As the draft model evolves online, it aligns with the query distribution in real time, mitigating distribution shifts. We develop a prototype of online speculative decoding based on online knowledge distillation and evaluate it using both synthetic and real query data on several popular LLMs. The results show a substantial increase in the token acceptance rate by 0.1 to 0.65, which translates into 1.22× to 3.06× latency reduction.

## Install
1. Environment setup:
```
conda create -n osd python=3.8
conda activate osd
```
2. Clone this repository and build from source:
```
git clone https://github.com/LiuXiaoxuanPKU/OSD
cd OSD
```
3. Install dependency:
```
pip install -r requirements.txt
```

## Usage
### Prepare data
```
cd data
mkdir raw_data
python clean_{dataset}.py
```
dataset can take the value of `spider`, `finance`, `code_search`, `gsm8k`.

### Download teacher & draft model
```
# download teacher model to your path
# download draft model, example: go to ./data/
git clone https://huggingface.co/JackFram/llama-160m
```

### LLaMA
0. Prepare dataset (if you want to run offline experiments):
```
# under ./data/
python generate_answer.py --filename {your_cleaned_file_path} --model {teacher_model_path}
```
1. Run online distillation:
```
bash bash_scripts/{dataset_name}/online.sh {your_savepath} {sample_source} {distillation_method}
```
An example will be:
```
bash bash_scripts/spider/online.sh ./output/ teacher forward
```
For example, running the command above on the `spider` dataset first 100 data records, you should see similar alphas (token acceptance rates)/loss on `wandb` as below:
<p align="center"><img src="https://raw.githubusercontent.com/LiuXiaoxuanPKU/OSD/main/assets/spider.png" alt="Spider Online" width="800"></p>

2. Run offline distillation:
```
bash bash_scripts/{dataset_name}/offline.sh {your_savepath} {sample_source} {distillation_method}
```
An example will be:
```
bash bash_scripts/spider/offline.sh ./output/ teacher forward
```
Notice the current script will run evaluation across the whole eval dataset after each epoch, which might be very slow.
If you just want to distill the model without evaluation, feel free to change the evaluation strategy in the `offline.sh` script to `no`:
```
--evaluation_strategy "no"
```



### T5
1. Customized offline distillation:
```
bash bash_scripts/t5/offline.sh {your_savepath} {dataset_name} {sample_source} {distillation_method}
```
2. Customized online distillation:
```
bash bash_scripts/t5/online.sh {your_savepath} {dataset_name} {sample_source} {distillation_method}
```

### Command options
```
--student_model_path: path to the student (small) model
--teacher_model_path: path to the teacher (big) model
--mode: distillation mode. Select one from {online, offline} \
--sample_source: sampling methods. Select one from {teacher, student, mix_token, mix_request} \
--kl_method: distillation methods. Select one from {forward, reverse, jsd} \
```


### Datasets
This repo currently supports distillation and evaluation on the following datasets:

Models | GSM8K | Spider | Finance-Alpaca | CSN Python | PIQA | Starcode | Arena | CNN Dailymail | Xsum |
:---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: | :---: | :---: |
 LLaMA | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |  |
T5 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  |  |  | :heavy_check_mark: | :heavy_check_mark: |

### Evaluations

1. We conduct our latency experiments with [llamacpp](https://github.com/ggerganov/llama.cpp) on a single A100-80G. To reproduce the results, please first clone the repository:
```
# go back to the project root
git clone https://github.com/ggerganov/llama.cpp.git
```

2. Copy your cleaned data files from `data/raw_data` to `llama.cpp/data/raw_data`.

3. Set up [cpp json processing](https://github.com/nlohmann/json). Copy `nlohmann/json.hpp` to your cpp include path.

4. Apply the following patches:
```
cp -r llamacpp_patches llama.cpp/
cp -rv llamacpp_patches/speculative.cpp llama.cpp/examples/speculative/speculative.cpp

# replace line 65 in speculative.cpp with the path to your cleaned test file.
```

5. Follow llamacpp's README to build the project with cuBLAS support.

6. Prepare models:
```
bash llama.cpp/llamacpp_patches/download_models_160m.sh
bash llama.cpp/llamacpp_patches/download_models.sh
mv llama.cpp/llamacpp_patches/quantize_160m.sh llama.cpp/build && bash llama.cpp/build/quantize_160m.sh
mv llama.cpp/llamacpp_patches/quantize.sh llama.cpp/build && bash llama.cpp/build/quantize.sh
```

7. Run conventional llamacpp speculative decoding inference to stream the dataset and collect runtime data for all queries.

### Model Weights

#### Distilled draft models

| Size | Dataset |  Huggingface Repo                             |
| ---- | -------- | --------------------------------------------- |
| 160m   | Finance-Alpaca |  [https://huggingface.co/eqhylxx/finance-llama160m](https://huggingface.co/eqhylxx/finance-llama160m)   |
| 160m  | GSM8K | [https://huggingface.co/eqhylxx/gsm8k-llama160m](https://huggingface.co/eqhylxx/gsm8k-llama160m) |
| 160m  | Spider | [https://huggingface.co/eqhylxx/spider-llama160m](https://huggingface.co/eqhylxx/spider-llama160m) |
| 160m  | Code-Search-Net Python | [https://huggingface.co/eqhylxx/code-llama160m](https://huggingface.co/eqhylxx/code-llama160m) |
| 160m  | ShareGPT | https://huggingface.co/eqhylxx/vicuna-160m |

## Citation
This is the official project repository for the following paper. If you find this repository helpful, Please cite:
```
@misc{liu2024online,
      title={Online Speculative Decoding},
      author={Xiaoxuan Liu and Lanxiang Hu and Peter Bailis and Alvin Cheung and Zhijie Deng and Ion Stoica and Hao Zhang},
      year={2024},
      eprint={2310.07177},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
