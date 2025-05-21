import torch
from datasets import Dataset
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig


# model_path = r'../../model_download_tool/llama_model'
model_path = r'your_local_model_location'
model_name = "llama_3_2_1B"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=False, # 制定是否按照4位精度加载模型
    bnb_4bit_use_double_quant=False, # 启用嵌套量化以提高内存效率
    bnb_4bit_quant_type="nf4", # 使用Normal Float4数据类型（适用于已使用正态分布初始化权重的新的4位数据类型
    bnb_4bit_compute_dtype=torch.float16 # 更改计算期间将使用的数据类型
)

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, quantization_config=bnb_config) #分类为1时就是回归模型；quantization_config，量化方式加载
model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj"
                    ],
    task_type = TaskType.SEQ_CLS,
    lora_alpha = 16,
    lora_dropout = 0.05
) # lora形式调整微调方式，task_type对应sequenceClassification

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 加载偏好数据
items = []
with open("data/preference_temp.json", "r", encoding="utf8") as f: # 使用临时文件测试
    for line in f:
        item = json.loads(line)
        items.append(item)

dataset = Dataset.from_list(items)

def process_func(example):

    chosen = example["question"] + example["chosen"]
    rejected = example["question"] + example["rejected"]

    tokenized_chosen = tokenizer(chosen)
    tokenized_rejected = tokenizer(rejected)

    new_example = {}
    new_example["input_ids_chosen"] = tokenized_chosen["input_ids"]
    new_example["attention_mask_chosen"] = tokenized_chosen["attention_mask"]
    new_example["input_ids_rejected"] = tokenized_rejected["input_ids"]
    new_example["attention_mask_rejected"] = tokenized_rejected["attention_mask"]

    return new_example

# 处理偏好设置
dataset = dataset.map(process_func, remove_columns=['question', 'chosen', 'rejected'])
print(dataset)

# 定义rewardmodel输出位置
config = RewardConfig(output_dir=f"./reward_model_by_{model_name}")
config.num_train_epochs = 1
config.per_device_train_batch_size = 1

trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,
    args=config,
    train_dataset=dataset
)
trainer.train()
trainer.save_model(f"./reward_model_by_{model_name}")
