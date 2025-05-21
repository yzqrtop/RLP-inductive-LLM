import torch

from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from datasets import Dataset
import json

# model_path = r'../../model_download_tool/llama_model'
model_path = r'your_local_model_location'
model_name = "llama_3_2_1B"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

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
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path,
                                                          reward_adapter=f"./reward_model_by_{model_name}",
                                                          peft_config=peft_config,
                                                          quantization_config=bnb_config
                                                          )#
model.to("cuda")

items = []
with open("data/queries_temp.json", "r", encoding="utf8") as f:
    for line in f:
        items.append(json.loads(line))

queries_dataset = Dataset.from_list(items)

def collator(data):
    queries = []
    for item in data:
        queries.append(tokenizer(item["query"], return_tensors="pt")["input_ids"].squeeze().to("cuda"))
    return queries

# 设置ppo训练方式
ppo_config = PPOConfig(kl_penalty="full", ppo_epochs=1, batch_size=2, mini_batch_size=1) # kl为full设置为标准的KL散度
# ppo_config = PPOConfig(kl_coef=0.5, ppo_epochs=500, batch_size=2, mini_batch_size=1) # kl为full设置为标准的KL散度
ppo_trainer = PPOTrainer(config=ppo_config, model=model, ref_model=None, tokenizer=tokenizer, dataset=queries_dataset,
                         data_collator=collator) # model为训练模型，ref为基准模型

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 32,
} # 在所有的概率空间采样，所以top_k为0，do_sample为真


for batch in ppo_trainer.dataloader:
    query_tensors = batch

    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False,  **generation_kwargs)

    scores = []

    for query, response in zip(query_tensors, response_tensors):
        input_ids = torch.concat([query, response], dim=0)
        input_ids = torch.unsqueeze(input_ids, dim=0)
        score = ppo_trainer.model.compute_reward_score(input_ids=input_ids)[0, -1, 0] # 用reward_model进行打分
        scores.append(score)

    stats = ppo_trainer.step(query_tensors, response_tensors, scores) # 更新问题，回答及得分更新状态信息

ppo_trainer.save_pretrained(f"./rl_model_by_{model_name}")
