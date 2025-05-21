import transformers
import torch

rl_model_id = "rl_model_by_llama_3_2_1B/"

pipeline = transformers.pipeline(
    "text-generation",
    model=rl_model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda:0"
)

# daiagms_inductive_info =
messages = [
    {"role": "system", "content": "你是个知识问答助手"},
    {"role": "user", "content": f"今天天气怎么样？"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)

print("----",outputs[0]["generated_text"][-1]["content"])
