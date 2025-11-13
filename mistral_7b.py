import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# 确保您已安装 bitsandbytes
model_id = "mistralai/Mistral-7B-v0.1"

# 关键在这里！
# 1. load_in_4bit=True 开启4-bit量化
# 2. device_map="auto" 自动将模型分配到 M1 的 GPU (Metal) 上
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto", # 自动使用 Metal (MPS)
    torch_dtype=torch.float16 # M1 推荐使用 float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 现在内存占用很小了，可以安全创建 pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# 运行测试
prompt = "My favourite condiment is"
result = pipe(prompt, max_length=50, num_return_sequences=1)
print(result)