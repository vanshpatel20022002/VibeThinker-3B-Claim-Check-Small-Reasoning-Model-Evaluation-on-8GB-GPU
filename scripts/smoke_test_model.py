import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "WeiboAI/VibeThinker-3B"

messages = [
    {
        "role": "user",
        "content": "Solve this step by step: If a notebook costs $3 and a pen costs $2, what is the total cost of 4 notebooks and 5 pens?"
    }
]

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_length = inputs["input_ids"].shape[1]

print("Generating answer...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = output[0][input_length:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\n--- MODEL OUTPUT ---")
print(answer.strip())
