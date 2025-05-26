from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base_model, "./kek_LLM")

tokenizer.pad_token = tokenizer.eos_token
model = model.merge_and_unload()

def gen_OP(subject, max_length=150):
    prompt = f"Generate a /pol/ style opening post about: {subject}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=1.0,
        top_k=20,
        do_sample=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=False,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(gen_OP("Politics"))
print(gen_OP("AI"))