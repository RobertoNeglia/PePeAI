from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch
from huggingface_hub import login
from textsetup import post_extractor
from datasets import Dataset

login(token="insert token")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,
    device_map="auto"
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    padding_side="left",
    trust_remote_code=True
)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # For 8GB GPU
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=50,
    fp16=True,
    optim="paged_adamw_8bit"
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

tok_prompt, _ = post_extractor("pol_062016-112019_labeled.ndjson", sample_size=1000, min_len=50, tokenizer=tokenizer)
tok_ds = Dataset.from_list(tok_prompt).train_test_split(test_size=0.1)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["test"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("./phi2-meme-generator")
tokenizer.save_pretrained("./phi2-meme-generator")