import os
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs("/tmp/huggingface", exist_ok=True)
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch
from huggingface_hub import login
from textsetup import post_extractor
from datasets import Dataset
from torch.utils.data import DataLoader

login(token="insert token")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

peft_config = LoraConfig(
    r=12,
    lora_alpha=42,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=25,
    gradient_accumulation_steps=1,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_steps=50,
    fp16=True,
    optim="paged_adamw_8bit",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    padding_side="left",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

tok_prompts, _ = post_extractor("pol_062016-112019_labeled.ndjson", sample_size=100000, min_len=50, tokenizer=tokenizer)
tok_ds = Dataset.from_list(tok_prompts).train_test_split(test_size=0.1)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

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