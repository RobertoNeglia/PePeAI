# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling  # for model and training
from peft import LoraConfig, get_peft_model # for LoRA fine-tuning
import torch
from textsetup import post_extractor # Custom module for processing 4chan posts
from datasets import Dataset # For dataset handling

# Configure 4-bit quantization to reduce memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Load weights in 4-bit
    bnb_4bit_quant_type="nf4", # Use normalized float
    bnb_4bit_compute_dtype=torch.float16, # Compute in float16
)
# Load the pre-trained phi-2 model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", # Select the phi-2 model
    quantization_config=bnb_config, # Apply the quantization config
    device_map="auto" # Automatically map model to available devices
)
# Configure LoRA for fine-tuning
peft_config = LoraConfig(
    r=8, # Rank of the low-rank matrices
    lora_alpha=32, # Scaling factor for LoRA weights
    target_modules=["q_proj", "v_proj"], # Apply LoRA to query and value projection layers
    lora_dropout=0.05, # Dropout probability for LoRA layers
    bias="none", # Don't train bias parameters
    task_type="CAUSAL_LM" # Task type is causal language modeling
)
model = get_peft_model(model, peft_config) # Apply LoRA to the model

# Load the tokenizer for phi-2
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    padding_side="left", # Pad on the left for causal LM
    trust_remote_code=True # Trust remote code from the model repo
)
# Configure training arguments
training_args = TrainingArguments(
    output_dir="./results", # Directory to save training outputs
    per_device_train_batch_size=1,  # Small batch size for 8GB GPU memory
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps to effectively increase batch size
    num_train_epochs=3, # Number of passes over the dataset
    logging_steps=50,   # show logs/progression every 50 steps
    fp16=True, # Use mixed-precision training
    optim="paged_adamw_8bit" # Use Adam optimizer with paging
)
# Configure tokenizer settings
tokenizer.pad_token = tokenizer.eos_token # Use EOS token as padding token
tokenizer.padding_side = "left" # Consistent padding side

# Process 4chan posts using custom extractor and tokenize them
# Parameters:
# - "pol_062016-112019_labeled.ndjson": Input file containing 4chan posts
# - sample_size=1000: Number of posts to sample
# - min_len=50: Minimum post character length (not token length)
# - tokenizer: Tokenizer to use for processing
tok_prompt, _ = post_extractor("pol_062016-112019_labeled.ndjson", sample_size=1000, min_len=50, tokenizer=tokenizer)
# Create train/test split for monotoring training progress
tok_ds = Dataset.from_list(tok_prompt).train_test_split(test_size=0.1)

# Configure data collator for language modeling (without masked language modeling)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Initialize Trainer for model training
trainer = Trainer(
    model=model, # LoRA-configured phi-2 model
    args=training_args, # Training configuration
    train_dataset=tok_ds["train"], # Training dataset
    eval_dataset=tok_ds["test"], # Evaluation dataset
    data_collator=data_collator, # How to collate batches
    tokenizer=tokenizer # Tokenizer for processing
)
# Start training
trainer.train()
# Save the fine-tuned model and tokenizer
model.save_pretrained("./kek_LLM")
tokenizer.save_pretrained("./kek_LLM")