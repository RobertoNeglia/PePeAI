# Import necessary libraries
import os
# Environment configuration for sigma2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Allows PyTorch to allocate GPU memory
os.makedirs("/tmp/huggingface", exist_ok=True) # Create directory for Hugging Face cache
os.environ["HF_HOME"] = "/tmp/huggingface" # Set Hugging Face cache directory
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disable tokenizer parallelism to avoid warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling # for model and training
from peft import LoraConfig, get_peft_model # for LoRA fine-tuning
import torch
from huggingface_hub import login # For Hugging Face model hub access
from textsetup import post_extractor # Custom module for processing 4chan posts
from datasets import Dataset # For dataset handling

# Authenticate with Hugging Face Hub
login(token="insert token") #!!! replace "insert token" with your actual token !!!
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
    r=12, # Rank of the low-rank matrices
    lora_alpha=42, # Scaling factor for LoRA weights
    target_modules=["q_proj", "k_proj", "v_proj", "dense"], # # Apply LoRA to query, value, key and 'dense' projection layers
    lora_dropout=0.1, # Dropout probability for LoRA layers
    bias="none", # Don't train bias parameters
    task_type="CAUSAL_LM" # Task type is causal language modeling
)
# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Enable for memory optimization
model.enable_input_require_grads() # Only compute gradients when required
model.gradient_checkpointing_enable() # Use gradcheckpoints (Trade compute for memory)

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./results", # Directory to save training outputs
    per_device_train_batch_size=25, # batch size for 16GB GPU memory
    gradient_accumulation_steps=1, # No accumulation needed with larger batch size
    learning_rate=3e-4, # Specify learning rate
    num_train_epochs=3, # Number of passes over the dataset
    logging_steps=50, # Show logs/progression every 50 steps
    fp16=True, # Use mixed-precision training
    optim="paged_adamw_8bit", # Use Adam optimizer with paging
    # trying to use multiple CPUs to speed up training (not working optimaly)
    dataloader_num_workers=4,   # Number of workers for data loading
    dataloader_pin_memory=True, # Pin memory for faster data transfer to GPU
    dataloader_persistent_workers=True, # Keep data loader workers alive for faster training
)
# Load and configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    padding_side="left", # Left-padding for causal LM
    trust_remote_code=True # Trust custom code from model repo
)
tokenizer.pad_token = tokenizer.eos_token   # Use EOS token for padding
tokenizer.padding_side = "left" # Consistent left-padding

# Process 4chan posts using custom extractor and tokenize them
# Parameters:
# - "pol_062016-112019_labeled.ndjson": Input file containing 4chan posts
# - sample_size=100000: Number of posts to sample
# - min_len=50: Minimum post character length (not token length)
# - tokenizer: Tokenizer to use for processing
tok_prompts, _ = post_extractor("pol_062016-112019_labeled.ndjson", sample_size=100000, min_len=50, tokenizer=tokenizer)
# Create train/test split for monotoring training progress
tok_ds = Dataset.from_list(tok_prompts).train_test_split(test_size=0.1)

# Configure data collator for language modeling (without masked language modeling)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
# Initialize Trainer for model training
trainer = Trainer(
    model=model, # LoRA-configured phi-2 model
    args=training_args, # Training configuration
    train_dataset=tok_ds["train"],  # Training dataset
    eval_dataset=tok_ds["test"], # Evaluation dataset
    data_collator=data_collator, # How to collate batches
    tokenizer=tokenizer, # Tokenizer for processing
)
# Start training
trainer.train()
# Save the fine-tuned model and tokenizer
model.save_pretrained("./kek_LLM")
tokenizer.save_pretrained("./kek_LLM")