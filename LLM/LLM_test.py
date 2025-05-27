# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM # For loading model and tokenizer
from peft import PeftModel # For loading LoRA weights
import torch

# Load the tokenizer for base model
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
# Load the base model with the same configuration as in training
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
# Load the LoRA weights from the training
model = PeftModel.from_pretrained(base_model, "./kek_LLM")

# Configure tokenizer settings as in training
tokenizer.pad_token = tokenizer.eos_token
model = model.merge_and_unload()

def gen_OP(subject, max_length=150):
    """
    Generates a 4chan-style opening post about a given subject.
    
    Args:
        subject (str): Topic for the generated post (e.g., "Politics", "AI")
        max_length (int): Maximum number of new tokens to generate
        
    Returns:
        str: Generated post with the prompt included
    """
    # Create the generation prompt matching the training format
    prompt = f"Generate a /pol/ style opening post about: {subject}"
    # Tokenize the input and move to GPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # Generate text with customized sampling parameters
    outputs = model.generate(
        **inputs, # Pass tokenized inputs
        max_new_tokens=max_length, # Limit of response length
        temperature=1.0, # creativity level
        top_k=20, # Sample from top 20 most likely tokens
        do_sample=True, # Enable sampling (vs. greedy decoding)
        no_repeat_ngram_size=3, # Prevent 3-gram repetitions
        repetition_penalty=1.5, # Penalize repeated phrases
        pad_token_id=tokenizer.eos_token_id, # Padding token ID
    )
    # Decode the generated tokens to text and remove the prompt part
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[43:]

if __name__ == "__main__":
    # Test the generator with sample topics
    print(gen_OP("Politics"))
    print("\n !!!!!!!!!!! \n")
    print(gen_OP("AI"))