# Import necessary libraries
import os

os.environ["HF_HOME"] = "/tmp/huggingface"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)  # For loading model and tokenizer
from peft import PeftModel  # For loading LoRA weights
import torch
from transformers import pipeline  # For text summarization and sentiment analysis
from generate_pepe import generate_pepe  # Custom module for image generation

# Load the tokenizer for base model
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
# Load the base model with the same configuration as in training
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
# Load the LoRA weights from the training
model = PeftModel.from_pretrained(base_model, "../LLM/kek_LLM")
model = model.to("cuda")  # Move model to GPU for faster inference

# Load the fine-tuned BERT-Emotion model
sentiment_analysis = pipeline("text-classification", model="boltuix/bert-emotion")


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
        **inputs,  # Pass tokenized inputs
        max_new_tokens=max_length,  # Limit of response length
        temperature=1.0,  # creativity level
        top_k=20,  # Sample from top 20 most likely tokens
        do_sample=True,  # Enable sampling (vs. greedy decoding)
        no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
        repetition_penalty=1.5,  # Penalize repeated phrases
        pad_token_id=tokenizer.eos_token_id,  # Padding token ID
    )
    # Decode the generated tokens to text and remove the prompt part
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[43:]


def test_sentiment(text):
    """
    Analyzes the sentiment of the given text.

    Args:
        text (str): Text to analyze

    Returns:
        str: Sentiment label
    """
    result = sentiment_analysis(text)
    return result[0]["label"] if result else "neutral"


if __name__ == "__main__":
    # Test the generator with sample topics
    topic = "cats"
    text = gen_OP(topic, max_length=150)
    print(text)
    sentiment = test_sentiment(text)
    print("sentiment: -> " + sentiment)
    img = generate_pepe(
        sentiment,
        num_inference_steps=50,
        guidance_scale=10,
        negative_prompt="ugly, blurry, bad quality, text, watermark, logo, signature, low resolution, low quality",
    )
    img.save("generated_pepe.jpg")
