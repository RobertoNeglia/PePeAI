# Import necessary libraries
import orjson # Fast JSON
from tqdm import tqdm # Progress bar utility
from bs4 import BeautifulSoup # HTML cleaner
import re
from bs4 import MarkupResemblesLocatorWarning # Warning for BeautifulSoup
import warnings # For suppressing warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning) # Suppress warnings about HTML
url_pattern = re.compile(r'https?://\S+|www\.\S+') # Regex to match URLs

def clean_text(text):
    """Cleans raw 4chan post text by:
    1. Removing HTML line breaks (<wbr>) and newlines to Isolate URLs
    2. Extracting text from HTML
    3. Removing URLs
    Args:
        text (str): Raw text from 4chan post that may contain HTML and URLs
        
    Returns:
        str: Cleaned text with HTML removed and URLs stripped out
    """
    if not text:
        return ""
    text = text.replace("<wbr>", "").replace("\n", "")
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text("\n")
    text = url_pattern.sub('', text)
    return text

def post_extractor(ndjson_path="pol_062016-112019_labeled.ndjson", sample_size=100, min_len=10, tokenizer=None):
    """
    Extracts and processes posts from 4chan JSONL file.
    
    Args:
        ndjson_path (str): Path to NDJSON file containing 4chan posts
        sample_size (int): Maximum number of posts to extract
        min_len (int): Minimum character length of post to include (not token length)
        tokenizer: tokenizer for immediate text processing
    
    Returns:
        Tuple of (tokenized_prompts, metadata) if tokenizer provided
        or (empty_list, metadata) if no tokenizer
    """
    meta_data = [] # Will store post metadata and cleaned text
    prompts = [] # Will store post metadata and cleaned text
    
    # Open the NDJSON file and read line by line
    with open(ndjson_path, "r", encoding="utf-8") as f:
        # Will store post metadata and cleaned text
        for line in tqdm(f, desc="Processing posts", unit="lines"):
            #Stop if we have collected enough samples
            if len(meta_data) >= sample_size:
                break
            try:
                # Load the JSON data from the line
                data = orjson.loads(line)
                # Iterate through posts in the data
                for post in data.get("posts", []):
                    # Check if post is an opening post (resto == 0), has a subject and meets the minimum length criteria
                    if (post.get("resto") == 0 and 
                        "sub" in post and 
                        len(post.get("com", "")) > min_len):
                        # Clean the post text
                        cleaned_text = clean_text(post.get("com", ""))
                        # check if cleaned text meets the minimum length criteria
                        if len(cleaned_text) > min_len:
                            # Append metadata for the post
                            meta_data.append({
                                "sub": post["sub"],
                                "text": cleaned_text,
                                "post_id": post["no"],
                                "toxic_score": post.get("perspectives", {}).get("TOXICITY", 0)
                            })
                            # If a tokenizer is provided, tokenize the cleaned text
                            if tokenizer != None:
                                # Create a prompt for the model
                                prompt_text = f"Generate a /pol/ style opening post about: {post['sub']}"
                                full_text = f"{prompt_text} {cleaned_text}{tokenizer.eos_token}"
                                # Apply Tokenizer
                                tokenized = tokenizer(
                                    full_text,
                                    truncation=True,
                                    max_length=512,
                                    padding="max_length",
                                )
                                # Append the tokenized prompt to the prompts list with masking
                                prompts.append({
                                    "input_ids": tokenized["input_ids"],
                                    "attention_mask": tokenized["attention_mask"],
                                    "labels": tokenized["input_ids"].copy()
                                })
            except Exception as e:
                print('oh shit')
                print(f"Skipping malformed line: {str(e)}")

    return prompts, meta_data


def post_printer(ops,from_post=0, to_posts=10):
    """
    Prints a selection of posts with their metadata.
    
    Args:
        ops: List of opening post dictionaries
        from_post: Starting index
        to_posts: Number of posts to print
    """
    post_n=0
    for op in ops:
        # Print post metadata and text with formatting
        if post_n >= from_post:
            print(f"ID:{op['post_id']}, Sub: {op['sub']}, Toxicity: {op['toxic_score']} \npost:{op['text']}\n")
        if to_posts == post_n:
            break
        post_n += 1
    
if __name__ == "__main__":
    # Example usage when run directly
    sample_prompt, meta_prompt = post_extractor(sample_size=100, min_len=50)
    post_printer(meta_prompt, 0, 15) # Example usage when run directly
