import orjson
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
url_pattern = re.compile(r'https?://\S+|www\.\S+')

def clean_text(text):
    if not text:
        return ""
    text = text.replace("<wbr>", "").replace("\n", "")
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text("\n")
    text = url_pattern.sub('', text)
    return text

def post_extractor(ndjson_path="pol_062016-112019_labeled.ndjson", sample_size=100, min_len=10, tokenizer=None):
    meta_data = []
    prompts = []
    
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing posts", unit="lines"):
            if len(meta_data) >= sample_size:
                break
            try:
                data = orjson.loads(line)
                for post in data.get("posts", []):
                    if (post.get("resto") == 0 and 
                        "sub" in post and 
                        len(post.get("com", "")) > min_len):
                        
                        cleaned_text = clean_text(post.get("com", ""))
                        if len(cleaned_text) > min_len:
                            meta_data.append({
                                "sub": post["sub"],
                                "text": cleaned_text,
                                "post_id": post["no"],
                                "toxic_score": post.get("perspectives", {}).get("TOXICITY", 0)
                            })
                        
                            if tokenizer != None:
                                prompt_text = f"Generate a /pol/ style opening post about: {post['sub']}"
                                full_text = f"{prompt_text} {cleaned_text}{tokenizer.eos_token}"

                                tokenized = tokenizer(
                                    full_text,
                                    truncation=True,
                                    max_length=512,
                                    padding="max_length",
                                )
                                
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
    post_n=0
    for op in ops:
        if post_n >= from_post:
            print(f"ID:{op['post_id']}, Sub: {op['sub']}, Toxicity: {op['toxic_score']} \npost:{op['text']}\n")
        if to_posts == post_n:
            break
        post_n += 1
    
if __name__ == "__main__":
    sample_prompt, meta_prompt = post_extractor(sample_size=100, min_len=50)
    post_printer(meta_prompt, 0, 15)
