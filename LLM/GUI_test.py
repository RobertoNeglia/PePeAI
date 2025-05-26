import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def generate_text(subject):
    prompt = f"Generate a /pol/ style opening post about: {subject}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=1.0,
        top_k=20,
        do_sample=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[43:]

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, "./kek_LLM").merge_and_unload()
tokenizer.pad_token = tokenizer.eos_token

def GUI_exe():
    root = tk.Tk()
    root.title("GPT-4chan")

    tk.Label(root, text="Generate a post about:").pack(pady=5)
    entry = tk.Entry(root, width=50)
    entry.pack(pady=5)

    output = scrolledtext.ScrolledText(root, width=60, height=10, wrap=tk.WORD)
    output.pack(pady=5)
    image_label = tk.Label(root)
    image_label.pack(pady=5)

    def on_generate():
        topic = entry.get()
        if not topic:
            return
        generated_text = generate_text(topic)
        output.delete(1.0, tk.END)
        output.insert(tk.END, generated_text)

        
        meme_img = Image.open("Pepeee.jpg")
        meme_photo = ImageTk.PhotoImage(meme_img)
        image_label.config(image=meme_photo)
        image_label.image = meme_photo
    
    tk.Button(root, text="Generate Post", command=on_generate).pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    GUI_exe()