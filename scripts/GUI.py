# Import necessary libraries
import tkinter as tk  # Import tkinter for GUI
from tkinter import scrolledtext  # Import scrolledtext for scrollable text area
from PIL import ImageTk  # Import PIL for image handling
from LLM_test import (
    gen_OP,
)  # Import the LLM text generatoin function from LLM_test.py
from generate_pepe import (
    generate_pepe,
)  # Import the image generation function from generate_pepe.py

# from sentiment_analysis import (
#     sentiment_analysis,
# )  # Import the text summarization function from summarize_text.py


def GUI_exe():
    """Main function to create and run the GUI application"""
    # Initialize main application window
    root = tk.Tk()
    root.title("PePeAI")  # Window title
    # Create and pack UI elements with vertical padding
    tk.Label(root, text="Generate a post about:").pack(pady=5)
    # Text entry field for user input
    entry = tk.Entry(root, width=50)
    entry.pack(pady=5)
    # Scrollable text area for displaying generated posts
    output = scrolledtext.ScrolledText(root, width=60, height=10, wrap=tk.WORD)
    output.pack(pady=5)
    # Label for displaying meme images
    image_label = tk.Label(root)
    image_label.pack(pady=5)

    def on_generate():
        """Callback function for the generate button"""
        topic = entry.get()  # Get user input
        # make sure the topic is not empty
        if not topic:
            return
        # Generate text using LLM LoRA model
        generated_text = gen_OP(topic)
        # Clear previous output and display new text
        output.delete(1.0, tk.END)
        output.insert(tk.END, generated_text)
        # Summarize the generated text
        # summarized_text = sentiment_analysis(generated_text) #not in current usage
        # Generate image using LoRA diffusion model
        generated_image = generate_pepe(
            topic,
            num_inference_steps=100,
            guidance_scale=10,
            negative_prompt="bad quality, low resolution, blurry, out of focus, watermark",
        )
        # Save the generated image to a file
        meme_img = generated_image
        meme_img = meme_img.resize((400, 400))
        meme_photo = ImageTk.PhotoImage(meme_img)
        # Update image display
        image_label.config(image=meme_photo)
        image_label.image = meme_photo

    # Create generate button with callback function
    tk.Button(root, text="Generate Post", command=on_generate).pack(pady=10)
    # Start the main event loop
    root.mainloop()


if __name__ == "__main__":
    GUI_exe()
