from diffusers import StableDiffusionPipeline
import torch
from transformers import pipeline

# model_path = "RobertoNeglia/pepe_generator_sd2_sentiment"
# pipe = StableDiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2", torch_dtype=torch.float16
# )
model_path = "RobertoNeglia/pepe_generator_sd2base_sentiment"
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base", torch_dtype=torch.float16
)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

# Load the fine-tuned BERT-Emotion model
sentiment_analysis = pipeline("text-classification", model="boltuix/bert-emotion")
sentiment_analysis.model.to("cuda")


def generate_pepe(
    text, num_inference_steps=50, guidance_scale=7.5, negative_prompt=None
):
    """
    Generates an image of Pepe the Frog based on the provided prompt.

    Args:
        prompt (str): Text prompt describing the desired image.
        num_inference_steps (int): Number of denoising steps for image generation.
        guidance_scale (float): Scale for classifier-free guidance.

    Returns:
        PIL.Image: Generated image of Pepe the Frog.
    """
    topic = sentiment_analysis(text)[0]["label"]
    prompt = "pepe the frog, " + topic
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    ).images[0]
    return image
