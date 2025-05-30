# # Load the fine-tuned BERT-Emotion model
from diffusers import StableDiffusionPipeline
import torch
from transformers import pipeline

sentiment_analysis = pipeline("text-classification", model="boltuix/bert-emotion")
sentiment_analysis.model.to("cuda")


def sentiment_analysis(text, max_length=100):
    """
    Analyzes the sentiment of the given text.

    Args:
        text (str): The text to summarize.
        max_length (int): The maximum length of the summary.

    Returns:
        str: The summarized text.
    """

    summary = sentiment_analysis(text)[0]["label"]
    return summary
