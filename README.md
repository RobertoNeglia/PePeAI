# PePeAI
Create 4chan posts with images.

## Image Generator environment Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/RobertoNeglia/PePeAI.git
    cd PePeAI
    ```

2. Install conda from [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Anaconda](https://www.anaconda.com/products/distribution).

3. Create a virtual environment:
    ```bash
    conda env create -f environment_img_lora.yml
    conda activate genai
    ```

## To train the LoRA model on the images:

1. ~~Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tornikeonoprishvili/pepe-memes-dataaset).~~ (not really needed - the dataset is on Hugging Face)

2. Install the diffusers library from source:
    ```bash
    git clone https://github.com/huggingface/diffusers
    cd diffusers
    pip install .
    ```

3. Export environment variables:
    ```bash
    export MODEL_NAME="stabilityai/stable-diffusion-2" # or "runwayml/stable-diffusion-v1-5" for SD1.5
    export DATASET_NAME="RobertoNeglia/pepe_dataset"
    export OUTPUT_DIR="/path/to/output/dir" # e.g. "/home/user/pepe_lora"
    export HUB_MODEL_ID="yourusername/model_name" # e.g. "RobertoNeglia/pepe_generator"
    ```
    Replace `/path/to/output/dir` with the path where you want to save the model and `yourusername/model_name` with your Hugging Face username and desired model name.

4. Login to Hugging Face Hub:
    ```bash
    huggingface-cli login
    ```
    Follow the instructions to log in to your Hugging Face account.  
    If you want to push the model to the Hub, make sure you have the right permissions to do so, or generate a new token. 

5. Login to WandB (for logging):
    ```bash
    wandb login
    ```
    Follow the instructions to log in to your WandB account.

6. Run the training script from the diffusers library:
    ```bash
    accelerate launch  train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME   \
    --dataloader_num_workers=8   \
    --resolution=512 \
    --center_crop \
    --random_flip   \
    --train_batch_size=24   \ # Adjust based on your GPU memory
    --gradient_accumulation_steps=4   \
    --max_train_steps=1500   \ # Number of training steps
    --learning_rate=1e-04   \ # Learning rate for the optimizer
    --max_grad_norm=1   \ # Maximum gradient norm
    --lr_scheduler="cosine" \ # Learning rate scheduler
    --lr_warmup_steps=0   \
    --output_dir=${OUTPUT_DIR}   \
    --push_to_hub   \ # Push the model to the Hugging Face Hub
    --hub_model_id=${HUB_MODEL_ID}   \
    --report_to=wandb   \ # Log to WandB
    --checkpointing_steps=500   \ # Save checkpoints every 500 steps
    --validation_prompt="A very sad Pepe the Frog"   \ # Prompt for validation
    --seed=1337 \
    --caption_column="features" # Column in the dataset containing the captions
    ```

7. Look [here](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README.md) for more information on training the LoRA model.

## To generate images with the LoRA model:

```python
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

model_path = "RobertoNeglia/pepe_generator_sd2" # or RobertoNeglia/pepe_generator for SD1.5, or your own model path
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16) # or "runwayml/stable-diffusion-v1-5" for SD1.5
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "A very sad Pepe the Frog, crying in a dark room, digital art"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
plt.imshow(image)
plt.axis("off")
plt.show()
```

