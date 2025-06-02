# PePeAI
Create 4chan posts with images.

## Overview
![PoliticianPepe](images/PoliticianPepe.png)

## Environment Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/RobertoNeglia/PePeAI.git
    cd PePeAI
    ```

2. Install conda from [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Anaconda](https://www.anaconda.com/products/distribution).

3. Create a virtual environment:
    ```bash
    conda env create -f environment.yml
    conda activate genai
    git clone https://github.com/huggingface/diffusers
    cd diffusers
    pip install .
    ```

    If you are running into environment issues, try running terminal commands:
    ```bash
    conda create -n genai python=3.10 -y
    conda activate genai
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers peft accelerate safetensors
    git clone https://github.com/huggingface/diffusers
    cd diffusers
    pip install .
    ```

## To generate post with the GUI:

1. Run GUI.py.
    ```bash
    cd scripts
    python GUI.py
    ```
2. Write a subject in textbox.
3. Click on the generate button.

## Training of the LoRA image generator and LLM model
### To train the LoRA image generator on the images:

1. ~~Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tornikeonoprishvili/pepe-memes-dataaset).~~ (not really needed - the dataset is on Hugging Face)

2. Export environment variables:
    ```bash
    export MODEL_NAME="stabilityai/stable-diffusion-2-base" # or "runwayml/stable-diffusion-v1-5" for SD1.5
    export DATASET_NAME="RobertoNeglia/pepe_dataset_sentiment"
    export OUTPUT_DIR="/path/to/output/dir" # e.g. "/home/user/pepe_lora"
    export HUB_MODEL_ID="yourusername/model_name" # e.g. "RobertoNeglia/pepe_generator"
    ```
    Replace `/path/to/output/dir` with the path where you want to save the model and `yourusername/model_name` with your Hugging Face username and desired model name.

3. Login to Hugging Face Hub:
    ```bash
    huggingface-cli login
    ```
    Follow the instructions to log in to your Hugging Face account.  
    If you want to push the model to the Hub, make sure you have the right permissions to do so, or generate a new token. 

4. Login to WandB (for logging):
    ```bash
    wandb login
    ```
    Follow the instructions to log in to your WandB account.

5. Run the training script from the diffusers library:
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
    --max_train_steps=4300   \ # Number of training steps
    --learning_rate=1e-04   \ # Learning rate for the optimizer
    --max_grad_norm=1   \ # Maximum gradient norm
    --lr_scheduler="cosine" \ # Learning rate scheduler
    --lr_warmup_steps=0   \
    --output_dir=${OUTPUT_DIR}   \
    --push_to_hub   \ # Push the model to the Hugging Face Hub
    --hub_model_id=${HUB_MODEL_ID}   \
    --report_to=wandb   \ # Log to WandB
    --checkpointing_steps=500   \ # Save checkpoints every 500 steps
    --validation_prompt="pepe the frog, happiness"   \ # Prompt for validation
    --seed=42 \
    --caption_column="features" # Column in the dataset containing the captions
    --validation_epochs=50 # Validate every 50 epochs
    --rank=16 # Rank for LoRA
    ```

6. Look [here](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README.md) for more information on training the LoRA model.

### To generate images with the LoRA image generator:

```python
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

model_path = "RobertoNeglia/pepe_generator_sd2base_sentiment"
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "pepe the frog, sad, crying, digital art, high quality"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
plt.imshow(image)
plt.axis("off")
plt.show()
```

### To train the LLM model on the /pol/-4chan posts:

1. Download the [LLM data](https://zenodo.org/records/3606810) to the clone's driectory.

3. Run termial commands in clone's driectory:
    ```bash
    unzstd pol_0616-1119_labeled.tar.zst
    tar -xvf pol_0616-1119_labeled.tar
    ```

3. Run LLM_training.py.

### To generate text with the LoRA LLM:

1. Open LLM_test.py.
2. Insert a subject in the name==main block for the prompt "Generate a /pol/ style opening post about: {subject}".
    ```python
    if __name__ == "__main__":
        # Test the generator with sample topics
        topic = "cats"
        text = gen_OP(topic, max_length=150)
        print(text)
    ```
3. Run LLM_test.py
    ```bash
    cd scripts
    python LLM_test.py
    ```
