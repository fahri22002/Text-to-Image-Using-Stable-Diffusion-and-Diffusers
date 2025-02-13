from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import streamlit as st
import torch
     
# Define the model ID for the Stable Diffusion pipeline
model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
# Alternative model (commented out)
# model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"

# Load the Stable Diffusion model with optimized settings
pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")  # Move the model to the GPU for faster processing

def generate_image(pipe, prompt, params):
    """
    Generates images based on the given prompt and parameters using the Stable Diffusion pipeline.
    
    Args:
        pipe: The Stable Diffusion pipeline object.
        prompt (str): The text prompt describing the desired image.
        params (dict): Additional parameters for image generation.
    
    Returns:
        Displays the generated image(s) in Streamlit.
    """
    img = pipe(prompt, **params).images  # Generate images from the prompt

    num_images = len(img)
    if num_images > 1:
        # Display multiple images in separate columns
        cols = st.columns(num_images)
        for i in range(num_images):
            with cols[i]:
                st.image(img[i], caption=f"Generated Image {i+1}", use_column_width=True)
    else:
        # Display a single image
        st.image(img[0], caption="Generated Image", use_column_width=True)
