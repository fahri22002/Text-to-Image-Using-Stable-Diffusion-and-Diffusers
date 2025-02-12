from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import streamlit as st
import torch
     
model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
model_id2 = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")


def generate_image(pipe, prompt, params):
    img = pipe(prompt, **params).images
    
    num_images = len(img)
    if num_images > 1:
        cols = st.columns(num_images)
        for i in range(num_images):
            with cols[i]:
                st.image(img[i], caption=f"Generated Image {i+1}", use_column_width=True)
    else:
        st.image(img[0], caption="Generated Image", use_column_width=True)
