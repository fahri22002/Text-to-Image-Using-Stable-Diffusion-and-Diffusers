import streamlit as st
import torch
from stable_function import generate_image, pipe

# Title
st.title("Stable Diffusion Image Generator")

# Input fields
prompt = st.text_input("Enter your prompt:", "Dinosaur walking in the forest")
num_images = st.number_input("Number of images:", min_value=1, max_value=5, value=1, step=1)
width = st.number_input("Image width:", min_value=256, max_value=1024, value=512, step=64)
height = st.number_input("Image height:", min_value=256, max_value=1024, value=640, step=64)

def run_generation():
    params = {
        'num_inference_steps': 100,  # Default inference steps
        'width': width,
        'height': height,
        'num_images_per_prompt': num_images
    }
    generate_image(pipe, prompt, params)

# Button to trigger generation
if st.button("Generate"):
    run_generation()
