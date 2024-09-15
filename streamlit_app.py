import streamlit as st
from transformers import pipeline

# Load the distilgpt2 model from Hugging Face
model = pipeline('text-generation', model='distilgpt2')

# Streamlit UI
st.title("Text Generation App using distilgpt2")

# Input text from user
user_input = st.text_input("Enter a prompt to generate text:")

# Button to trigger text generation
if st.button("Generate Text"):
    if user_input:
        # Use distilgpt2 to generate text
        generated_text = model(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
        st.write(generated_text)
    else:
        st.write("Please enter a prompt.")
