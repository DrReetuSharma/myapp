import streamlit as st
from transformers import pipeline

# Load a larger model or modify the pipeline settings for better results
model = pipeline('text-generation', model='gpt2', do_sample=True, temperature=0.7  , top_p=0.9)


# Streamlit UI
st.title("Text Generation App using GPT-2")

# Input text from user
user_input = st.text_input("Enter a prompt to generate text:")

# Button to trigger text generation
if st.button("Generate Text"):
    if user_input:
        # Use GPT-2 to generate text
        generated_text = model(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
        st.write(generated_text)
    else:
        st.write("Please enter a prompt.")
