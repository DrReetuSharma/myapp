import streamlit as st
from transformers import pipeline

# Load the BERT-based question-answering model once
@st.cache_resource
def load_model():
    return pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

qa_pipeline = load_model()

# Streamlit UI
st.title("Question Answering App using BERT")

# Input text (context) and question from user
context = st.text_area("Enter the context (passage of text):", "")
question = st.text_input("Enter your question:", "")

# Button to trigger the question-answering
if st.button("Get Answer"):
    if context and question:
        try:
            # Use the BERT QA pipeline to get the answer
            result = qa_pipeline(question=question, context=context)
            answer = result.get('answer', 'No answer found.')
            st.write(f"**Answer:** {answer}")
        except Exception as e:
            st.write(f"An error occurred: {e}")
    else:
        st.write("Please provide both context and a question.")
