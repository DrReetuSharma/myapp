import streamlit as st
from transformers import pipeline

# Load the BERT-based question-answering model
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Streamlit UI
st.title("Question Answering App using BERT")

# Input text (context) and question from user
context = st.text_area("Enter the context (passage of text):")
question = st.text_input("Enter your question:")

# Button to trigger the question-answering
if st.button("Get Answer"):
    if context and question:
        # Use the BERT QA pipeline to get the answer
        result = qa_pipeline(question=question, context=context)
        answer = result['answer']
        st.write(f"**Answer:** {answer}")
    else:
        st.write("Please provide both context and a question.")
