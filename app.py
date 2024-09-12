import streamlit as st
import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Set up your OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Create a basic Streamlit interface
st.title("LangChain + GPT-3/4 Chatbot")
st.write("Ask any question in chemistry, and the AI will answer it!")

# Get user input
user_input = st.text_input("Your question:")

# Define LangChain prompt template
template = PromptTemplate(input_variables=["question"], template="Answer the following question: {question}")

# Set up OpenAI LLM with LangChain
llm = OpenAI(model_name="gpt-4", temperature=0.7)  # You can use "gpt-3.5-turbo" or "gpt-4"
chain = LLMChain(llm=llm, prompt=template)

# Generate a response using LangChain
if user_input:
    response = chain.run(question=user_input)
    st.write(f"Answer: {response}")

