import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Initialize Pinecone
pinecone.init(api_key="your_pinecone_api_key", environment="us-west1-gcp")
index_name = "langchain-index"

# Streamlit UI layout
st.title('RAG Agent with LangChain and LLM')
user_input = st.text_input("Ask me anything:")

if user_input:
    st.write(f"You asked: {user_input}")

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    index = pinecone.Index(index_name)
    vector_store = Pinecone(index, embeddings.embed_query, "openai")

    # Initialize the LLM
    llm = OpenAI(temperature=0.7)

    # Set up a Retrieval-Augmented Generation (RAG) chain
    rag_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store.as_retriever())

    # Get the response from the RAG agent
    response = rag_chain.run(user_input)

    # Display the AI response
    st.write("AI Response:")
    st.write(response)
