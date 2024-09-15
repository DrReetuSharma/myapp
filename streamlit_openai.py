import streamlit as st
from langchain.chat_models import ChatOpenAI  # Importing the chat model
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import pinecone
import os

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")
index_name = "langchain-index"

# Streamlit UI layout
st.title('Conversational AI with LangChain, Chat Model, and Memory')

# Ask user for OpenAI API key
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

if openai_api_key:
    # Store the key temporarily for the session
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Initialize conversational memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Ask the user for input
    user_input = st.text_input("Ask me anything:")

    if user_input:
        # Display previous conversation history
        st.write("Conversation history:")
        st.write(memory.load_memory_variables({}).get("chat_history", []))

        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()
        index = pinecone.Index(index_name)
        vector_store = Pinecone(index, embeddings.embed_query, "openai")

        # Initialize the Chat Model (OpenAI's GPT-3.5/4-like model)
        chat_model = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)  # ChatOpenAI used here

        # Set up a Retrieval-Augmented Generation (RAG) chain with memory
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model, 
            retriever=vector_store.as_retriever(), 
            memory=memory
        )

        # Get the response from the RAG agent
        response = rag_chain.run(user_input)

        # Store the conversation history
        memory.save_context({"input": user_input}, {"output": response})

        # Display the AI response
        st.write("AI Response:")
        st.write(response)

        # Display updated conversation history
        st.write("Updated conversation history:")
        st.write(memory.load_memory_variables({}).get("chat_history", []))
else:
    st.write("Please enter your OpenAI API key to proceed.")
