from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
  task="text-generation"
)

model = ChatHuggingFace(llm = llm, temperature=1, max_new_tokens=100)

st.header("Basic Chatbot")



user_input = st.text_input("Enter your prompt:")

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)
    




