from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
  task="text-generation"
)

model = ChatHuggingFace(llm = llm, temperature=1, max_new_tokens=100)

st.header("Basic Chatbot")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')

# user_input = st.text_input("Enter your prompt:")
prompt = template.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })

# | Syntax                                  | Description                                 | Use Case                                               |
# | --------------------------------------- | ------------------------------------------- | ------------------------------------------------------ |
# | `template.invoke(**args_dict)`          | Unpacks a dictionary into keyword arguments | When arguments are dynamic or programmatically defined |
# | `template.invoke(arg1=val1, arg2=val2)` | Passes keyword arguments explicitly         | When arguments are known and static                    |

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    # result = model.invoke(prompt)
    st.write(result.content)