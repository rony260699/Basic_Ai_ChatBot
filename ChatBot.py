### This chatbot without web ui. 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm, temperature=0.2)
chat_history = [
    SystemMessage(content = "You are a helpful AI assistant.")
]
while True:
    user_prompt = input("You: ")
    chat_history.append(HumanMessage(content=user_prompt))
    if user_prompt.lower() == "exit":
        
        break

    else:
        ans = model.invoke(chat_history)
        chat_history.append(AIMessage(content=ans.content))
        print("Bot:", ans.content)

print(chat_history)
