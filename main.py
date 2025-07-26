from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os 

from agents import Chatbot

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

os.environ["GEMINI_API_KEY"] = api_key



if __name__ == "__main__":
    bot = Chatbot(api_key=api_key)  # or let it use environment variable
    
    # Method 1: Single message
    response = bot.chat("Hello, how are you?")
    print(response)
    
    # Method 2: Interactive chat
    bot.interactive_chat()
    
    # Method 3: Chat with history
    conversation = [
        {"role": "user", "content": "My name is Kaushal"},
        {"role": "assistant", "content": "Nice to meet you, John!"},
        {"role": "user", "content": "What's my name?"}
    ]
    response = bot.chat_with_history(conversation)
    print(response)



