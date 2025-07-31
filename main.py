from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os 
import json

from agents import Chatbot, WebScrapingAgent, SQL_CRUD_agent, FlatFileQueryAgent



load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

os.environ["GEMINI_API_KEY"] = api_key

search_key = os.getenv("TAVILY_API_KEY")

os.environ["TAVILY_API_KEY"] = search_key


if __name__ == "__main__":
    flat_file_agent = FlatFileQueryAgent(
        api_key=api_key,
        database_directory = "./Databases"
    )

    flat_file_agent.run_and_stream_watch("Tell me what are the top 5 best selling bikes and how much stock is left for them")

# if __name__ == "__main__":
#     sql_agent = SQL_CRUD_agent(
#         api_key=api_key,
#         database_directory="./Databases"
#     )

#     sql_agent.run_and_stream_watch("What are the most popular brands and what is the average selling price for the cars for each brand")

# if __name__ == "__main__":
#     search_bot = WebScrapingAgent(
#         api_key=api_key,
#         tavily_api_key=search_key
#     ) 

#     # 2. Define the research prompt
#     prompt = "Tell me about excersizes for long life. list down them in priority order"
    
#     # 3. Run the agent
#     structured_result = search_bot.run_and_stream_watch(prompt)
    
#     # 4. Print the result
#     print("\n--- ✅ FINAL STRUCTURED OUTPUT ---")
#     print(json.dumps(structured_result, indent=2))


# if __name__ == "__main__":
#     bot = Chatbot(api_key=api_key)  # or let it use environment variable
    
#     # Method 1: Single message
#     response = bot.chat("Hello, how are you?")
#     print(response)
    
#     # Method 2: Interactive chat
#     bot.interactive_chat()
    
#     # Method 3: Chat with history
#     conversation = [
#         {"role": "user", "content": "My name is Kaushal"},
#         {"role": "assistant", "content": "Nice to meet you, John!"},
#         {"role": "user", "content": "What's my name?"}
#     ]
#     response = bot.chat_with_history(conversation)
#     print(response)



