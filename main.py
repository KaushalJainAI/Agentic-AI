from dotenv import load_dotenv
from typing import Annotated, Literal, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os 
import json

from agents import Chatbot, WebScrapingAgent, DatabaseQueryOrchestrator, VectorKnowledgeAgent

from datetime import datetime




load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

os.environ["GEMINI_API_KEY"] = api_key

search_key = os.getenv("TAVILY_API_KEY")

os.environ["TAVILY_API_KEY"] = search_key




# if __name__ == "__main__":
#     # Initialize the agent
#     agent = VectorKnowledgeAgent(
#         api_key=api_key,
#         index_path="Knowledge/my_knowledge_base"
#     )

#     # Add knowledge to the database
#     knowledge_docs = [
#         "Python is a high-level programming language known for its simplicity and readability.",
#         "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
#         "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently."
#     ]

#     metadata = [
#         {"category": "programming", "topic": "python", "difficulty": "beginner"},
#         {"category": "ai", "topic": "machine_learning", "difficulty": "intermediate"},
#         {"category": "databases", "topic": "vector_db", "difficulty": "advanced"}
#     ]

#     # Add knowledge
#     result = agent.add_knowledge(knowledge_docs, metadata)
#     print(f"Added {result['chunks_added']} knowledge chunks")

#     # Query for context
#     query_result = agent.query_knowledge("How to use Python for machine learning?", k=3)
#     print(f"Found {query_result.total_results} relevant contexts")

#     # Generate contextual prompt
#     prompt_result = agent.generate_contextual_prompt(
#         user_query="Create a Python script for machine learning",
#         task_type="code_generation",
#         category="ai",
#         context_limit=2
#     )

#     print("Optimized Prompt:")
#     print(prompt_result["optimized_prompt"])


#     def add_documents_from_files(agent, file_paths: List[str]):
#         """Add knowledge from multiple files"""
#         for file_path in file_paths:
#             with open(file_path, 'r') as f:
#                 content = f.read()
            
#             metadata = {
#                 "source": file_path,
#                 "file_type": file_path.split('.')[-1],
#                 "added_date": datetime.now().isoformat()
#             }
            
#             agent.add_knowledge(content, metadata)

#     def smart_context_retrieval(agent, query: str, task_type: str) -> str:
#         """Retrieve context dynamically based on query complexity"""
        
#         # Determine optimal context size based on query
#         query_words = len(query.split())
#         context_limit = min(max(query_words // 3, 2), 8)
        
#         # Adjust similarity threshold based on task type
#         similarity_thresholds = {
#             "code_generation": 0.4,
#             "explanation": 0.3,
#             "analysis": 0.5
#         }
#         min_similarity = similarity_thresholds.get(task_type, 0.3)
        
#         return agent.query_knowledge(
#             query, 
#             k=context_limit, 
#             min_similarity=min_similarity
#         )
    
    # Example usage



# # Usage Example
# if __name__ == "__main__":
#     orchestrator = DatabaseQueryOrchestrator(
#         database_directory="./Databases",
#         api_key=api_key
#     )
#     result = orchestrator.query("Tell me about the cars sold recently")
#     print(json.dumps(result, indent=2))

# if __name__ == "__main__":
#     sql_agent = SQL_CRUD_agent(
#         api_key=api_key,
#         database_directory="./Databases"
#     )

#     sql_agent.run_and_stream_watch("What are the most popular brands and what is the average selling price for the cars for each brand")

# if __name__ == "__main__":
#     search_bot = WebScrapingAgent(
#         api_key=api_key,
#         tavily_api_key=search_key,
#         temperature=0.2
#     ) 

#     # 2. Define the research prompt
#     prompt = "https://takeuforward.org/   go to this website and scrape the notes for all the questions in A2Z DSA sheet. I want the solution notes to be scrapped from the website"
    
#     # 3. Run the agent
#     structured_result = search_bot.run_and_stream_watch(prompt)

#     from pathlib import Path

#     # Create the file path
#     file_path = Path("Databases") / "DSA Questions" / "tuf saved.txt"

#     # Create directories if they don't exist
#     file_path.parent.mkdir(parents=True, exist_ok=True)

#     # Write to the file
#     with open(file_path, 'w') as f:
#         f.write("Your content here")
        
#     print(f"File written successfully to: {file_path}")
    
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



