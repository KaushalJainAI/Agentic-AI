from typing import Annotated, Literal, Optional, List, Dict, Any, Union, Callable
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
import os
import json
import logging
import sqlite3
from datetime import datetime
import uuid
from collections import defaultdict
from enum import Enum

# Main Application Launcher for SuperAgent System
import sys
import logging
import asyncio
import signal
from threading import Thread
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your modules
from config import SuperAgentConfig
from super_agent_system import SuperAgent, SuperAgentAPI
from connections import TelegramBot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('superagent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SuperAgentLauncher:
    """Main launcher for the SuperAgent Flask API system"""
    
    def __init__(self):
        self.super_agent = None
        self.api_server = None
        self.api_thread = None
        # self.shutdown_event = Event()
        
    def validate_setup(self):
        """Validate configuration before starting"""
        logger.info("Validating SuperAgent configuration...")
        
        config_status = SuperAgentConfig.validate_config()
        
        if not config_status["valid"]:
            logger.error("Configuration validation failed:")
            for issue in config_status["issues"]:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("Configuration validation passed:")
        for key, value in config_status["config_summary"].items():
            logger.info(f"  - {key}: {value}")
        
        return True
    
    def initialize_super_agent(self):
        """Initialize the SuperAgent with all specialized agents"""
        logger.info("Initializing SuperAgent system...")
        
        try:
            self.super_agent = SuperAgent(
                model=SuperAgentConfig.DEFAULT_MODEL,
                model_provider=SuperAgentConfig.MODEL_PROVIDER,
                temperature=SuperAgentConfig.TEMPERATURE,
                api_key=SuperAgentConfig.LLM_API_KEY,
                tavily_api_key=SuperAgentConfig.TAVILY_API_KEY,
                database_directory=SuperAgentConfig.DATABASE_DIRECTORY,
                knowledge_base_path=SuperAgentConfig.KNOWLEDGE_BASE_PATH,
                memory_db_path=SuperAgentConfig.MEMORY_DB_PATH
            )
            
            logger.info("SuperAgent initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SuperAgent: {e}")
            return False
    
    def start_api_server(self):
        """Start the Flask API server"""
        logger.info("Starting Flask API server...")
        
        try:
            self.api_server = SuperAgentAPI(
                self.super_agent,
                host=SuperAgentConfig.API_HOST,
                port=SuperAgentConfig.API_PORT
            )
            
            logger.info(f"API server started on {SuperAgentConfig.API_HOST}:{SuperAgentConfig.API_PORT}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    def _run_api_server(self):
        """Internal method to run API server with proper configuration"""
        try:
            # Run Flask with threading enabled for production-like behavior
            self.api_server.run(
                host=SuperAgentConfig.API_HOST,
                port=SuperAgentConfig.API_PORT,
                debug=False,
                threaded=True,
                use_reloader=False
            )
        except Exception as e:
            logger.error(f"API server error: {e}")
            self.shutdown_event.set()
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received. Cleaning up...")
            self.shutdown_event.set()
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Shutting down SuperAgent system...")
        self.shutdown_event.set()
        logger.info("SuperAgent system shutdown complete.")
    
    def run(self):
        """Main execution method - runs Flask server directly"""
        logger.info("🚀 Starting SuperAgent Flask API Server")
        
        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()
        
        # Step 1: Validate configuration
        if not self.validate_setup():
            logger.error("Configuration validation failed. Exiting.")
            sys.exit(1)
        
        # Step 2: Initialize SuperAgent
        if not self.initialize_super_agent():
            logger.error("SuperAgent initialization failed. Exiting.")
            sys.exit(1)
        
        # Step 3: Initialize API server
        if not self.start_api_server():
            logger.error("API server initialization failed. Exiting.")
            sys.exit(1)
        
        logger.info("🎉 SuperAgent Flask API is ready!")
        logger.info("=" * 60)
        logger.info("System Status:")
        logger.info(f"  ✅ SuperAgent Core: Active")
        logger.info(f"  ✅ API Server: http://{SuperAgentConfig.API_HOST}:{SuperAgentConfig.API_PORT}")
        logger.info(f"  ✅ Available Agents: 7 specialized agents")
        logger.info("=" * 60)
        logger.info("Starting Flask server...")
        
        try:
            # Run the Flask server directly (blocking call)
            self._run_api_server()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.shutdown()

# Usage
if __name__ == "__main__":
    launcher = SuperAgentLauncher()
    launcher.run()


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



