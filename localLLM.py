import asyncio
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
import ollama
from langchain.agents import initialize_agent, Tool


# Import your existing agents (assuming they're in the same directory)
from agents import (
    Chatbot, WebScrapingAgent, DatabaseQueryOrchestrator, 
    VectorKnowledgeAgent, DatabaseDiscoveryAgent, SQLQueryAgent, FlatFileQueryAgent
)

# Setup logger
logger = logging.getLogger(__name__)

from ollama import Client

client = Client(host="http://localhost:11434")

response = client.chat(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "Explain AI in simple terms. Keep it short."}],
    options={"temperature": 0.7}
)

print(f"Qwen response: {response['message']['content']}")

# # Flask Backend API
# from flask import Flask, request, jsonify
# import asyncio
# from threading import Thread

# class SuperAgentAPI:
#     """Flask API for SuperAgent"""
    
#     def __init__(self, super_agent: SuperAgent, host="127.0.0.1", port=5000):
#         self.super_agent = super_agent
#         self.app = Flask(__name__)
#         self.host = host
#         self.port = port
        
#         # Setup routes
#         self.app.route("/chat", methods=["POST"])(self.chat_endpoint)
#         self.app.route("/history/<user_id>", methods=["GET"])(self.history_endpoint)
#         self.app.route("/clear/<user_id>", methods=["DELETE"])(self.clear_endpoint)
#         self.app.route("/health", methods=["GET"])(self.health_endpoint)
    
#     def chat_endpoint(self):
#         """Handle chat messages"""
#         try:
#             data = request.get_json()
#             user_id = data.get("userId", "anonymous")
#             message = data.get("message", "")
            
#             if not message:
#                 return jsonify({"error": "Message is required"}), 400
            
#             # Process message asynchronously
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
            
#             response = loop.run_until_complete(
#                 self.super_agent.process_message(user_id, message)
#             )
            
#             loop.close()
            
#             return jsonify({"reply": response})
            
#         except Exception as e:
#             logger.error(f"API error: {e}")
#             return jsonify({"error": str(e)}), 500
    
#     def history_endpoint(self, user_id):
#         """Get conversation history"""
#         try:
#             limit = request.args.get("limit", 10, type=int)
#             history = self.super_agent.get_conversation_history(user_id, limit)
#             return jsonify({"history": history})
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
    
#     def clear_endpoint(self, user_id):
#         """Clear conversation history"""
#         try:
#             success = self.super_agent.clear_conversation_memory(user_id)
#             return jsonify({"success": success})
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
    
#     def health_endpoint(self):
#         """Health check endpoint"""
#         return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
#     def run(self, debug=False):
#         """Run the Flask API"""
#         logger.info(f"Starting SuperAgent API on {self.host}:{self.port}")
#         self.app.run(host=self.host, port=self.port, debug=debug)


# # Main execution
# if __name__ == "__main__":
#     # Initialize SuperAgent
#     super_agent = SuperAgent(
#         api_key=os.getenv("LLM_API_KEY"),
#         tavily_api_key=os.getenv("TAVILY_API_KEY"),
#         database_directory="./Databases",
#         knowledge_base_path="./knowledge_base"
#     )
    
#     # Initialize API
#     api = SuperAgentAPI(super_agent)
    
#     # Run API server
#     api.run(debug=True)


