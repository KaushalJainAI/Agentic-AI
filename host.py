import os
import logging
from flask import Flask, request, jsonify
from datetime import datetime
from typing import Dict, Any

# Import your SuperAgent Orchestrator
from langgraph_super_agent import SuperAgentOrchestrator
from agents import (
    Chatbot, WebScrapingAgent, DatabaseQueryOrchestrator, 
    SQLQueryAgent
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SuperAgentFlaskApp:
    """
    Flask application that integrates the LangGraph SuperAgent Orchestrator
    with a clean API interface for Telegram bot communication.
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        self.orchestrator = None
        self._initialize_agents()
        self._setup_routes()
        
    def _initialize_agents(self):
        """Initialize all agents and the orchestrator"""
        try:
            # Initialize your agents (same as in your original code)
            agents = {
                "Chatbot": Chatbot(), 
                "DatabaseOrchestrator": DatabaseQueryOrchestrator(), 
                "WebScrapingAgent": WebScrapingAgent(),
                # Add other agents as needed
            }
            
            # Create the orchestrator
            self.orchestrator = SuperAgentOrchestrator(agents)
            logger.info("SuperAgent Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def _setup_routes(self):
        """Setup Flask routes - CORRECTED VERSION"""
        # Method 1: Direct route registration (RECOMMENDED)
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return self.health_check()
        
        @self.app.route('/chat', methods=['GET', 'POST'])
        def chat_endpoint():
            return self.chat_endpoint()

    
    def health_check(self):
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "SuperAgent Flask Server"
        })
    
    def chat_endpoint(self):
        """
        Main chat endpoint that processes user messages through the SuperAgent
        
        Expected JSON payload:
        {
            "message": "user message",
            "userId": "unique_user_id",
            "username": "telegram_username"
        }
        
        Returns:
        {
            "reply": "agent response",
            "success": true,
            "metadata": {...}
        }
        """
        try:
            # Validate request
            if not request.is_json:
                return jsonify({
                    "error": "Content-Type must be application/json",
                    "success": False
                }), 400
            
            data = request.get_json()
            
            # Validate required fields
            required_fields = ["message", "userId"]
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return jsonify({
                    "error": f"Missing required fields: {missing_fields}",
                    "success": False
                }), 400
            
            user_message = data["message"].strip()
            user_id = data["userId"]
            username = data.get("username", "Unknown")
            
            if not user_message:
                return jsonify({
                    "error": "Message cannot be empty",
                    "success": False
                }), 400
            
            logger.info(f"Processing message from user {user_id} ({username}): {user_message[:100]}...")
            
            # Execute the workflow using your SuperAgent Orchestrator
            result = self.orchestrator.execute_workflow(
                query=user_message,
                thread_id=f"telegram_user_{user_id}"
            )            
            # Process the result and create response
            if result["success"]:
                # Extract the meaningful response from the agent results
                reply = self._format_agent_response(result)
                
                response_data = {
                    "reply": reply,
                    "success": True,
                    "metadata": {
                        # "agent_used": result.get("selected_agent", "unknown"),
                        "workflow_status": result.get("status", "unknown"),
                        "completed_steps": result.get("completed_steps", []),
                        "thread_id": result.get("thread_id")
                    }
                }
                
                logger.info(f"Successfully processed message for user {user_id}")
                return jsonify(response_data)
                
            else:
                # Handle workflow failures
                error_message = "I apologize, but I encountered an issue processing your request. Please try again."
                if result.get("errors"):
                    logger.error(f"Workflow errors for user {user_id}: {result['errors']}")
                
                return jsonify({
                    "reply": error_message,
                    "success": False,
                    "metadata": {
                        "errors": result.get("errors", []),
                        "status": result.get("status", "failed")
                    },
                }, 200)
                
        except Exception as e:
            logger.error(f"Unexpected error in chat endpoint: {e}")
            return jsonify({
                "reply": "I'm experiencing technical difficulties. Please try again in a moment.",
                "success": False,
                "error": "Internal server error"
            }), 200
    
    def _format_agent_response(self, result: Dict[str, Any]) -> str:
        """Format agent response for systematic Telegram display"""
        try:
            agent_results = result['results']
            # print(type(agent_results))

            if not agent_results:
                return "I processed your request, but didn't generate a specific response."
            
            # print(agent_results.keys())

            # Handle different agent response types based on your actual output
            if "chat" in agent_results:
                # Chatbot responses - direct text
                response = str(agent_results["chat"])
                
            elif "run" in agent_results:
                # WebScrapingAgent responses - extract main_content
                run_result = agent_results["run"]
                if isinstance(run_result, dict):
                    response = run_result.get("main_content", 
                            run_result.get("summary", str(run_result)))
                else:
                    response = str(run_result)
                    
            else:
                # Generic handling for other agent types
                first_key = list(agent_results.keys())[0]
                first_result = agent_results[first_key]
                response = str(first_result)
            
            # Clean up response for better readability
            # response = self._clean_response_text(response)
            
            # Telegram character limit handling
            if len(response) > 3500:
                response = response[:3500] + "\n\n...(response truncated for length)"
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting agent response: {e}")
            return "I processed your request, but had trouble formatting the response."

    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting SuperAgent Flask Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Configuration class
class SuperAgentConfig:
    """Configuration for the SuperAgent Flask application"""
    API_HOST = os.getenv('SUPERAGENT_HOST', 'localhost')
    API_PORT = int(os.getenv('SUPERAGENT_PORT', 5000))
    BOT_USERNAME = os.getenv('BOT_USERNAME', 'SuperAgentBot')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

if __name__ == '__main__':
    try:
        # Create and run the Flask application
        app = SuperAgentFlaskApp()
        app.run(
            host=SuperAgentConfig.API_HOST,
            port=SuperAgentConfig.API_PORT,
            debug=SuperAgentConfig.DEBUG
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Flask server stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error starting Flask server: {e}")
        raise




# import os
# # from google import genai
# import re
# from openai import OpenAI 
# from dotenv import load_dotenv
# load_dotenv()


# # key = os.getenv("GEMINI_API_KEY")
# # client_gemini = genai.Client(key=key)

# deep_seek_key = os.getenv("DEEPSEEK_API_KEY")

# open_router_key = os.getenv("OPEN_ROUTER_KEY")
# open_router_url = "https://openrouter.ai/api/v1"

# # client = OpenAI(api_key=open_router_key, base_url=open_router_url)



# class RemoteLLM:
#     def __init__(self, model_name = "deepseek/deepseek-chat-v3-0324:free" , key = open_router_key):
#         self.model_name = model_name
#         self.llm = OpenAI(api_key=key, base_url=open_router_url)
#         self.requests = []

#     def prompt(self, query):
#         self.requests.append(query)
#         response = self.llm.chat.completions.create(
#             model = self.model_name,
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant"},
#                 {"role": "user", "content": query},
#                 ],
#             stream=False
#             )
#         print(response.choices[0].message.content)
#         return response.choices[0].message.content