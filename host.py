import os
import logging
from flask import Flask, request, jsonify
from datetime import datetime
from typing import Dict, Any

import re

# Import your SuperAgent Orchestrator
from langgraph_super_agent import SuperAgentOrchestrator
from agents import (
    Chatbot, WebSearchingAgent, DatabaseQueryOrchestrator, 
    VectorKnowledgeAgent
)
ppx_api_key = os.getenv("PERPLEXITY_API_KEY")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SuperAgentFlaskApp:
    """
    Flask application that integrates the LangGraph SuperAgent Orchestrator
    with synchronous processing for Telegram bot communication.
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
                "Chatbot": Chatbot(use_local=False, model_provider = 'perplexity', model= 'sonar', local_model="qwen3:4b", api_key=ppx_api_key), 
                "DatabaseOrchestrator": DatabaseQueryOrchestrator(), 
                "WebScrapingAgent": WebSearchingAgent(use_local=True, model_provider = 'perplexity', model= 'sonar', local_model="qwen3:4b", max_website_count=5, api_key=ppx_api_key),
                # "VectorKnowlwdgeAgent": VectorKnowledgeAgent() 
                # Add other agents as needed
            }
            
            # Create the orchestrator
            self.orchestrator = SuperAgentOrchestrator(agents)
            logger.info("SuperAgent Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def _setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return self.health_check()
        
        @self.app.route('/chat', methods=['POST'])
        def chat_endpoint():
            return self.chat_endpoint()
        
        @self.app.route('/clearMem', methods=['POST'])
        def clear_memory():
            return jsonify({"success": self.orchestrator.clear_memory()})
    
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
        and returns immediate response. No queuing - the Telegram bot handles that.
        
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
                        "workflow_status": result.get("status", "unknown"),
                        "completed_steps": result.get("completed_steps", []),
                        "thread_id": result.get("thread_id"),
                        # "agent_used": self._extract_agent_used(result)
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
                    }
                }), 200
                
        except Exception as e:
            logger.error(f"Unexpected error in chat endpoint: {e}")
            return jsonify({
                "reply": "I'm experiencing technical difficulties. Please try again in a moment.",
                "success": False,
                "error": "Internal server error"
            }), 500
    
    def _format_agent_response(self, result: Dict[str, Any]) -> str:
        """Format agent response for Telegram display"""
        try:
            agent_results = result.get('results', {})
            
            if not agent_results:
                return "I processed your request, but didn't generate a specific response."
            
            # Handle different agent response types based on your actual output
            if "chat" in agent_results:
                # Chatbot responses - direct text
                response = str(agent_results["chat"])
                
            elif "run" in agent_results:
                # WebScrapingAgent responses - extract main_content
                run_result = agent_results["run"]
                if isinstance(run_result, dict):
                    response = run_result.get("main_content", 
                            run_result.get("summary", str(run_result))) + run_result.get("summary", "")
                else:
                    response = str(run_result)
                    
            else:
                # Generic handling for other agent types
                first_key = list(agent_results.keys())[0]
                first_result = agent_results[first_key]
                response = str(first_result)
            
            # Telegram character limit handling
            if len(response) > 13500:
                response = response[:13500] + "\n\n...(response truncated for length)"

            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            print(response)            

            return response
            
        except Exception as e:
            logger.error(f"Error formatting agent response: {e}")
            return "I processed your request, but had trouble formatting the response."
    
    def _extract_agent_used(self, result: Dict[str, Any]) -> str:
        """Extract which agent was used from the result"""
        try:
            agent_results = result.get('results', {})
            if agent_results:
                # Return the first agent that produced results
                return list(agent_results.keys())[0]
            return "unknown"
        except Exception:
            return "unknown"
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting SuperAgent Flask Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

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
