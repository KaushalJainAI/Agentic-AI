# SuperAgent Configuration File
import os
from typing import Dict, Any

class SuperAgentConfig:
    """Configuration settings for SuperAgent system"""
    
    # API Keys (set these in your .env file)
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    TELEGRAM_BOT_API_KEY = os.getenv("TELEGRAM_BOT_API_KEY", "")
    
    # Model Configuration
    DEFAULT_MODEL = "gemini-2.5-pro"
    MODEL_PROVIDER = "google_genai"
    TEMPERATURE = 0.7
    
    # Directory Paths
    DATABASE_DIRECTORY = "./Databases"
    KNOWLEDGE_BASE_PATH = "./knowledge_base"
    MEMORY_DB_PATH = "./conversation_memory.db"
    
    # API Configuration
    API_HOST = "127.0.0.1"
    API_PORT = 5000
    
    # Telegram Bot Configuration
    BOT_USERNAME = "@KJ_personal_assist_bot"
    
    # Agent Settings
    AGENT_SETTINGS = {
        "vector_knowledge": {
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "similarity_threshold": 0.3
        },
        "web_scraping": {
            "max_urls": 20,
            "timeout": 10,
            "max_content_length": 4000
        },
        "database_query": {
            "max_results": 50,
            "query_timeout": 30
        }
    }
    
    # Human-in-the-loop settings
    HUMAN_APPROVAL_REQUIRED_FOR = [
        "database_modifications",
        "file_operations",
        "external_api_calls",
        "sensitive_data_queries"
    ]
    
    # Memory settings
    MAX_CONVERSATION_HISTORY = 20
    MEMORY_RETENTION_DAYS = 30
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        if not cls.LLM_API_KEY:
            issues.append("LLM_API_KEY not set")
        
        if not cls.TELEGRAM_BOT_API_KEY:
            issues.append("TELEGRAM_BOT_API_KEY not set")
        
        # Check if directories exist
        if not os.path.exists(cls.DATABASE_DIRECTORY):
            os.makedirs(cls.DATABASE_DIRECTORY, exist_ok=True)
            
        if not os.path.exists(os.path.dirname(cls.KNOWLEDGE_BASE_PATH)):
            os.makedirs(os.path.dirname(cls.KNOWLEDGE_BASE_PATH), exist_ok=True)
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config_summary": {
                "model": cls.DEFAULT_MODEL,
                "api_host": f"{cls.API_HOST}:{cls.API_PORT}",
                "database_dir": cls.DATABASE_DIRECTORY,
                "knowledge_base": cls.KNOWLEDGE_BASE_PATH
            }
        }