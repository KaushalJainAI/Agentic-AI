from dotenv import load_dotenv
from typing import Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os 

import requests

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

os.environ["GEMINI_API_KEY"] = api_key


class Chatbot:
    """
    A chatbot class that wraps LangGraph functionality with Google Gemini model.
    
    Attributes:
        model: The model name to use
        model_provider: The model provider (e.g., "google_genai")
        temperature: Temperature setting for the model
        api_key: API key for authentication
        llm: The initialized chat model
        graph: The compiled LangGraph
    """

    def __init__(
        self, 
        model: str = "gemini-2.5-flash-lite-preview-06-17",
        model_provider: str = "google_genai",
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Chatbot with specified parameters.
        
        Args:
            model: Model name to use
            model_provider: Provider for the model
            temperature: Temperature setting for response generation
            api_key: API key (if None, will try to get from environment)
        """
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set LLM_API_KEY in environment variable.")
        
        # Initialize the LLM
        self.llm = init_chat_model(
            model=self.model,
            model_provider=self.model_provider,
            temperature=self.temperature,
            api_key=self.api_key
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build and compile the LangGraph."""
        
        class State(TypedDict):
            """
            Represents the state of our graph.
            
            Attributes:
                messages: List of messages in the conversation
            """
            messages: Annotated[list, add_messages]
        
        graph_builder = StateGraph(State)
        
        def chatbot_node(state: State):
            """Process messages through the LLM."""
            return {
                "messages": [self.llm.invoke(state["messages"])]
            }
        
        graph_builder.add_node("chatbot", chatbot_node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        
        return graph_builder.compile()
    
    def chat(self, message: str) -> str:
        """
        Send a message to the chatbot and get a response.
        
        Args:
            message: The user's message
            
        Returns:
            The chatbot's response as a string
        """
        state = self.graph.invoke({
            "messages": [{"role": "user", "content": message}]
        })
        
        return state['messages'][-1].content
    
    def chat_with_history(self, messages: list) -> str:
        """
        Chat with conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The chatbot's response as a string
        """
        state = self.graph.invoke({"messages": messages})
        return state['messages'][-1].content
    
    def interactive_chat(self):
        """
        Start an interactive chat session.
        Type 'quit', 'exit', or 'bye' to end the session.
        """
        print("Chatbot initialized! Type 'quit', 'exit', or 'bye' to end the session.")
        print("-" * 50)
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            try:
                response = self.chat(user_input)
                print(f"Bot: {response}")
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.")
    
    def update_temperature(self, temperature: float):
        """
        Update the temperature setting and rebuild the model.
        
        Args:
            temperature: New temperature value
        """
        self.temperature = temperature
        self.llm = init_chat_model(
            model=self.model,
            model_provider=self.model_provider,
            temperature=self.temperature,
            api_key=self.api_key
        )
        self.graph = self._build_graph()

