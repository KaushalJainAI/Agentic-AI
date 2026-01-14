from dotenv import load_dotenv
from typing import Annotated, Literal, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, create_model, field_validator, ConfigDict
from langchain.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup
import os 
import glob
import sqlite3
import logging 
import json
import pandas as pd
import requests
import io
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime
import ollama
from langchain_ollama import ChatOllama


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
        api_key: Optional[str] = None,
        use_local: bool = False,
        local_model: str = "qwen3:4b",
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
        self.use_local = use_local
        self.local_model = local_model
        
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set LLM_API_KEY in environment variable.")
        
        if use_local:
            self.llm = ChatOllama(
                model=local_model,
                temperature=temperature
            )
        else:
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
    
    # def chat_with_history(self, messages: list) -> str:
    #     """
    #     Chat with conversation history.
        
    #     Args:
    #         messages: List of message dictionaries with 'role' and 'content' keys
            
    #     Returns:
    #         The chatbot's response as a string
    #     """
    #     state = self.graph.invoke({"messages": messages})
    #     return state['messages'][-1].content
    
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
                return response
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.")
                return None

    def update_temperature(self, temperature: float):
        """
        Update the temperature setting and rebuild the model.
        
        Args:
            temperature: New temperature value
        """
        self.temperature = temperature
        if self.use_local:
            self.llm = ChatOllama(
                model=self.local_model,
                temperature=self.temperature
            )
        else:
            self.llm = init_chat_model(
                model=self.model,
                model_provider=self.model_provider,
                temperature=self.temperature,
                api_key=self.api_key
            )
        self.graph = self._build_graph()

        return f"Temperature updated successfully to {temperature}."


from tavily import TavilyClient

class SchemaDefinition(BaseModel):
    """
    The definition for a single field in the output schema.
    This explicit model prevents the KeyError for 'type' or 'description'.
    """
    type: str = Field(..., description="The Python type for the field, e.g., 'str', 'int', 'float', 'List[str]'.")
    description: str = Field(..., description="A clear, single-sentence description of the field's purpose.")


class Plan(BaseModel):
    """
    The complete plan for research, including a search query and a structured output schema.
    Using SchemaDefinition here provides strong validation.
    """
    search_query: str = Field(..., description="A concise and effective search engine query designed to find the required information.")
    PlanSchema: Dict[str, SchemaDefinition] = Field(..., description="The structured output schema, where each key is a field name and the value defines its type and description.")
    num_websites: int = Field(default=5, description="Number of websites to scrape (2 for basic facts, 5+ for deep research)")
    deep_research: bool = Field(default=False, description="True when the query requires exploring many websites to generate report")


class GraphState(BaseModel):
    """Represents the state of our graph."""
    prompt: str = Field(default="")
    search_query: str = Field(default="")
    PlanSchema: Dict[str, Any] = Field(default_factory=dict)
    urls: List[str] = Field(default_factory=list)
    scraped_content: str = Field(default="")
    structured_output: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)
    research_summary: str = Field(default="")
    num_websites: int = Field(default=5)
    deep_research: bool = Field(default=False)


class WebSearchingAgent:
    """
    An agent that dynamically scrapes web content based on a user prompt.
    """
    def __init__(
        self, 
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        max_website_count: int = 10,
        local_model: str = "qwen3:4b",
        use_local: bool = False,
    ):
        """
        Initialize the WebScrapingAgent with specified parameters.

        Args:
            model: Model name to use
            model_provider: Provider for the model
            temperature: Temperature setting for response generation
            api_key: API key (if None, will try to get from environment)
            tavily_api_key: Tavily API key for web search
            max_website_count: Maximum number of websites to search
            local_model: Local model name if using local
            use_local: Whether to use local model
        """
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.max_website_count = max_website_count
        self.use_local = use_local
        self.local_model = local_model

        if not self.api_key and not use_local:
            raise ValueError("API key is required. Provide it directly or set LLM_API_KEY in environment variable.")

        if use_local:
            self.llm = ChatOllama(
                model=local_model,
                temperature=temperature
            )
        else:
            self.llm = init_chat_model(
                model=self.model,
                model_provider=self.model_provider,
                temperature=self.temperature,
                api_key=self.api_key
            )

        # Initialize Tavily client
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

        # Compile the graph and store it as an instance variable
        self.app = self._build_graph()


    def _build_graph(self):
        """Builds and compiles the LangGraph workflow with parallel branches."""
        workflow = StateGraph(GraphState)
        workflow.add_node("plan_node", self.plan_node)
        workflow.add_node("search_node", self.search_node)
        workflow.add_node("scrape_node", self.scrape_node)
        workflow.add_node("merge_node", self.merge_node)
        workflow.add_node("extract_node", self.extract_node)

        # Parallel branches - Fixed routing function
        def route_to_parallel(state):
            return ["plan_node", "search_node"]  # Route to both for parallel execution

        workflow.add_conditional_edges(START, route_to_parallel)

        # Search branch: search -> merge
        workflow.add_edge("search_node", "merge_node")

        # Plan branch: plan -> merge  
        workflow.add_edge("plan_node", "merge_node")

        # Scraping after taking the decision how many nodes to scrape
        workflow.add_edge("merge_node", "scrape_node")

        # After merge, proceed to extract
        workflow.add_edge("scrape_node", "extract_node")
        workflow.add_edge("extract_node", END)

        return workflow.compile()


    def _scrape_website(self, url: str) -> str:
        """Internal method to scrape a single webpage."""
        print(f"--- SCRAPING: {url} ---")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join(soup.get_text().split())
            return f"Successfully scraped content from {url}:\n\n{text[:4000]}"
        except requests.RequestException as e:
            return f"Failed to scrape {url}. Error: {e}"


    def plan_node(self, state: GraphState) -> Dict[str, Any]:
        """Node to generate a search query and a Pydantic PlanSchema."""
        print("--- PLAN ---")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
            """You are an expert scraping agent. Your task is to create a focused search query and a detailed PlanSchema to answer the user's request.

            For the PlanSchema field, return a dictionary where:
            - Each key is a field name (string)
            - Each value is an object with exactly two properties:
              * "type": a string representing the Python type (e.g., "str", "int", "List[str]", "Optional[str]")
              * "description": a string describing what this field represents

            Keep the PlanSchema focused with 3-6 relevant fields maximum.

            For num_websites:
            - Use 2 for simple factual queries (like "What is the capital of France?")  
            - Use 5-8 for moderate research (like "Recent developments in AI")
            - Use 8+ for deep research requiring comprehensive analysis

            Set deep_research=True for queries requiring analysis of multiple sources, comparisons, or comprehensive reports.

            Ensure the output strictly matches the Plan model structure to avoid errors."""),
            ("user", "Research request: {prompt}")
        ])

        try:
            planner = prompt_template | self.llm.with_structured_output(Plan)
            plan_result = planner.invoke({"prompt": state.prompt})

            # Convert SchemaDefinition objects to simple dictionaries
            schema_as_dict = {}
            for key, schema_def in plan_result.PlanSchema.items():
                schema_as_dict[key] = {
                    "type": schema_def.type,
                    "description": schema_def.description
                }

            return {
                "search_query": plan_result.search_query, 
                "PlanSchema": schema_as_dict,
                "num_websites": plan_result.num_websites,
                "deep_research": plan_result.deep_research,
            }

        except Exception as e:
            print(f"Error in plan_node: {e}")
            return {
                "search_query": state.prompt,  # Use prompt as fallback query
                "PlanSchema": {
                    "main_content": {
                        "type": "str", 
                        "description": "Main content extracted from the research"
                    },
                    "summary": {
                        "type": "str", 
                        "description": "Brief summary of the findings"
                    }
                },
                "num_websites": 3,
                "deep_research": False,
                "error_message": str(e)
            }


    def search_node(self, state: GraphState) -> Dict[str, Any]:
        """Node to perform a web search using prompt as fallback."""
        print("--- SEARCH ---")
        query = state.search_query if state.search_query else state.prompt  # Fallback to prompt for parallelism

        # Use the planned number of websites or default
        max_results = min(state.num_websites or self.max_website_count, self.max_website_count)

        search_results = self.tavily_client.search(query=query, max_results=max_results)
        urls = [result['url'] for result in search_results['results']]
        return {"urls": urls}


    def scrape_node(self, state: GraphState) -> Dict[str, Any]:  # Fixed signature - removed unused plan parameter
        """
        Node to scrape content from all provided URLs and combine them.
        """
        if state.error_message or not state.urls:
            print("Skipping scrape due to previous failure or no URLs.")
            return {"scraped_content": "", "error_message": "No URLs to scrape."}

        print("--- SCRAPE (Cumulative) ---")
        urls_to_try = state.urls[:state.num_websites] if state.num_websites else state.urls  # Use planned number
        all_scraped_content = []

        for i, url in enumerate(urls_to_try):
            print(f"Scraping URL {i + 1}/{len(urls_to_try)}: {url}")
            content = self._scrape_website(url)

            # Check if the scrape was successful and add it to our list
            if "Successfully scraped content" in content:
                print(f"--- Scrape successful for {url} ---")
                all_scraped_content.append(content)
            else:
                print(f"--- Scrape failed for {url}: {content.splitlines()[0]} ---")

        # If no content was gathered from any URL
        if not all_scraped_content:
            print("--- All scraping attempts failed. ---")
            return {"scraped_content": "", "error_message": "All scraping attempts failed."}

        # Combine all successful scrapes into one large string
        cumulative_content = "\n\n--- NEW SOURCE ---\n\n".join(all_scraped_content)
        print("--- Finished cumulative scraping. ---")
        return {"scraped_content": cumulative_content}


    def merge_node(self, state: GraphState) -> Dict[str, Any]:
        """Node to merge results from plan and search branches."""
        print("--- MERGE ---")

        # If we have both search query and URLs, we're good to proceed
        if state.search_query and state.urls:
            print(f"--- Merge successful: Query '{state.search_query}' with {len(state.urls)} URLs ---")
            return {}

        # If search failed but we have a query from planning, try search again
        if state.search_query and not state.urls:
            print(f"--- Re-running search with planned query: '{state.search_query}' ---")
            try:
                max_results = min(state.num_websites or self.max_website_count, self.max_website_count)
                search_results = self.tavily_client.search(query=state.search_query, max_results=max_results)
                urls = [result['url'] for result in search_results['results']]
                return {"urls": urls}
            except Exception as e:
                return {"error_message": f"Refined search failed: {e}"}

        # If neither worked, return error
        return {"error_message": "Both planning and search phases failed"}


    def _safe_type_conversion(self, type_str: str) -> type:
        """Safely convert string type representation to actual type."""
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'List[str]': List[str],
            'List[int]': List[int],
            'List[float]': List[float],
            'Optional[str]': Optional[str],
            'Optional[int]': Optional[int],
            'Optional[float]': Optional[float],
            'Dict[str, Any]': Dict[str, Any],
            'Dict[str, str]': Dict[str, str],
        }
        return type_mapping.get(type_str, str)  # Default to str if not found


    def extract_node(self, state: GraphState) -> Dict[str, Any]:
        """Node to extract information based on the dynamic PlanSchema."""
        print("--- EXTRACT ---")

        try:
            # Create dynamic model with safe type conversion - FIXED field definitions format
            field_definitions = {}

            # Add research_summary field properly
            field_definitions["research_summary"] = (str, Field(description="Summary of research findings"))

            # Add planned schema fields
            for key, val in state.PlanSchema.items():
                field_type = self._safe_type_conversion(val['type'])
                field_definitions[key] = (field_type, Field(description=val['description']))

            DynamicModel = create_model('DynamicModel', **field_definitions)

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert data extractor. Extract the relevant information from the provided text that precisely answers the user's goal and format it according to the provided PlanSchema."),
                ("user", "User Goal: {prompt}\n\nWebpage Content:\n{content}")
            ])

            extractor = prompt_template | self.llm.with_structured_output(DynamicModel)
            extracted_data = extractor.invoke({
                "prompt": state.prompt, 
                "content": state.scraped_content
            })

            extracted_dict = extracted_data.model_dump()
            research_summary = extracted_dict.pop("research_summary", "")

            return {
                "structured_output": extracted_dict,
                "research_summary": research_summary
            }

        except Exception as e:
            print(f"Error in extract_node: {e}")
            return {
                "structured_output": {"error": f"Extraction failed: {str(e)}"},
                "research_summary": "Failed to generate summary due to extraction error."
            }


    def run(self, prompt: str):
        """The main entry point to run the agent."""
        inputs = GraphState(prompt=prompt)
        final_state_dict = self.app.invoke(inputs.model_dump())  # LangGraph works with dicts internally
        # Validate final state with BaseModel for safety
        final_state = GraphState(**final_state_dict)

        output = {
            "structured_output": final_state.structured_output,
            "research_summary": final_state.research_summary,
            "sources": final_state.urls,
            "deep_research": final_state.deep_research,
            "websites_scraped": len([url for url in final_state.urls if url]) if final_state.urls else 0
        }

        return json.dumps(output, indent=2)


    def run_and_stream_watch(self, prompt: str):
        """
        The main entry point to run the agent with streaming output.
        This method streams the output of each node to the console.
        """
        inputs = GraphState(prompt=prompt)
        final_state_dict = {}

        # Use app.stream() to see the output of each node
        print("--- 🚀 Starting Agent Run ---")
        for output in self.app.stream(inputs.model_dump(), {"recursion_limit": 10}):
            for key, value in output.items():
                print(f"\n--- ✅ Output from node: {key} ---")
                print(json.dumps(value, indent=2, ensure_ascii=False))
                final_state_dict.update(value)  # Merge dict updates

        print("\n--- 🏁 Agent Finished ---")
        # Validate final state with BaseModel
        final_state = GraphState(**final_state_dict)

        output = {
            "structured_output": final_state.structured_output,
            "research_summary": final_state.research_summary,
            "sources": final_state.urls,
            "deep_research": final_state.deep_research,
            "websites_scraped": len([url for url in final_state.urls if url]) if final_state.urls else 0
        }

        return json.dumps(output, indent=2)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryPlan(BaseModel):
    """Model for SQL query planning."""
    sql_query: str = Field(description="The SQL query to execute")
    reasoning: str = Field(description="Reasoning behind the query structure")
    expected_columns: List[str] = Field(description="Expected columns in the result")

class ExecutionStep(BaseModel):
    """Model for a single execution step in flat file processing."""
    operation: str = Field(description="Operation type: load, merge, filter, groupby, select")
    args: Dict[str, Any] = Field(description="Arguments for the operation")
    result_df: str = Field(description="Name of the resulting dataframe")
    description: str = Field(description="Human readable description of this step")

class ExecutionPlan(BaseModel):
    """Model for flat file execution planning."""
    steps: List[ExecutionStep] = Field(description="List of execution steps")
    reasoning: str = Field(description="Overall reasoning for the execution plan")
    expected_result: str = Field(description="Description of expected final result")




class DatabaseDiscoveryAgent:
    """
    Agent that finds the most relevant database (SQL or flat file) based on the query.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        database_directory: str = "./Databases",
        local_model: str = "qwen3:4b",
        use_local: bool = False,
    ):
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.database_directory = database_directory
        self.use_local = use_local
        self.local_model = local_model
        

        
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set LLM_API_KEY in environment variable.")
        
        try:
            if use_local:
                self.llm = ChatOllama(
                    model=local_model,
                    temperature=temperature
                )
            else:
                self.llm = init_chat_model(
                    model=self.model,
                    model_provider=self.model_provider,
                    temperature=self.temperature,
                    api_key=self.api_key
                )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")

    def _get_all_databases(self) -> List[Dict[str, Any]]:
        """Get all databases (both SQL and flat files) with their descriptions."""
        databases = []
        
        # SQL database extensions
        sql_extensions = ['*.db', '*.sqlite', '*.sqlite3']
        
        # Flat file extensions
        flat_extensions = ['*.csv', '*.tsv', '*.xlsx', '*.json']
        
        all_extensions = sql_extensions + flat_extensions
        
        for extension in all_extensions:
            pattern = os.path.join(self.database_directory, '**', extension)
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                db_name = os.path.basename(file_path)
                db_directory = os.path.dirname(file_path)
                description = "No description file found."
                
                # Look for description.txt in the same directory
                description_file = os.path.join(db_directory, 'description.txt')
                if os.path.exists(description_file):
                    try:
                        with open(description_file, 'r', encoding='utf-8') as f:
                            description = f.read().strip()
                    except Exception as e:
                        description = f"Error reading description file: {e}"
                
                # Determine database type
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in ['.db', '.sqlite', '.sqlite3']:
                    db_type = 'sql'
                else:
                    db_type = 'flat_file'
                
                databases.append({
                    "file_path": file_path,
                    "name": db_name,
                    "description": description,
                    "type": db_type,
                    "directory": db_directory
                })
        
        return databases

    def select_best_database(self, query: str) -> Dict[str, Any]:
        """
        Select the most relevant database for the given query.
        Returns database info and recommended agent type.
        """
        try:
            available_databases = self._get_all_databases()
            
            if not available_databases:
                return {"error": "No databases found in the specified directory"}
            
            # Create safe descriptions for the LLM prompt
            safe_db_lines = []
            for db in available_databases:
                safe_name = db['name'].replace('{', '{{').replace('}', '}}')
                safe_desc = db['description'].replace('{', '{{').replace('}', '}}')
                safe_type = db['type'].replace('{', '{{').replace('}', '}}')
                safe_db_lines.append(f"- {safe_name} ({safe_type}): {safe_desc}")
            
            db_list = "\n".join(safe_db_lines)
            
            class DatabaseRecommendation(BaseModel):
                selected_database_name: str = Field(description="Name of the selected database file")
                database_type: str = Field(description="Type of database: 'sql' or 'flat_file'")
                recommended_agent: str = Field(description="Recommended agent: 'SQL_Agent' or 'FlatFile_Agent'")
                reasoning: str = Field(description="Reasoning for the selection")
                confidence_score: float = Field(description="Confidence in the selection (0-1)")
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", 
                f"""You are a database selection expert. Given a user query and a list of available databases, select the most relevant database and recommend the appropriate agent.

                Available databases:
                {db_list}

                For each database, you can see:
                - Database name
                - Type (sql for SQLite databases, flat_file for CSV/Excel/JSON files)
                - Description from the description.txt file

                Rules:
                - If you select a 'sql' type database, recommend 'SQL_Agent'
                - If you select a 'flat_file' type database, recommend 'FlatFile_Agent'
                - Choose based on which database description best matches the user's query
                - Provide a confidence score between 0 and 1"""),
                ("user", "User Query: {query}")
            ])
            
            selector = prompt_template | self.llm.with_structured_output(DatabaseRecommendation)
            recommendation = selector.invoke({"query": query})
            
            # Find the full database info
            selected_db = None
            for db in available_databases:
                if db["name"] == recommendation.selected_database_name:
                    selected_db = db
                    break
            
            if not selected_db:
                return {"error": f"Could not find selected database: {recommendation.selected_database_name}"}
            
            return {
                "database_info": selected_db,
                "recommended_agent": recommendation.recommended_agent,
                "reasoning": recommendation.reasoning,
                "confidence_score": recommendation.confidence_score
            }
            
        except Exception as e:
            return {"error": f"Failed to select database: {str(e)}"}


class SQLQueryAgent:
    """
    Agent focused purely on executing SQL queries on a given database.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        local_model: str = "qwen3:4b",
        use_local: bool = False,
    ):
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.use_local = use_local
        self.local_model = local_model

        
        if not self.api_key:
            raise ValueError("API key is required.")
        
        try:
            if use_local:
                self.llm = ChatOllama(
                    model=local_model,
                    temperature=temperature
                )
            else:
                self.llm = init_chat_model(
                    model=self.model,
                    model_provider=self.model_provider,
                    temperature=self.temperature,
                    api_key=self.api_key
                )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")

    def _get_database_schema(self, db_path: str) -> Dict[str, Any]:
        """Get the schema of a SQLite database."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema = {}
            for table_tuple in tables:
                table_name = table_tuple[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                schema[table_name] = {
                    'columns': [
                        {
                            'name': col[1],
                            'type': col[2],
                            'not_null': bool(col[3]),
                            'primary_key': bool(col[5])
                        }
                        for col in columns
                    ]
                }
                
                # Get sample data (first 3 rows)
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                sample_data = cursor.fetchall()
                schema[table_name]['sample_data'] = sample_data
            
            conn.close()
            return schema
            
        except Exception as e:
            logger.error(f"Error getting schema for {db_path}: {e}")
            return {}

    def query_database(self, db_path: str, user_query: str) -> Dict[str, Any]:
        """
        Execute a query on the specified SQL database.
        """
        try:
            # Get database schema
            schema = self._get_database_schema(db_path)
            if not schema:
                return {"error": "Could not analyze database schema"}
            
            # Format schema for LLM
            schema_description = ""
            for table_name, table_info in schema.items():
                schema_description += f"\nTable: {table_name}\n"
                schema_description += "Columns:\n"
                for col in table_info["columns"]:
                    schema_description += f"  - {col['name']} ({col['type']})\n"
                
                if table_info.get("sample_data"):
                    schema_description += "Sample data:\n"
                    for row in table_info["sample_data"][:2]:
                        schema_description += f"  {row}\n"
            
            # Generate SQL query
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", 
                 f"""You are an expert SQL query generator. Given a user request and database schema, generate an appropriate SQL query.
                 
                 Database Schema:
                 {schema_description}
                 
                 Rules:
                 - Generate valid SQLite syntax
                 - Use appropriate JOINs if multiple tables are needed
                 - Include LIMIT clauses if the result might be large
                 - Be precise and only query what's needed"""),
                ("user", "Query request: {query}")
            ])
            
            query_generator = prompt_template | self.llm.with_structured_output(QueryPlan)
            query_plan = query_generator.invoke({"query": user_query})
            
            # Execute the query
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(query_plan.sql_query)
            rows = cursor.fetchall()
            
            # Convert rows to list of dictionaries
            results = []
            for row in rows:
                results.append(dict(row))
            
            conn.close()
            
            # Format results
            class FormattedOutput(BaseModel):
                summary: str = Field(description="Summary of the findings")
                total_records: int = Field(description="Number of records found")
                key_findings: List[str] = Field(description="Key insights from the data")
                sql_query_used: str = Field(description="The SQL query that was executed")
                raw_data: List[Dict[str, Any]] = Field(description="Raw query results")
            
            if not results:
                return {
                    "summary": "No results found for the query",
                    "total_records": 0,
                    "key_findings": ["No data matched the query criteria"],
                    "sql_query_used": query_plan.sql_query,
                    "raw_data": []
                }
            
            # Format the results using LLM
            format_template = ChatPromptTemplate.from_messages([
                ("system", 
                 """You are a data formatter. Given query results and the original user request, format the data meaningfully."""),
                ("user", "Original request: {query}\n\nQuery results: {results}")
            ])
            
            formatter = format_template | self.llm.with_structured_output(FormattedOutput)
            formatted_output = formatter.invoke({
                "query": user_query,
                "results": json.dumps(results, indent=2)
            })
            
            # Add the SQL query to the output
            output_dict = formatted_output.dict()
            output_dict["sql_query_used"] = query_plan.sql_query
            
            return output_dict
            
        except Exception as e:
            return {"error": f"Failed to query database: {str(e)}"}

class FlatFileQueryAgent:
    """
    Agent focused purely on querying flat files (CSV, TSV, Excel, JSON).
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        local_model: str = "qwen3:4b",
        use_local: bool = False,
    ):
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.use_local = use_local
        self.local_model = local_model
        


        if not self.api_key:
            raise ValueError("API key is required.")
        
        try:
            if use_local:
                self.llm = ChatOllama(
                    model=local_model,
                    temperature=temperature
                )
            else:
                self.llm = init_chat_model(
                    model=self.model,
                    model_provider=self.model_provider,
                    temperature=self.temperature,
                    api_key=self.api_key
                )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")

    def _analyze_file_directory(self, directory_path: str) -> str:
        """Analyze all files in a directory and return schema summary."""
        file_extensions = ['*.csv', '*.tsv', '*.xlsx', '*.json']
        all_files = []
        
        for ext in file_extensions:
            pattern = os.path.join(directory_path, ext)
            all_files.extend(glob.glob(pattern))
        
        if not all_files:
            return "No data files found in the directory."
        
        schema_details = ["## Available Files and their Schemas:"]
        file_columns = {}
        
        # Analyze each file
        for file_path in all_files:
            filename = os.path.basename(file_path)
            try:
                # Read the file
                if filename.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                elif filename.endswith('.json'):
                    df = pd.read_json(file_path)
                else:
                    sep = '\t' if filename.endswith('.tsv') else ','
                    df = pd.read_csv(file_path, sep=sep)
                
                # Get file info
                with io.StringIO() as buffer, redirect_stdout(buffer):
                    df.info()
                    info_string = buffer.getvalue()
                
                schema_details.append(f"\n### File: `{filename}`")
                schema_details.append(f"``````")
                
                file_columns[filename] = df.columns.tolist()
                
            except Exception as e:
                schema_details.append(f"\n- Could not analyze {filename}. Error: {e}")
        
        # Analyze relationships
        column_to_files = defaultdict(list)
        for filename, columns in file_columns.items():
            for column in columns:
                column_to_files[column].append(filename)
        
        relationship_summary = ["\n## Potential Relationships:"]
        found_relationships = False
        for column, files in column_to_files.items():
            if len(files) > 1:
                relationship_summary.append(f"- **Column `{column}`** found in: `{', '.join(files)}`")
                found_relationships = True
        
        if not found_relationships:
            relationship_summary.append("- No obvious relationships found.")
        
        return "\n".join(schema_details + relationship_summary)

    def query_files(self, directory_path: str, user_query: str) -> Dict[str, Any]:
        """
        Query flat files in the specified directory.
        """
        try:
            # Analyze file schemas
            schema_summary = self._analyze_file_directory(directory_path)
            
            if "No data files found" in schema_summary:
                return {"error": "No data files found in the directory"}
            
            # Generate execution plan
            prompt_template = ChatPromptTemplate.from_messages([
                ("system",
                 f"""You are a data analysis expert. Create a step-by-step pandas execution plan.
                 
                 Available Operations:
                 - `load`: Load file. args: {{"filename": "file.csv"}}
                 - `merge`: Join DataFrames. args: {{"left": "df1", "right": "df2", "on": "column", "how": "inner"}}
                 - `filter`: Filter rows. args: {{"df_name": "df1", "query_string": "column > 100"}}
                 - `groupby`: Group and aggregate. args: {{"df_name": "df1", "by": ["col1"], "agg": {{"col2": "sum"}}}}
                 - `select`: Select columns. args: {{"df_name": "df1", "columns": ["col1", "col2"]}}
                 
                 File Schemas:
                 {schema_summary}"""),
                ("user", "Question: {query}")
            ])
            
            planner = prompt_template | self.llm.with_structured_output(ExecutionPlan)
            plan = planner.invoke({"query": user_query, "schema_summary": schema_summary})
            
            # Execute the plan
            dataframes = {}
            final_df_name = ""
            
            for step in plan.steps:
                op = step.operation.lower()
                args = step.args
                final_df_name = step.result_df
                
                if op == 'load':
                    file_path = os.path.join(directory_path, args['filename'])
                    if file_path.endswith('.xlsx'):
                        dataframes[step.result_df] = pd.read_excel(file_path)
                    elif file_path.endswith('.json'):
                        dataframes[step.result_df] = pd.read_json(file_path)
                    else:
                        sep = '\t' if file_path.endswith('.tsv') else ','
                        dataframes[step.result_df] = pd.read_csv(file_path, sep=sep)
                elif op == 'merge':
                    left_df = dataframes[args['left']]
                    right_df = dataframes[args['right']]
                    dataframes[step.result_df] = pd.merge(left_df, right_df, on=args['on'], how=args.get('how', 'inner'))
                elif op == 'filter':
                    df = dataframes[args['df_name']]
                    dataframes[step.result_df] = df.query(args['query_string'])
                elif op == 'groupby':
                    df = dataframes[args['df_name']]
                    dataframes[step.result_df] = df.groupby(args['by']).agg(args['agg']).reset_index()
                elif op == 'select':
                    df = dataframes[args['df_name']]
                    dataframes[step.result_df] = df[args['columns']]
            
            final_df = dataframes.get(final_df_name)
            if final_df is None or final_df.empty:
                return {"message": "The query resulted in no data"}
            
            # Format results
            results_json = final_df.head(20).to_dict('records')
            
            return {
                "summary": f"Successfully processed {len(final_df)} records",
                "total_records": len(final_df),
                "execution_plan": plan.reasoning,
                "raw_data": results_json
            }
            
        except Exception as e:
            return {"error": f"Failed to query files: {str(e)}"}


class DatabaseQueryOrchestrator:
    """
    Orchestrator that coordinates between the three agents.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        database_directory: str = "./Databases",
        local_model: str = "qwen3:4b",
        use_local: bool = False,
    ):
        self.use_local = use_local
        self.local_model = local_model
        

        # Initialize all three agents
        self.discovery_agent = DatabaseDiscoveryAgent(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            api_key=api_key,
            database_directory=database_directory,
            local_model=local_model,
            use_local=use_local
        )
        
        self.sql_agent = SQLQueryAgent(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            api_key=api_key,
            local_model=local_model,
            use_local=use_local
        )
        
        self.flatfile_agent = FlatFileQueryAgent(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            api_key=api_key,
            local_model=local_model,
            use_local=use_local
        )
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Main entry point that orchestrates the three agents.
        """
        try:
            # Step 1: Discover the best database
            discovery_result = self.discovery_agent.select_best_database(user_query)
            
            if "error" in discovery_result:
                return discovery_result
            
            database_info = discovery_result["database_info"]
            recommended_agent = discovery_result["recommended_agent"]
            
            # Step 2: Route to appropriate agent
            if recommended_agent == "SQL_Agent":
                result = self.sql_agent.query_database(
                    database_info["file_path"], 
                    user_query
                )
            else:  # FlatFile_Agent
                result = self.flatfile_agent.query_files(
                    database_info["directory"], 
                    user_query
                )
            
            # Add discovery metadata to the result
            result["discovery_info"] = {
                "selected_database": database_info["name"],
                "database_type": database_info["type"],
                "reasoning": discovery_result["reasoning"],
                "confidence": discovery_result["confidence_score"]
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Orchestration failed: {str(e)}"}
        
    
    def run_and_stream_watch(self, prompt: str) -> Dict[str, Any]:
        """
        The main entry point to run the agent.
        This method now streams the output of each node to the console.
        """
        inputs = {"prompt": prompt}
        final_state = {}

        print("--- 🚀 Starting Agent Run ---")
        
        try:
            # Step 1: Discovery Agent Node
            print(f"\n--- ✅ Output from node: discovery_agent ---")
            discovery_result = self.discovery_agent.select_best_database(prompt)
            print(json.dumps(discovery_result, indent=2, ensure_ascii=False))
            final_state.update(discovery_result)
            
            if "error" in discovery_result:
                print("\n--- 🏁 Agent Finished (with error) ---")
                return {"error": discovery_result["error"]}
            
            database_info = discovery_result["database_info"]
            recommended_agent = discovery_result["recommended_agent"]
            
            # Step 2: Query Execution Agent Node
            agent_node_name = recommended_agent.lower().replace("_agent", "_node")
            print(f"\n--- ✅ Output from node: {agent_node_name} ---")
            
            if recommended_agent == "SQL_Agent":
                query_result = self.sql_agent.query_database(
                    database_info["file_path"], 
                    prompt
                )
            else:  # FlatFile_Agent
                query_result = self.flatfile_agent.query_files(
                    database_info["directory"], 
                    prompt
                )
            
            print(json.dumps(query_result, indent=2, ensure_ascii=False))
            final_state.update(query_result)
            
            # Step 3: Final Assembly Node
            print(f"\n--- ✅ Output from node: assembly_node ---")
            
            # Add discovery metadata to the result
            final_result = query_result.copy()
            final_result["discovery_info"] = {
                "selected_database": database_info["name"],
                "database_type": database_info["type"],
                "reasoning": discovery_result["reasoning"],
                "confidence": discovery_result["confidence_score"]
            }
            
            # Create structured output
            structured_output = {
                "structured_output": {
                    "status": "success",
                    "query": prompt,
                    "selected_database": database_info["name"],
                    "database_type": database_info["type"],
                    "agent_used": recommended_agent,
                    "summary": final_result.get("summary", "Query completed"),
                    "total_records": final_result.get("total_records", 0),
                    "key_findings": final_result.get("key_findings", []),
                    "execution_details": {
                        "sql_query_used": final_result.get("sql_query_used"),
                        "execution_plan": final_result.get("execution_plan"),
                        "confidence_score": discovery_result["confidence_score"]
                    },
                    "raw_data": final_result.get("raw_data", [])
                }
            }
            
            print(json.dumps(structured_output, indent=2, ensure_ascii=False))
            final_state.update(structured_output)
            
            print("\n--- 🏁 Agent Finished ---")
            return final_state.get('structured_output', {"error": "Agent failed to produce a structured output."})
            
        except Exception as e:
            error_output = {"error": f"Orchestration failed: {str(e)}"}
            print(f"\n--- ❌ Error in orchestration ---")
            print(json.dumps(error_output, indent=2, ensure_ascii=False))
            print("\n--- 🏁 Agent Finished (with error) ---")
            return error_output


import os
import pickle
import uuid
from typing import List, Dict, Any, Optional, Union
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pydantic import BaseModel, Field

# --- Multi-modal Content Types ---
class MultiModalContent(BaseModel):
    text: Optional[str] = None
    image_path: Optional[str] = None  # Local or remote path
    pdf_path: Optional[str] = None
    audio_path: Optional[str] = None
    transcript: Optional[str] = None

# --- Reasoning Chains (for each answer) ---
class ReasonChainStep(BaseModel):
    step: str
    evidence: Optional[str]
    next_step: Optional[int]

class ReasoningChain(BaseModel):
    chain: List[ReasonChainStep] = Field(default_factory=list)
    decision_tree: Optional[Dict[str, Any]] = None

class KnowledgeEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    solution: str
    category: str
    tags: List[str] = Field(default_factory=list)
    rating: int = 0  # Feedback: upvotes - downvotes
    metadata: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    content: Optional[MultiModalContent] = None
    reasoning: Optional[ReasoningChain] = None

class WorkflowMemory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str
    workflow_plan: List[str]
    final_answer: str
    category: str
    tags: List[str] = Field(default_factory=list)
    rating: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    content: Optional[MultiModalContent] = None
    reasoning: Optional[ReasoningChain] = None

class UnifiedMemoryAgent:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        faiss_index_path: str = "./faiss_multi_index",
        mongo_uri: str = "mongodb://localhost:27017/",
        mongo_db: str = "agent_db",
        knowledge_collection: str = "knowledge",
        workflow_collection: str = "workflows"
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        self.faiss_index_path = faiss_index_path
        self.faiss_index, self.faiss_map = self._load_or_create_faiss_index()

        self.mongo_client = MongoClient(mongo_uri)
        self.mongo_db = self.mongo_client[mongo_db]
        self.knowledge_col = self.mongo_db[knowledge_collection]
        self.workflow_col = self.mongo_db[workflow_collection]

    def _get_embedding(self, entry: Union[KnowledgeEntry, WorkflowMemory]) -> np.ndarray:
        if entry.content and entry.content.text:
            return self.embedding_model.encode([entry.content.text], convert_to_numpy=True)
        else:
            # Fallback: use query string
            if isinstance(entry, KnowledgeEntry):
                return self.embedding_model.encode([entry.query], convert_to_numpy=True)
            else:
                return self.embedding_model.encode([entry.original_query], convert_to_numpy=True)

    def _load_or_create_faiss_index(self) -> (faiss.IndexFlatL2, Dict[int, Dict[str, Any]]):
        index_file = f"{self.faiss_index_path}.index"
        map_file = f"{self.faiss_index_path}_map.pkl"
        if os.path.exists(index_file):
            index = faiss.read_index(index_file)
            with open(map_file, 'rb') as f:
                map_dict = pickle.load(f)
        else:
            index = faiss.IndexFlatL2(self.embedding_dim)
            map_dict = {}
        return index, map_dict

    def _save_faiss_index(self):
        index_file = f"{self.faiss_index_path}.index"
        map_file = f"{self.faiss_index_path}_map.pkl"
        faiss.write_index(self.faiss_index, index_file)
        with open(map_file, 'wb') as f:
            pickle.dump(self.faiss_map, f)

    def save_knowledge(self, entry: KnowledgeEntry) -> str:
        doc = entry.dict()
        self.knowledge_col.insert_one(doc)
        vec = self._get_embedding(entry)
        faiss_index = self.faiss_index.ntotal
        self.faiss_index.add(vec.astype('float32'))
        self.faiss_map[faiss_index] = {
            "mongo_id": entry._id,
            "type": "knowledge",
            "category": entry.category,
            "tags": entry.tags,
            "user_id": entry.user_id
        }
        self._save_faiss_index()
        return entry._id

    def save_workflow(self, workflow: WorkflowMemory) -> str:
        doc = workflow.dict()
        self.workflow_col.insert_one(doc)
        vec = self._get_embedding(workflow)
        faiss_index = self.faiss_index.ntotal
        self.faiss_index.add(vec.astype('float32'))
        self.faiss_map[faiss_index] = {
            "mongo_id": workflow._id,
            "type": "workflow",
            "category": workflow.category,
            "tags": workflow.tags,
            "user_id": workflow.user_id
        }
        self._save_faiss_index()
        return workflow._id

    def retrieve_similar(
            self,
            new_query: str,
            category: Optional[str] = None,
            tags: Optional[List[str]] = None,
            user_id: Optional[str] = None,
            top_k: int = 3,
            similarity_threshold: float = 0.6,
            keyword: Optional[str] = None,
            years_filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        vec = self.embedding_model.encode([new_query], convert_to_numpy=True)
        if self.faiss_index.ntotal == 0:
            return []
        distances, faiss_indices = self.faiss_index.search(vec.astype('float32'), k=top_k)
        results = []
        for dist, idx in zip(distances[0], faiss_indices[0]):
            info = self.faiss_map.get(idx)
            if not info or dist > similarity_threshold:
                continue
            # Hybrid: keyword/tag/user/category/time filtering
            doc = None
            if info["type"] == "workflow":
                doc_data = self.workflow_col.find_one({"_id": info["mongo_id"]})
                doc_obj = WorkflowMemory(**doc_data) if doc_data else None
            else:
                doc_data = self.knowledge_col.find_one({"_id": info["mongo_id"]})
                doc_obj = KnowledgeEntry(**doc_data) if doc_data else None
            if doc_obj:
                # Filter by category/tags/user/keyword/year
                if category and getattr(doc_obj, "category", None) != category:
                    continue
                if tags and not set(tags).intersection(set(getattr(doc_obj, "tags", []))):
                    continue
                if user_id and getattr(doc_obj, "user_id", None) != user_id:
                    continue
                if keyword and keyword not in (getattr(doc_obj, "query", "") + getattr(doc_obj, "solution", "")):
                    continue
                if years_filter:
                    year = int(str(doc_obj.metadata.get("timestamp", "")).split("-")[0])
                    from_year = datetime.now().year - years_filter
                    if year < from_year:
                        continue
                results.append({"type": info["type"], "dist": dist, "entry": doc_obj})
        return results

    def update_rating(self, entry_id: str, is_workflow: bool, delta: int):
        col = self.workflow_col if is_workflow else self.knowledge_col
        doc = col.find_one({"_id": entry_id})
        if doc:
            new_rating = doc.get("rating", 0) + delta
            col.update_one({"_id": entry_id}, {"$set": {"rating": new_rating}})
            return new_rating
        return None

    def solve_with_memory(self, bug_or_query: str,
                         category: Optional[str] = None,
                         tags: Optional[List[str]] = None,
                         user_id: Optional[str] = None,
                         keyword: Optional[str] = None,
                         years_filter: Optional[int] = None) -> Dict[str, Any]:
        candidates = self.retrieve_similar(
            bug_or_query, category, tags, user_id,
            similarity_threshold=0.75, keyword=keyword,
            years_filter=years_filter)
        solution = ""
        context_used = []
        for candidate in sorted(candidates, key=lambda x: getattr(x["entry"], "rating", 0), reverse=True):
            context_used.append({
                "type": candidate["type"],
                "category": candidate["entry"].category,
                "tags": candidate["entry"].tags,
                "rating": candidate["entry"].rating,
                "user_id": candidate["entry"].user_id,
                "match_query": getattr(candidate["entry"], "query", getattr(candidate["entry"], "original_query", "")),
                "dist": candidate["dist"]
            })
            if candidate["type"] == "knowledge":
                solution = candidate["entry"].solution
                break
            elif candidate["type"] == "workflow" and not solution:
                solution = f"Workflow steps: {candidate['entry'].workflow_plan}\nFinal answer: {candidate['entry'].final_answer}"
                if candidate['entry'].reasoning:
                    solution += f"\nDecision Chain: {candidate['entry'].reasoning.chain}"
        if not solution:
            solution = "No relevant solution found. Try refining your query or add new knowledge."
        # Save Q&A for future recall
        new_kn = KnowledgeEntry(query=bug_or_query, solution=solution, category=category or "uncategorized", tags=tags or [], user_id=user_id)
        self.save_knowledge(new_kn)
        return {
            "user_query": bug_or_query,
            "solution": solution,
            "category": category,
            "contexts_used": context_used
        }

    def get_stats(self) -> Dict:
        return {
            "faiss_index_size": self.faiss_index.ntotal,
            "knowledge_entries": self.knowledge_col.count_documents({}),
            "workflow_entries": self.workflow_col.count_documents({}),
            "categories": list(set([m.get("category") for m in self.faiss_map.values() if "category" in m]))
        }

# -- Example Usage --
if __name__ == "__main__":
    agent = UnifiedMemoryAgent()

    # Save multi-modal knowledge (image, PDF transcript, tags, reasoning)
    agent.save_knowledge(KnowledgeEntry(
        query="Explain the Mona Lisa painting.",
        solution="The Mona Lisa is painted by Leonardo da Vinci in the 16th century.",
        category="art",
        tags=["painting", "renaissance", "da vinci"],
        metadata={"timestamp": "2023-07-01"},
        content=MultiModalContent(
            text="The Mona Lisa is a famous portrait...",
            image_path="/images/monalisa.jpg"
        ),
        reasoning=ReasoningChain(chain=[
            ReasonChainStep(step="Identify painting", evidence="Visual analysis"),
            ReasonChainStep(step="Research artist", evidence="Historical records")
        ])
    ))

    # Save workflow with user and tags
    agent.save_workflow(WorkflowMemory(
        original_query="Legal case: property dispute procedure",
        workflow_plan=[
            "Review all document evidence",
            "Consult legal precedents (past 3 years)",
            "Negotiate mediation",
            "File formal complaint if unresolved"
        ],
        final_answer="Case resolved with mediation.",
        category="legal",
        tags=["dispute", "property", "mediation"],
        metadata={"timestamp": "2022-05-05"},
        user_id="lawyer42",
        reasoning=ReasoningChain(chain=[
            ReasonChainStep(step="Check mediation history"),
            ReasonChainStep(step="Apply legal precedent")
        ])
    ))

    # Search with hybrid tags/user/year filtering and give feedback
    matches = agent.solve_with_memory(
        "What's the procedure for property dispute mediation?",
        category="legal",
        tags=["mediation"],
        years_filter=3,
        user_id="lawyer42"
    )
    print("Legal retrieval by tags, user and year:", matches["solution"])
    print("Context Used:", matches["contexts_used"])

    # Upvote a workflow solution
    if matches["contexts_used"]:
        first_id = matches["contexts_used"][0]["match_query"]
        agent.update_rating(first_id, is_workflow=True, delta=1)
        print("Upvoted workflow rating.")

    print("\n--- Stats ---")
    print(agent.get_stats())
