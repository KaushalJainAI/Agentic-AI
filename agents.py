from dotenv import load_dotenv
from typing import Annotated, Literal, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, create_model
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

from tavily import TavilyClient

# --- State Definition (remains the same) ---

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
    schema: Dict[str, SchemaDefinition] = Field(..., description="The structured output schema, where each key is a field name and the value defines its type and description.")

class GraphState(TypedDict):
    """Represents the state of our graph."""
    prompt: str
    search_query: str
    # Note: We still use Dict[str, Any] here for the state itself, as the validation happens in the node.
    schema: Dict[str, Any] 
    urls: List[str]
    scraped_content: str
    structured_output: Dict[str, Any]

# --- The Agent Object ---
class WebScrapingAgent:
    """
    An agent that dynamically scrapes web content based on a user prompt.
    """
    def __init__(
        self, 
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None
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

        # Compile the graph and store it as an instance variable
        self.app = self._build_graph()
        self.tavily_client = TavilyClient(api_key=tavily_api_key)

    
    def _build_graph(self):
        """Builds and compiles the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        workflow.add_node("plan_node", self.plan_node)
        workflow.add_node("search_node", self.search_node)
        workflow.add_node("scrape_node", self.scrape_node)
        workflow.add_node("extract_node", self.extract_node)
        workflow.set_entry_point("plan_node")
        workflow.add_edge("plan_node", "search_node")
        workflow.add_edge("search_node", "scrape_node")
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
        """Node to generate a search query and a Pydantic schema."""
        print("--- PLAN ---")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
            """You are an expert research planner. Your task is to create a focused search query and a detailed schema to answer the user's request.
            
            For the schema field, return a dictionary where:
            - Each key is a field name (string)
            - Each value is an object with exactly two properties:
            * "type": a string representing the Python type (e.g., "str", "int", "List[str]", "Optional[str]")
            * "description": a string describing what this field represents
            
            Keep the schema focused with 3-6 relevant fields maximum."""
            ),
            ("user", "Research request: {prompt}")
        ])
        
        try:
            planner = prompt_template | self.llm.with_structured_output(Plan)
            plan_result = planner.invoke({"prompt": state['prompt']})
            
            # Convert SchemaDefinition objects to simple dictionaries
            schema_as_dict = {}
            for key, schema_def in plan_result.schema.items():
                schema_as_dict[key] = {
                    "type": schema_def.type,
                    "description": schema_def.description
                }
            
            return {
                "search_query": plan_result.search_query, 
                "schema": schema_as_dict
            }
        
        except Exception as e:
            print(f"Error in plan_node: {e}")
            # Fallback schema
            return {
                "search_query": state['prompt'],
                "schema": {
                    "main_content": {
                        "type": "str", 
                        "description": "Main content extracted from the research"
                    },
                    "summary": {
                        "type": "str", 
                        "description": "Brief summary of the findings"
                    }
                }
            }


    def search_node(self, state: GraphState) -> Dict[str, Any]:
        """Node to perform a web search."""
        print("--- SEARCH ---")
        search_results = self.tavily_client.search(query=state['search_query'], max_results=5)
        urls = [result['url'] for result in search_results['results']]
        return {"urls": urls}

    def scrape_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Node to scrape content from all provided URLs and combine them.
        """
        if state.get("error_message") or not state.get("urls"):
            print("Skipping scrape due to previous failure or no URLs.")
            return {"scraped_content": "", "error_message": "No URLs to scrape."}

        print("--- SCRAPE (Cumulative) ---")
        urls_to_try = state.get("urls", [])
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
        """Node to extract information based on the dynamic schema."""
        print("--- EXTRACT ---")
        
        try:
            # Create dynamic model with safe type conversion
            field_definitions = {}
            for key, val in state['schema'].items():
                field_type = self._safe_type_conversion(val['type'])
                field_definitions[key] = (field_type, Field(description=val['description']))
            
            DynamicModel = create_model('DynamicModel', **field_definitions)
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert data extractor. Extract the relevant information from the provided text that precisely answers the user's goal and format it according to the provided schema."),
                ("user", "User Goal: {prompt}\n\nWebpage Content:\n{content}")
            ])
            
            extractor = prompt_template | self.llm.with_structured_output(DynamicModel)
            extracted_data = extractor.invoke({
                "prompt": state['prompt'], 
                "content": state['scraped_content']
            })
            
            return {"structured_output": extracted_data.dict()}
        
        except Exception as e:
            print(f"Error in extract_node: {e}")
            return {"structured_output": {"error": f"Extraction failed: {str(e)}"}}


    def run(self, prompt: str):
        """The main entry point to run the agent."""
        inputs = {"prompt": prompt}
        final_state = self.app.invoke(inputs)
        return final_state['structured_output']

    def run_and_stream_watch(self, prompt: str):
        """
        The main entry point to run the agent.
        This method now streams the output of each node to the console.
        """
        inputs = {"prompt": prompt}
        final_state = {}

        # Use app.stream() to see the output of each node
        print("--- 🚀 Starting Agent Run ---")
        for output in self.app.stream(inputs, {"recursion_limit": 10}):
            for key, value in output.items():
                print(f"\n--- ✅ Output from node: {key} ---")
                # Pretty print the dictionary to see the state at each step
                print(json.dumps(value, indent=2, ensure_ascii=False))
                final_state = value # The last value holds the final state

        print("\n--- 🏁 Agent Finished ---")
        return final_state.get('structured_output', {"error": "Agent failed to produce a structured output."})
    

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseDefinition(BaseModel):
    """Definition of a database file with metadata."""
    file_path: str = Field(description="Path to the database file")
    description: str = Field(description="Description of what data this database contains")
    relevance_score: float = Field(description="How relevant this database is to the query (0-1)")

class DatabaseSelection(BaseModel):
    """Selection of the most relevant database."""
    selected_database: str = Field(description="Path to the selected database file")
    reasoning: str = Field(description="Why this database was selected")

class QueryPlan(BaseModel):
    """Plan for querying the database."""
    sql_query: str = Field(description="SQL query to execute")
    expected_columns: List[str] = Field(description="Expected column names in the result")
    query_explanation: str = Field(description="Explanation of what the query does")

class SQLGraphState(TypedDict):
    """Represents the state of our SQL agent graph."""
    prompt: str
    available_databases: List[Dict[str, Any]]
    selected_database: str
    database_schema: Dict[str, Any]
    sql_query: str
    query_results: List[Dict[str, Any]]
    structured_output: Dict[str, Any]
    error_message: Optional[str]

class SQL_CRUD_agent:
    """
    An agent that searches for relevant database files and queries them for information.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        database_directory: str = "./Databases",
        database_descriptions: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the SQL CRUD agent.
        
        Args:
            model: Model name to use
            model_provider: Provider for the model
            temperature: Temperature setting for response generation
            api_key: API key (if None, will try to get from environment)
            database_directory: Directory containing database files
            database_descriptions: Dictionary mapping database file names to descriptions
        """
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.database_directory = database_directory
        self.database_descriptions = database_descriptions or {}
        
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set LLM_API_KEY in environment variable.")
        
        # Initialize the LLM
        try:
            self.llm = init_chat_model(
                model=self.model,
                model_provider=self.model_provider,
                temperature=self.temperature,
                api_key=self.api_key
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")

        # Compile the graph and store it as an instance variable
        self.app = self._build_graph()

    def _build_graph(self):
        """Builds and compiles the LangGraph workflow."""
        workflow = StateGraph(SQLGraphState)
        
        # Add nodes
        workflow.add_node("discover_databases", self.discover_databases_node)
        workflow.add_node("select_database", self.select_database_node)
        workflow.add_node("analyze_schema", self.analyze_schema_node)
        workflow.add_node("generate_query", self.generate_query_node)
        workflow.add_node("execute_query", self.execute_query_node)
        workflow.add_node("format_results", self.format_results_node)
        
        # Define the workflow
        workflow.set_entry_point("discover_databases")
        workflow.add_edge("discover_databases", "select_database")
        workflow.add_edge("select_database", "analyze_schema")
        workflow.add_edge("analyze_schema", "generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", "format_results")
        workflow.add_edge("format_results", END)
        
        return workflow.compile()

    def _get_database_files(self) -> List[str]:
        """
        Get all database files by searching recursively through the specified directory 
        and its subdirectories.
        """
        db_extensions = ['*.db', '*.sqlite', '*.sqlite3']
        db_files = []
        
        for extension in db_extensions:
            # The '**' component in the pattern combined with `recursive=True`
            # enables searching in all subdirectories.
            pattern = os.path.join(self.database_directory, '**', extension)
            db_files.extend(glob.glob(pattern, recursive=True))
        
        return db_files

    def _get_database_schema(self, db_path: str) -> Dict[str, Any]:
        """Get the schema of a database."""
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

    def discover_databases_node(self, state: SQLGraphState) -> Dict[str, Any]:
        """
        Node to discover available database files and their descriptions from local text files.
        """
        # logger.info("--- DISCOVER DATABASES AND DESCRIPTIONS ---")
        
        try:
            db_files = self._get_database_files()
            available_databases = []

            for db_file in db_files:
                db_name = os.path.basename(db_file)
                db_directory = os.path.dirname(db_file)
                description = "No description file found."  # Default description

                # Search for any .txt file in the database's directory
                description_files = glob.glob(os.path.join(db_directory, '*.txt'))
                
                if description_files:
                    # If a .txt file is found, try to read it
                    try:
                        with open(description_files[0], 'r', encoding='utf-8') as f:
                            description = f.read().strip()
                    except Exception as e:
                        description = f"Error reading description file: {e}"

                available_databases.append({
                    "file_path": db_file,
                    "name": db_name,
                    "description": description
                })
            
            if not available_databases:
                return {"error_message": "No database files found in the specified directory"}
            
            return {"available_databases": available_databases}
            
        except Exception as e:
            # logger.error(f"Error in discover_databases_node: {e}")
            return {"error_message": f"Failed to discover databases: {str(e)}"}

    def select_database_node(self, state: SQLGraphState) -> Dict[str, Any]:
        """Node to select the most relevant database."""
        # logger.info("--- SELECT DATABASE ---")
        
        if state.get("error_message"):
            return {}
        
        try:
            available_dbs = state["available_databases"]
            
            # --- FIX STARTS HERE ---
            # Create a "safe" list of descriptions by escaping curly braces
            safe_db_lines = []
            for db in available_dbs:
                # Escape braces in the name and description to prevent format errors
                safe_name = db['name'].replace('{', '{{').replace('}', '}}')
                safe_desc = db['description'].replace('{', '{{').replace('}', '}}')
                safe_db_lines.append(f"- {safe_name}: {safe_desc}")
            
            db_list = "\n".join(safe_db_lines)
            # --- FIX ENDS HERE ---

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", 
                f"""You are a database selection expert. Given a user query and a list of available databases, select the most relevant database filename.
                
                Available databases:
                {db_list}
                
                Return only the filename of the database that is most likely to contain information relevant to the user's query."""),
                ("user", "Query: {prompt}")
            ])
            
            selector = prompt_template | self.llm.with_structured_output(DatabaseSelection)
            selection = selector.invoke({"prompt": state["prompt"]})
            
            # Find the full path for the selected database with more robust matching
            selected_db_path = None
            for db in available_dbs:
                # Match the exact filename returned by the LLM
                if db["name"] == selection.selected_database:
                    selected_db_path = db["file_path"]
                    break
            
            if not selected_db_path:
                return {"error_message": f"Could not find the selected database file: {selection.selected_database}"}
            
            return {"selected_database": selected_db_path}
            
        except Exception as e:
            # logger.error(f"Error in select_database_node: {e}")
            return {"error_message": f"Failed to select database: {str(e)}"}

    def analyze_schema_node(self, state: SQLGraphState) -> Dict[str, Any]:
        """Node to analyze the schema of the selected database."""
        logger.info("--- ANALYZE SCHEMA ---")
        
        if state.get("error_message"):
            return {}
        
        try:
            db_path = state["selected_database"]
            schema = self._get_database_schema(db_path)
            
            if not schema:
                return {"error_message": "Could not analyze database schema"}
            
            return {"database_schema": schema}
            
        except Exception as e:
            logger.error(f"Error in analyze_schema_node: {e}")
            return {"error_message": f"Failed to analyze schema: {str(e)}"}

    def generate_query_node(self, state: SQLGraphState) -> Dict[str, Any]:
        """Node to generate SQL query based on the prompt and schema."""
        logger.info("--- GENERATE QUERY ---")
        
        if state.get("error_message"):
            return {}
        
        try:
            schema = state["database_schema"]
            
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
                ("user", "Query request: {prompt}")
            ])
            
            query_generator = prompt_template | self.llm.with_structured_output(QueryPlan)
            query_plan = query_generator.invoke({"prompt": state["prompt"]})
            
            return {"sql_query": query_plan.sql_query}
            
        except Exception as e:
            logger.error(f"Error in generate_query_node: {e}")
            return {"error_message": f"Failed to generate query: {str(e)}"}

    def execute_query_node(self, state: SQLGraphState) -> Dict[str, Any]:
        """Node to execute the SQL query."""
        logger.info("--- EXECUTE QUERY ---")
        
        if state.get("error_message"):
            return {}
        
        try:
            db_path = state["selected_database"]
            sql_query = state["sql_query"]
            
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # This allows accessing columns by name
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            
            # Convert rows to list of dictionaries
            results = []
            for row in rows:
                results.append(dict(row))
            
            conn.close()
            
            return {"query_results": results}
            
        except Exception as e:
            logger.error(f"Error in execute_query_node: {e}")
            return {"error_message": f"Failed to execute query: {str(e)}"}

    def format_results_node(self, state: SQLGraphState) -> Dict[str, Any]:
        """Node to format the query results."""
        logger.info("--- FORMAT RESULTS ---")
        
        if state.get("error_message"):
            return {"structured_output": {"error": state["error_message"]}}
        
        try:
            results = state["query_results"]
            
            if not results:
                return {"structured_output": {"message": "No results found"}}
            
            # Format results based on the original prompt
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", 
                 """You are a data formatter. Given query results and the original user request, format the data in a meaningful way.
                 
                 Return a structured summary that directly answers the user's question based on the query results."""),
                ("user", "Original request: {prompt}\n\nQuery results: {results}")
            ])
            
            # Create a simple output model
            class FormattedOutput(BaseModel):
                summary: str = Field(description="Summary of the findings")
                total_records: int = Field(description="Number of records found")
                key_findings: List[str] = Field(description="Key insights from the data")
                raw_data: List[Dict[str, Any]] = Field(description="Raw query results")
            
            formatter = prompt_template | self.llm.with_structured_output(FormattedOutput)
            formatted_output = formatter.invoke({
                "prompt": state["prompt"],
                "results": json.dumps(results, indent=2)
            })
            
            return {"structured_output": formatted_output.dict()}
            
        except Exception as e:
            logger.error(f"Error in format_results_node: {e}")
            return {"structured_output": {"error": f"Failed to format results: {str(e)}"}}

    def run(self, prompt: str) -> Dict[str, Any]:
        """The main entry point to run the SQL CRUD agent."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            inputs = {"prompt": prompt.strip()}
            final_state = self.app.invoke(inputs)
            return final_state.get('structured_output', {"error": "No output generated"})
        except Exception as e:
            logger.error(f"Error running SQL agent: {e}")
            return {"error": f"Agent execution failed: {str(e)}"}

    def run_and_stream_watch(self, prompt: str) -> Dict[str, Any]:
        """
        Run the agent with streaming output to see each step.
        """
        inputs = {"prompt": prompt}
        final_state = {}

        print("--- 🗄️ Starting SQL CRUD Agent Run ---")
        for output in self.app.stream(inputs, {"recursion_limit": 10}):
            for key, value in output.items():
                print(f"\n--- ✅ Output from node: {key} ---")
                print(json.dumps(value, indent=2, ensure_ascii=False))
                final_state = value

        print("\n--- 🏁 SQL Agent Finished ---")
        return final_state.get('structured_output', {"error": "Agent failed to produce a structured output."})

class PandasOperation(BaseModel):
    """A single data manipulation step using pandas."""
    operation: str = Field(description="The pandas operation to perform, e.g., 'load', 'merge', 'filter', 'groupby', 'select'.")
    args: Dict[str, Any] = Field(description="Arguments for the operation. E.g., for 'load', {'filename': 'data.csv'}. For 'merge', {'left': 'df1', 'right': 'df2', 'on': 'id_column'}.")
    result_df: str = Field(description="The name to assign to the resulting DataFrame.")

class ExecutionPlan(BaseModel):
    """A step-by-step plan of pandas operations to answer a query."""
    reasoning: str = Field(description="A brief explanation of the plan's logic.")
    steps: List[PandasOperation] = Field(description="The sequence of pandas operations to execute.")

# --- NEW: Updated Graph State for the new workflow ---

class FileQueryState(TypedDict):
    """Represents the state of our file query agent graph."""
    prompt: str
    file_schema_summary: str
    execution_plan: Optional[ExecutionPlan]
    # This will hold the final DataFrame before formatting
    final_dataframe: Optional[pd.DataFrame] 
    structured_output: Dict[str, Any]
    error_message: Optional[str]

# --- NEW: The Refactored Agent Class ---

class FlatFileQueryAgent:
    """
    An agent that queries a collection of interrelated flat files (CSV, TSV, Excel).
    """
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        database_directory: str = "./Databases",
        database_descriptions: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Flat File Query Agent.
        
        Args:
            model: Model name to use
            model_provider: Provider for the model
            temperature: Temperature setting for response generation
            api_key: API key (if None, will try to get from environment)
            database_directory: Directory containing database files
            database_descriptions: Dictionary mapping database file names to descriptions
        """
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.database_directory = database_directory
        self.database_descriptions = database_descriptions or {}
        
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set LLM_API_KEY in environment variable.")
        
        # Initialize the LLM
        try:
            self.llm = init_chat_model(
                model=self.model,
                model_provider=self.model_provider,
                temperature=self.temperature,
                api_key=self.api_key
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")
        
        
        self.app = self._build_graph()

    def _build_graph(self):
        """Builds the LangGraph workflow for file querying."""
        workflow = StateGraph(FileQueryState)
        
        # New nodes for the flat file workflow
        workflow.add_node("analyze_files_node", self.analyze_files_node)
        workflow.add_node("generate_plan_node", self.generate_plan_node)
        workflow.add_node("execute_plan_node", self.execute_plan_node)
        workflow.add_node("format_results_node", self.format_results_node)
        
        workflow.set_entry_point("analyze_files_node")
        workflow.add_edge("analyze_files_node", "generate_plan_node")
        workflow.add_edge("generate_plan_node", "execute_plan_node")
        workflow.add_edge("execute_plan_node", "format_results_node")
        workflow.add_edge("format_results_node", END)
        
        return workflow.compile()

    
    def analyze_files_node(self, state: FileQueryState) -> Dict[str, Any]:
        """
        Discovers files, analyzes their full schema (including data types) using df.info(),
        and infers potential relationships based on common column names.
        """
        # logger.info("--- ANALYZING FILE SCHEMAS & RELATIONSHIPS ---")
        try:
            file_extensions = ['*.csv', '*.tsv', '*.xlsx', '*.json']
            all_files = []
            for ext in file_extensions:
                all_files.extend(glob.glob(os.path.join(self.file_directory, '**', ext), recursive=True))

            if not all_files:
                return {"error_message": "No data files (CSV, TSV, Excel, JSON) found."}

            schema_details = ["## Available Tables (Files) and their Schemas:"]
            # This dictionary will hold {filename: [col1, col2]} for relationship analysis
            file_columns = {} 

            # 1. First Pass: Get the detailed schema for each file
            for file_path in all_files:
                filename = os.path.basename(file_path)
                try:
                    # Read the full file to get accurate type info
                    if filename.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                    elif filename.endswith('.json'):
                        df = pd.read_json(file_path)
                    else:
                        sep = '\t' if filename.endswith('.tsv') else ','
                        df = pd.read_csv(file_path, sep=sep)
                    
                    # Capture the output of df.info() as a string
                    with io.StringIO() as buffer, redirect_stdout(buffer):
                        df.info()
                        info_string = buffer.getvalue()
                    
                    schema_details.append(f"\n### File: `{filename}`")
                    schema_details.append(f"```\n{info_string}\n```")

                    # Store columns for the next step
                    file_columns[filename] = df.columns.tolist()

                except Exception as e:
                    schema_details.append(f"\n- Could not analyze {filename}. Error: {e}")

            # 2. Second Pass: Analyze columns to infer relationships
            column_to_files = defaultdict(list)
            for filename, columns in file_columns.items():
                for column in columns:
                    # Group files by the columns they contain
                    column_to_files[column].append(filename)
            
            relationship_summary = ["\n## Inferred Relationships (Potential Join Keys):"]
            found_relationships = False
            for column, files in column_to_files.items():
                # If a column name appears in more than one file, it's a potential key
                if len(files) > 1:
                    relationship_summary.append(f"- **Column `{column}`** is a potential key, found in: `{', '.join(files)}`")
                    found_relationships = True
            
            if not found_relationships:
                relationship_summary.append("- No obvious relationships were found based on common column names.")

            # 3. Combine all details into the final summary string
            final_summary = "\n".join(schema_details + relationship_summary)
            
            return {"file_schema_summary": final_summary}
            
        except Exception as e:
            return {"error_message": f"Failed during file analysis: {e}"}

    def generate_plan_node(self, state: FileQueryState) -> Dict[str, Any]:
        """Generates a step-by-step pandas execution plan."""
        # logger.info("--- GENERATING EXECUTION PLAN ---")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             """You are a data analysis expert. Your task is to create a step-by-step execution plan using pandas operations to answer a user's question based on the available files.
             
             Available Operations:
             - `load`: Loads a file into a DataFrame. args: {"filename": "file.csv"}
             - `merge`: Joins two DataFrames. args: {"left": "df1_name", "right": "df2_name", "on": "common_column", "how": "inner/left/right"}
             - `filter`: Filters rows in a DataFrame. args: {"df_name": "df_to_filter", "query_string": "column_name > 100"}
             - `groupby`: Groups data and performs an aggregation. args: {"df_name": "df_to_group", "by": ["col1", "col2"], "agg": {"col_to_agg": "sum/mean/count"}}
             - `select`: Selects specific columns. args: {"df_name": "df_to_select", "columns": ["col1", "col2"]}
             
             Analyze the user's prompt and the available file schemas to create the plan. Infer the join keys based on column names.
             
             File Schemas:
             {file_schema}"""),
            ("user", "Question: {prompt}")
        ])
        
        planner = prompt_template | self.llm.with_structured_output(ExecutionPlan)
        plan = planner.invoke({
            "prompt": state['prompt'],
            "file_schema": state['file_schema_summary']
        })
        return {"execution_plan": plan}

    def execute_plan_node(self, state: FileQueryState) -> Dict[str, Any]:
        """Executes the pandas plan, now with JSON loading capability."""
        # logger.info("--- EXECUTING PLAN ---")
        plan = state.get("execution_plan")
        if not plan:
            return {"error_message": "No execution plan found."}
            
        dataframes = {} # To store in-memory dataframes
        final_df_name = ""

        try:
            for step in plan.steps:
                op = step.operation.lower()
                args = step.args
                final_df_name = step.result_df

                if op == 'load':
                    file_path = os.path.join(self.file_directory, args['filename'])
                    if file_path.endswith('.xlsx'):
                        dataframes[step.result_df] = pd.read_excel(file_path)
                    # NEW: Add a condition to load JSON files into a DataFrame
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
                else:
                    return {"error_message": f"Unknown operation: {op}"}

            final_df = dataframes.get(final_df_name)
            return {"final_dataframe": final_df}

        except Exception as e:
            return {"error_message": f"Failed during plan execution: {e}"}

    def format_results_node(self, state: FileQueryState) -> Dict[str, Any]:
        """Formats the final DataFrame into a structured output."""
        # logger.info("--- FORMATTING RESULTS ---")
        if state.get("error_message"):
            return {"structured_output": {"error": state["error_message"]}}
        
        final_df = state.get("final_dataframe")
        if final_df is None or final_df.empty:
            return {"structured_output": {"message": "The query resulted in no data."}}

        # Convert DataFrame to a list of dictionaries for JSON compatibility
        results_json = final_df.head(20).to_json(orient="records")

        return {"structured_output": json.loads(results_json)}

    def run(self, prompt: str) -> Dict[str, Any]:
        """The main entry point to run the agent."""
        inputs = {"prompt": prompt}
        final_state = self.app.invoke(inputs)

    def run_and_stream_watch(self, prompt: str):
        """
        The main entry point to run the agent.
        This method now streams the output of each node to the console.
        """
        inputs = {"prompt": prompt}
        final_state = {}

        # Use app.stream() to see the output of each node
        print("--- 🚀 Starting Agent Run ---")
        for output in self.app.stream(inputs, {"recursion_limit": 10}):
            for key, value in output.items():
                print(f"\n--- ✅ Output from node: {key} ---")
                # Pretty print the dictionary to see the state at each step
                print(json.dumps(value, indent=2, ensure_ascii=False))
                final_state = value # The last value holds the final state

        print("\n--- 🏁 Agent Finished ---")
        return final_state.get('structured_output', {"error": "Agent failed to produce a structured output."})

