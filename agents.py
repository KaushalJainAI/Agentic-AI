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
from langchain_community.chat_models import ChatOllama 


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
        self.llm = init_chat_model(
            model=self.model,
            model_provider=self.model_provider,
            temperature=self.temperature,
            api_key=self.api_key
        )
        self.graph = self._build_graph()

        return f"Temperature updated successfully to {temperature}."


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
    Planschema: Dict[str, SchemaDefinition] = Field(..., description="The structured output schema, where each key is a field name and the value defines its type and description.")

class GraphState(TypedDict):
    """Represents the state of our graph."""
    prompt: str
    search_query: str
    # Note: We still use Dict[str, Any] here for the state itself, as the validation happens in the node.
    PlanSchema: Dict[str, Any] 
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
        tavily_api_key: Optional[str] = None,
        max_website_count: int = 10
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
        self.max_website_count = max_website_count
        
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
        """Node to generate a search query and a Pydantic PlanSchema."""
        print("--- PLAN ---")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
            """You are an expert scrapping agent. Your task is to create a focused search query and a detailed PlanSchema to answer the user's request.
            
            For the PlanSchema field, return a dictionary where:
            - Each key is a field name (string)
            - Each value is an object with exactly two properties:
            * "type": a string representing the Python type (e.g., "str", "int", "List[str]", "Optional[str]")
            * "description": a string describing what this field represents
            
            Keep the PlanSchema focused with 3-6 relevant fields maximum."""
            ),
            ("user", "Research request: {prompt}")
        ])
        
        try:
            planner = prompt_template | self.llm.with_structured_output(Plan)
            plan_result = planner.invoke({"prompt": state['prompt']})
            
            # Convert PlanSchemaDefinition objects to simple dictionaries
            schema_as_dict = {}
            for key, schema_def in plan_result.PlanSchema.items():
                schema_as_dict[key] = {
                    "type": schema_def.type,
                    "description": schema_def.description
                }
            
            return {
                "search_query": plan_result.search_query, 
                "PlanSchema": schema_as_dict
            }
        
        except Exception as e:
            print(f"Error in plan_node: {e}")
            # Fallback schema
            return {
                "search_query": state['prompt'],
                "PlanSchema": {
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
        search_results = self.tavily_client.search(query=state['search_query'], max_results=self.max_website_count)
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
        """Node to extract information based on the dynamic PlanSchema."""
        print("--- EXTRACT ---")
        
        try:
            # Create dynamic model with safe type conversion
            field_definitions = {}
            for key, val in state['PlanSchema'].items():
                field_type = self._safe_type_conversion(val['type'])
                field_definitions[key] = (field_type, Field(description=val['description']))
            
            DynamicModel = create_model('DynamicModel', **field_definitions)
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert data extractor. Extract the relevant information from the provided text that precisely answers the user's goal and format it according to the provided PlanSchema."),
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
        database_directory: str = "./Databases"
    ):
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.database_directory = database_directory
        
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set LLM_API_KEY in environment variable.")
        
        try:
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
        api_key: Optional[str] = None
    ):
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key is required.")
        
        try:
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
        api_key: Optional[str] = None
    ):
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key is required.")
        
        try:
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
        database_directory: str = "./Databases"
    ):
        # Initialize all three agents
        self.discovery_agent = DatabaseDiscoveryAgent(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            api_key=api_key,
            database_directory=database_directory
        )
        
        self.sql_agent = SQLQueryAgent(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            api_key=api_key
        )
        
        self.flatfile_agent = FlatFileQueryAgent(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            api_key=api_key
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


from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import faiss

import numpy as np
import pickle


class KnowledgeEntry(BaseModel):
    """Structure for knowledge entries with validation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    content: str = Field(..., min_length=1, description="Knowledge content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated metadata")
    embedding: Optional[np.ndarray] = Field(default=None, description="Vector embedding")
    
    def model_dump_for_storage(self) -> Dict[str, Any]:
        """Custom serialization excluding numpy arrays"""
        data = self.model_dump(exclude={'embedding'})
        return data

class QueryResult(BaseModel):
    """Structure for query results with validation"""
    query: str = Field(..., min_length=1, description="Original search query")
    similar_contexts: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved contexts")
    total_results: int = Field(ge=0, description="Number of results found")
    similarity_scores: List[float] = Field(
        default_factory=list, 
        description="Similarity scores for results"
    )
    retrieved_knowledge: str = Field(default="", description="Combined knowledge text")
    
    @field_validator('similarity_scores')
    @classmethod
    def validate_similarity_scores(cls, v, info):
        """Ensure similarity scores are between 0 and 1"""
        if any(score < 0 or score > 1 for score in v):
            raise ValueError("Similarity scores must be between 0 and 1")
        return v
    
    @field_validator('total_results')
    @classmethod
    def validate_total_results(cls, v, info):
        """Ensure total_results matches similar_contexts length"""
        similar_contexts = info.data.get('similar_contexts', [])
        if len(similar_contexts) != v:
            raise ValueError("total_results must match length of similar_contexts")
        return v

class KnowledgeStats(BaseModel):
    """Statistics about the knowledge base"""
    total_entries: int = Field(ge=0, description="Total knowledge entries")
    categories: Dict[str, int] = Field(default_factory=dict, description="Categories breakdown")
    avg_content_length: float = Field(ge=0, description="Average content length")
    index_size: int = Field(ge=0, description="Faiss index size")
    last_updated: Optional[str] = Field(default=None, description="Last update timestamp")

class ContextualPromptResult(BaseModel):
    """Result from contextual prompt generation"""
    original_query: str = Field(..., description="Original user query")
    task_type: str = Field(default="general", description="Type of task")
    optimized_prompt: str = Field(..., description="Generated optimized prompt")
    context_summary: str = Field(..., description="Summary of context used")
    task_instructions: List[str] = Field(default_factory=list, description="Task-specific instructions")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in context relevance")
    contexts_used: List[Dict[str, Any]] = Field(default_factory=list, description="Contexts that were used")
    knowledge_retrieved: str = Field(default="", description="Raw knowledge retrieved")
    
    @field_validator('confidence_score')
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence score is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
        return v



class VectorKnowledgeAgent:
    """
    Agent for managing and querying a Faiss-based vector knowledge database.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai", 
        embedding_model: str = "Qwen/Qwen3-Embedding-4B",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        index_path: Optional[str] = "./knowledge_base"
    ):
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.index_path = index_path
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            raise ValueError(f"Failed to initialize embedding model: {e}")
        
        # Initialize LLM
        if not self.api_key:
            raise ValueError("API key is required.")
            
        try:
            self.llm = init_chat_model(
                model=self.model,
                model_provider=self.model_provider,
                temperature=self.temperature,
                api_key=self.api_key
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")
        
        # Initialize Faiss index
        self.index = None
        self.knowledge_store = []
        self.metadata_store = []
        
        # Load existing index if available
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one"""
        try:
            if os.path.exists(f"{self.index_path}.index"):
                self.index = faiss.read_index(f"{self.index_path}.index")
                
                # Load metadata
                with open(f"{self.index_path}_metadata.pkl", 'rb') as f:
                    self.metadata_store = pickle.load(f)
                    
                # Load knowledge store
                with open(f"{self.index_path}_knowledge.pkl", 'rb') as f:
                    self.knowledge_store = pickle.load(f)
                    
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
                self.index = faiss.IndexIDMap(self.index)
                logger.info("Created new Faiss index")
                
        except Exception as e:
            logger.error(f"Error loading/creating index: {e}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIDMap(self.index)
    
    def add_knowledge(
        self, 
        content: Union[str, List[str]], 
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> Dict[str, Any]:
        """
        Add knowledge to the vector database
        """
        try:
            # Handle single string input
            if isinstance(content, str):
                content = [content]
                metadata = [metadata] if metadata else [{}]
            
            # Handle metadata
            if metadata is None:
                metadata = [{}] * len(content)
            elif isinstance(metadata, dict):
                metadata = [metadata] * len(content)
            
            # Text splitting for large documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            all_chunks = []
            all_metadata = []
            
            for i, doc_content in enumerate(content):
                chunks = text_splitter.split_text(doc_content)
                
                for j, chunk in enumerate(chunks):
                    chunk_metadata = metadata[i].copy()
                    chunk_metadata.update({
                        'doc_index': i,
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'char_count': len(chunk)
                    })
                    
                    all_chunks.append(chunk)
                    all_metadata.append(chunk_metadata)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(all_chunks, convert_to_numpy=True)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            start_id = len(self.knowledge_store)
            ids = np.array(range(start_id, start_id + len(all_chunks)))
            
            self.index.add_with_ids(embeddings.astype('float32'), ids)
            
            # Store knowledge and metadata
            for i, (chunk, meta) in enumerate(zip(all_chunks, all_metadata)):
                knowledge_entry = KnowledgeEntry(
                    content=chunk,
                    metadata=meta,
                    embedding=embeddings[i]
                )
                self.knowledge_store.append(knowledge_entry)
                self.metadata_store.append(meta)
            
            # Save index
            self.save_index()
            
            return {
                "status": "success",
                "chunks_added": len(all_chunks),
                "total_knowledge_entries": len(self.knowledge_store)
            }
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return {"status": "error", "message": str(e)}
    
    def query_knowledge(
        self, 
        query: str, 
        k: int = 5,
        min_similarity: float = 0.3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        Query the knowledge database for relevant context
        """
        try:
            if self.index.ntotal == 0:
                return QueryResult(
                    query=query,
                    similar_contexts=[],
                    total_results=0,
                    similarity_scores=[],
                    retrieved_knowledge="No knowledge available in the database."
                )
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search the index
            similarities, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Filter results
            similar_contexts = []
            filtered_similarities = []
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1 or similarity < min_similarity:
                    continue
                    
                knowledge_entry = self.knowledge_store[idx]
                
                # Apply metadata filter if provided
                if filter_metadata:
                    if not all(knowledge_entry.metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue
                
                context_info = {
                    'content': knowledge_entry.content,
                    'metadata': knowledge_entry.metadata,
                    'similarity_score': float(similarity),
                    'index': int(idx)
                }
                
                similar_contexts.append(context_info)
                filtered_similarities.append(float(similarity))
            
            # Combine retrieved knowledge
            retrieved_knowledge = "\n\n".join([ctx['content'] for ctx in similar_contexts])
            
            return QueryResult(
                query=query,
                similar_contexts=similar_contexts,
                total_results=len(similar_contexts),
                similarity_scores=filtered_similarities,
                retrieved_knowledge=retrieved_knowledge
            )
            
        except Exception as e:
            logger.error(f"Error querying knowledge: {e}")
            return QueryResult(
                query=query,
                similar_contexts=[],
                total_results=0,
                similarity_scores=[],
                retrieved_knowledge=f"Error querying knowledge: {str(e)}"
            )
    
    def generate_contextual_prompt(
        self,
        user_query: str,
        task_type: str = "general",
        context_limit: int = 3,
        category: Optional[str] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a contextual prompt using retrieved knowledge
        """
        # try:
        # Query for relevant context
        knowledge_result = self.query_knowledge(user_query, k=context_limit)
        
        # Build context section
        context_sections = []
        for i, ctx in enumerate(knowledge_result.similar_contexts, 1):
            context_section = f"Context {i} (Similarity: {ctx['similarity_score']:.3f}):\n{ctx['content']}"
            
            if include_metadata and ctx['metadata']:
                relevant_metadata = {k: str(v).replace('{', '{{').replace('}', '}}') 
                                for k, v in ctx['metadata'].items() 
                                if k not in ['doc_index', 'chunk_index']}
                if relevant_metadata:
                    metadata_str = json.dumps(relevant_metadata, indent=2)
                    context_section += f"\nMetadata: {metadata_str}"
            
            context_sections.append(context_section)
        
        context_text = "\n\n" + "="*50 + "\n\n".join(context_sections) if context_sections else "\n\nNo relevant context found in knowledge base."
        
        # Debug: Print context_text to inspect
        print("Context Text:", context_text)
        
        # Generate enhanced prompt using LLM
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
            """You are an expert prompt engineer. Given a user query, task type, and relevant context from a knowledge base, create an optimized prompt that:
            
            1. Incorporates the most relevant context
            2. Is tailored for the specific task type: {{task_type}}
            3. Provides clear instructions
            4. Maintains context relevance
            
            Available Context:
            {context_text}
            
            Task Type: {{task_type}}
            Original Query: {{user_query}}
            Category: {{category}}
            
            Generate a well-structured prompt that an AI agent can use effectively."""),
            ("user", "Generate an optimized prompt for this query and context.")
        ])
        
        # Use structured output for the prompt generation
        class OptimizedPrompt(BaseModel):
            enhanced_prompt: str = Field(description="The optimized prompt incorporating context")
            context_summary: str = Field(description="Summary of key context used")
            task_instructions: List[str] = Field(description="Specific instructions for the task")
            confidence_score: float = Field(description="Confidence in context relevance (0-1)")
        
        prompt_generator = prompt_template | self.llm.with_structured_output(OptimizedPrompt)
        optimized_prompt = prompt_generator.invoke({
            "user_query": user_query,
            "category":category if category else "uncategorized",
            "task_type": task_type,
            "context_text": context_text
        })
        
        return {
            "original_query": user_query,
            "task_type": task_type,
            "optimized_prompt": optimized_prompt.enhanced_prompt,
            "context_summary": optimized_prompt.context_summary,
            "task_instructions": optimized_prompt.task_instructions,
            "confidence_score": optimized_prompt.confidence_score,
            "contexts_used": knowledge_result.similar_contexts,
            "knowledge_retrieved": knowledge_result.retrieved_knowledge
        }
            
        # except Exception as e:
        #     logger.error(f"Error generating contextual prompt: {e}")
        #     return {
        #         "original_query": user_query,
        #         "task_type": task_type,
        #         "category": category if category else "uncategorized",
        #         "optimized_prompt": f"Error generating contextual prompt: {str(e)}",
        #         "context_summary": "No context available due to error",
        #         "task_instructions": [],
        #         "confidence_score": 0.0,
        #         "contexts_used": [],
        #         "knowledge_retrieved": ""
        #     }
            
    
    def save_index(self):
        """Save the current index and metadata to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else '.', exist_ok=True)
            
            faiss.write_index(self.index, f"{self.index_path}.index")
            
            with open(f"{self.index_path}_metadata.pkl", 'wb') as f:
                pickle.dump(self.metadata_store, f)
                
            with open(f"{self.index_path}_knowledge.pkl", 'wb') as f:
                pickle.dump(self.knowledge_store, f)
                
            logger.info("Index saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            total_entries = len(self.knowledge_store)
            if total_entries == 0:
                return {"total_entries": 0, "categories": {}, "avg_content_length": 0}
            
            categories = {}
            total_length = 0
            
            for entry in self.knowledge_store:
                total_length += len(entry.content)
                category = entry.metadata.get('category', 'uncategorized')
                categories[category] = categories.get(category, 0) + 1
            
            return {
                "total_entries": total_entries,
                "categories": categories,
                "avg_content_length": total_length / total_entries,
                "index_size": self.index.ntotal if self.index else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {"error": str(e)}

