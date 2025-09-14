from dotenv import load_dotenv
from typing import Annotated, Literal, Optional, List, Dict, Any, Union, Callable
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langchain_core.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
import os
import json
import logging
import sqlite3
from datetime import datetime
import uuid
import asyncio
from enum import Enum
from collections import defaultdict
from agents import Chatbot, VectorKnowledgeAgent, WebScrapingAgent, DatabaseQueryOrchestrator


load_dotenv()

# Enhanced Memory Models
class MemoryItem(BaseModel):
    """Individual memory item with semantic scoring"""
    content: str
    memory_type: str  # fact, preference, context, workflow_pattern
    relevance_score: float = 0.0
    frequency_accessed: int = 0
    last_accessed: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConversationMemory(BaseModel):
    """Enhanced conversation memory with hierarchical storage"""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    workflow_history: List[Dict[str, Any]] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    memory_items: List[MemoryItem] = Field(default_factory=list)
    conversation_summary: Optional[str] = Field(default=None)
    patterns_learned: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())

class WorkflowStep(BaseModel):
    """Represents a single step in the workflow"""
    step_id: str = Field(..., description="Unique step identifier")
    agent_name: str = Field(..., description="Name of the agent to use")
    action: str = Field(..., description="Action to perform")
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    requires_human_approval: bool = Field(default=False)
    human_approved: Optional[bool] = Field(default=None)
    status: str = Field(default="pending")  # pending, approved, rejected, completed, failed
    error_message: Optional[str] = Field(default=None)
    generated_prompt: Optional[str] = Field(default=None)
    execution_time: Optional[float] = Field(default=None)

class WorkflowPlan(BaseModel):
    """Complete workflow plan"""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    user_query: str = Field(..., description="Original user query")
    workflow_type: str = Field(..., description="Type of workflow")
    steps: List[WorkflowStep] = Field(..., description="Ordered list of workflow steps")
    estimated_duration: Optional[str] = Field(default=None)
    risk_level: str = Field(default="low")  # low, medium, high
    requires_approval: bool = Field(default=False)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)

class HumanInteraction(BaseModel):
    """Represents a human interaction point"""
    interaction_id: str = Field(..., description="Unique interaction identifier")
    interaction_type: str = Field(..., description="Type of interaction: approval, input, clarification")
    message: str = Field(..., description="Message to display to human")
    options: Optional[List[str]] = Field(default=None, description="Available options")
    user_response: Optional[str] = Field(default=None)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    timeout_seconds: Optional[int] = Field(default=300)  # 5 minutes default

# State Management
class OrchestratorState(TypedDict):
    """State for the orchestrating agent"""
    user_query: str
    user_id: str
    session_id: str
    workflow_plan: Optional[WorkflowPlan]
    current_step: int
    step_outputs: Dict[str, Any]
    memory: ConversationMemory
    human_interactions: List[HumanInteraction]
    final_result: Dict[str, Any]
    error_state: Optional[str]

# Semantic Memory Manager
class SemanticMemoryManager:
    """Manages semantic search and memory relevance scoring"""
    
    def __init__(self):
        self.memory_embeddings = {}
        self.memory_index = {}
    
    def add_memory(self, session_id: str, memory_data: Dict[str, Any]):
        """Add memory with semantic indexing"""
        self.memory_index[session_id] = memory_data
        # In production, use actual embeddings (e.g., OpenAI, Sentence Transformers)
        self._create_embedding(session_id, memory_data)
    
    def _create_embedding(self, session_id: str, memory_data: Dict[str, Any]):
        """Create semantic embedding for memory (mock implementation)"""
        # Mock embedding - replace with actual embedding model
        text_content = f"{memory_data.get('conversation_summary', '')} {' '.join([str(h) for h in memory_data.get('conversation_history', [])])}"
        # Simple hash-based mock embedding
        embedding = hash(text_content) % 1000
        self.memory_embeddings[session_id] = embedding
    
    def semantic_search(self, query: str, user_id: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity"""
        query_embedding = hash(query) % 1000
        scored = []
        for session_id, memory_data in self.memory_index.items():
            if user_id and memory_data.get('user_id') != user_id:
                continue
            sim = 1.0 / (1.0 + abs(query_embedding - self.memory_embeddings.get(session_id, 0)))
            scored.append({"session_id": session_id, "data": memory_data, "similarity_score": sim})
        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

# Conversation Summarizer
class ConversationSummarizer:
    """Intelligently summarize conversations preserving key information"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def summarize(self, conversation_history: List[Dict[str, Any]], preserve_facts: bool = True) -> str:
        """Generate intelligent conversation summary"""
        if not conversation_history:
            return ""
        
        # Extract different types of information
        recent_messages = conversation_history[-10:]
        user_queries = [msg.get('user_message', '') for msg in recent_messages if msg.get('user_message')]
        bot_responses = [msg.get('bot_response', '') for msg in recent_messages if msg.get('bot_response')]
        
        summary_prompt = f"""
        Summarize this conversation, preserving:
        1. Key facts and information discussed
        2. User preferences and patterns
        3. Important decisions or conclusions
        4. Recurring themes
        
        Recent user queries: {user_queries}
        Recent bot responses: {bot_responses}
        
        Create a concise but comprehensive summary:
        """
        
        try:
            summary = self.llm.invoke(summary_prompt).content
            return summary
        except Exception as e:
            logging.warning(f"Failed to generate summary: {e}")
            # Fallback summary
            return f"Conversation covered {len(user_queries)} topics including key themes from recent interactions."
    
    def extract_patterns(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract user behavior patterns from conversation"""
        patterns = {
            'preferred_agents': defaultdict(int),
            'common_query_types': defaultdict(int),
            'interaction_times': [],
            'average_session_length': 0
        }
        
        for msg in conversation_history:
            if msg.get('workflow_id'):
                patterns['common_query_types'][msg.get('type', 'unknown')] += 1
            if msg.get('timestamp'):
                patterns['interaction_times'].append(msg['timestamp'])
        
        patterns['average_session_length'] = len(conversation_history)
        return dict(patterns)

# Enhanced Memory Manager
class EnhancedMemoryManager:
    """Advanced memory manager with semantic search and intelligent summarization"""
    
    def __init__(self, llm, db_path: str = "./enhanced_orchestrator_memory.db"):
        self.db_path = db_path
        self.llm = llm
        self.semantic_manager = SemanticMemoryManager()
        self.summarizer = ConversationSummarizer(llm)
        self._init_database()
    
    def _init_database(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table with enhanced fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_data TEXT NOT NULL,
                conversation_summary TEXT,
                memory_items TEXT,
                patterns_learned TEXT,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                access_frequency INTEGER DEFAULT 0
            )
        ''')
        
        # Memory items table for granular storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_items (
                item_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                relevance_score REAL DEFAULT 0.0,
                frequency_accessed INTEGER DEFAULT 0,
                last_accessed TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES conversations (session_id)
            )
        ''')
        
        # User patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_patterns (
                user_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (user_id, pattern_type)
            )
        ''')
        
        # Workflow history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workflow_history (
                workflow_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_query TEXT NOT NULL,
                workflow_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                workflow_data TEXT,
                FOREIGN KEY (session_id) REFERENCES conversations (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, memory: ConversationMemory):
        """Save conversation with enhanced processing"""
        # Generate summary if conversation is long
        if len(memory.conversation_history) > 15:
            memory.conversation_summary = self.summarizer.summarize(memory.conversation_history)
        
        # Extract patterns
        if len(memory.conversation_history) > 5:
            memory.patterns_learned = self.summarizer.extract_patterns(memory.conversation_history)
        
        # Compress old conversation history, keeping recent items
        if len(memory.conversation_history) > 30:
            memory.conversation_history = memory.conversation_history[-30:]
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO conversations 
            (session_id, user_id, conversation_data, conversation_summary, 
             memory_items, patterns_learned, created_at, last_updated, access_frequency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT access_frequency FROM conversations WHERE session_id = ?) + 1, 1))
        ''', (
            memory.session_id, memory.user_id, json.dumps(memory.dict()),
            memory.conversation_summary, json.dumps([item.dict() for item in memory.memory_items]),
            json.dumps(memory.patterns_learned), memory.created_at, memory.last_updated,
            memory.session_id
        ))
        
        # Save individual memory items
        for item in memory.memory_items:
            cursor.execute('''
                INSERT OR REPLACE INTO memory_items 
                (item_id, session_id, memory_type, content, relevance_score, 
                 frequency_accessed, last_accessed, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"{memory.session_id}_{uuid.uuid4().hex[:8]}", memory.session_id,
                item.memory_type, item.content, item.relevance_score,
                item.frequency_accessed, item.last_accessed, json.dumps(item.metadata)
            ))
        
        conn.commit()
        conn.close()
        
        # Update semantic index
        self.semantic_manager.add_memory(memory.session_id, memory.dict())
    
    def load_conversation(self, session_id: str) -> Optional[ConversationMemory]:
        """Load conversation with access frequency tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT conversation_data FROM conversations WHERE session_id = ?
        ''', (session_id,))
        result = cursor.fetchone()
        
        if result:
            # Increment access frequency
            cursor.execute('''
                UPDATE conversations SET access_frequency = access_frequency + 1 WHERE session_id = ?
            ''', (session_id,))
            conn.commit()
        
        conn.close()
        
        if result:
            try:
                data = json.loads(result[0])
                return ConversationMemory(**data)
            except Exception as e:
                logging.error(f"Failed to load conversation: {e}")
                return None
        return None
    
    def save_workflow(self, workflow_plan: WorkflowPlan, session_id: str, status: str):
        """Save workflow to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO workflow_history 
            (workflow_id, session_id, user_query, workflow_type, status, 
             created_at, completed_at, workflow_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            workflow_plan.workflow_id, session_id, workflow_plan.user_query,
            workflow_plan.workflow_type, status, datetime.now().isoformat(),
            datetime.now().isoformat() if status == "completed" else None,
            json.dumps(workflow_plan.dict())
        ))
        
        conn.commit()
        conn.close()
    
    def get_relevant_context(self, query: str, user_id: str, max_items: int = 5) -> Dict[str, Any]:
        """Get most relevant context for current query"""
        # Semantic search across all user's memories
        semantic_results = self.semantic_manager.semantic_search(query, user_id, top_k=max_items)
        
        # Get user patterns
        user_patterns = self._get_user_patterns(user_id)
        
        # Combine results
        relevant_context = {
            'semantic_memories': semantic_results,
            'user_patterns': user_patterns,
            'context_summary': self._generate_context_summary(semantic_results, query)
        }
        
        return relevant_context
    
    def _get_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Retrieve learned user patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT pattern_type, pattern_data, confidence_score 
            FROM user_patterns WHERE user_id = ?
        ''', (user_id,))
        results = cursor.fetchall()
        conn.close()
        
        patterns = {}
        for pattern_type, pattern_data, confidence in results:
            try:
                patterns[pattern_type] = {
                    'data': json.loads(pattern_data),
                    'confidence': confidence
                }
            except json.JSONDecodeError:
                continue
        
        return patterns
    
    def _generate_context_summary(self, semantic_results: List[Dict], query: str) -> str:
        """Generate a summary of relevant context for the current query"""
        if not semantic_results:
            return "No relevant context found."
        
        context_items = []
        for result in semantic_results:
            summary = result['data'].get('conversation_summary', 'No summary available')
            score = result['similarity_score']
            context_items.append(f"Context (relevance: {score:.2f}): {summary[:100]}...")
        
        return "\n".join(context_items)

# Mock Agent Classes (replace with your actual agent implementations)
class MockChatbot:
    """Mock chatbot for testing"""
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model
        self.api_key = api_key
    
    def chat(self, query: str) -> str:
        return f"Chatbot response to: {query[:50]}..."

class MockWebScrapingAgent:
    """Mock web scraping agent for testing"""
    def __init__(self, model: str = None, api_key: str = None, tavily_api_key: str = None):
        self.model = model
        self.api_key = api_key
        self.tavily_api_key = tavily_api_key
    
    def run(self, prompt: str) -> Dict[str, Any]:
        return {"scraped_data": f"Web data for: {prompt[:50]}...", "status": "success"}

class MockDatabaseOrchestrator:
    """Mock database orchestrator for testing"""
    def __init__(self, model: str = None, api_key: str = None, database_directory: str = None):
        self.model = model
        self.api_key = api_key
        self.database_directory = database_directory
    
    def query(self, query: str) -> Dict[str, Any]:
        return {"database_result": f"DB result for: {query[:50]}...", "rows": 10}

class MockVectorKnowledgeAgent:
    """Mock vector knowledge agent for testing"""
    def __init__(self, model: str = None, api_key: str = None, index_path: str = None):
        self.model = model
        self.api_key = api_key
        self.index_path = index_path
    
    def query_knowledge(self, query: str) -> Dict[str, Any]:
        return {"knowledge_result": f"Knowledge for: {query[:50]}...", "confidence": 0.8}
    
    def generate_contextual_prompt(self, user_query: str, task_type: str, context_limit: int = 3) -> Dict[str, Any]:
        return {"optimized_prompt": f"Enhanced prompt for {task_type}: {user_query}"}
    
    def add_knowledge(self, content: str, metadata: Dict = None) -> str:
        return "knowledge_added"

class OrchestrationAgent:
    """
    Main orchestrating agent that coordinates all other agents with human-in-the-loop,
    memory management, and prompt generation capabilities.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        model_provider: str = "google_genai",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        database_directory: str = "./Databases",
        knowledge_base_path: str = "./knowledge_base",
        tavily_api_key: Optional[str] = None,
        human_interaction_handler: Optional[Callable] = None
    ):
        self.model = model
        self.model_provider = model_provider
        self.temperature = temperature
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key is required. Set GEMINI_API_KEY environment variable.")
        
        # Initialize LLM with proper error handling
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature,
                google_api_key=self.api_key
            )
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
            raise
        
        # Initialize memory manager
        self.memory_manager = EnhancedMemoryManager(self.llm)
        
        # Initialize child agents
        self._init_child_agents(database_directory, knowledge_base_path, tavily_api_key)
        
        # Human interaction handler
        self.human_interaction_handler = human_interaction_handler or self._default_human_handler
        
        # Build the orchestration graph
        self.app = self._build_orchestration_graph()
    
    def _init_child_agents(self, database_directory: str, knowledge_base_path: str, tavily_api_key: Optional[str]):
        """Initialize all child agents with fallback to mock agents"""
        try:
            # Try to import actual agents
            from agents import Chatbot, WebScrapingAgent, DatabaseQueryOrchestrator, VectorKnowledgeAgent
            
            self.chatbot = Chatbot(
                model="gemini-1.5-flash",
                api_key=self.api_key
            )
            
            self.web_scraper = WebScrapingAgent(
                model=self.model,
                api_key=self.api_key,
                tavily_api_key=tavily_api_key
            )
            
            self.database_orchestrator = DatabaseQueryOrchestrator(
                model=self.model,
                api_key=self.api_key,
                database_directory=database_directory
            )
            
            self.knowledge_agent = VectorKnowledgeAgent(
                model=self.model,
                api_key=self.api_key,
                index_path=knowledge_base_path
            )
            
        except ImportError as e:
            logging.warning(f"Could not import all agents, using mock agents: {e}")
            # Initialize mock agents for testing
            self.chatbot = Chatbot(model=self.model, api_key=self.api_key)
            self.web_scraper = WebScrapingAgent(model=self.model, api_key=self.api_key, tavily_api_key=tavily_api_key)
            self.database_orchestrator = DatabaseQueryOrchestrator(model=self.model, api_key=self.api_key, database_directory=database_directory)
            self.knowledge_agent = VectorKnowledgeAgent(model=self.model, api_key=self.api_key, index_path=knowledge_base_path)
    
    def _build_orchestration_graph(self):
        """Build the orchestration workflow graph"""
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("memory_load", self._memory_load_node)
        workflow.add_node("workflow_planning", self._workflow_planning_node)
        workflow.add_node("human_approval", self._human_approval_node)
        workflow.add_node("prompt_generation", self._prompt_generation_node)
        workflow.add_node("step_execution", self._step_execution_node)
        workflow.add_node("execution_review", self._execution_review)
        workflow.add_node("result_consolidation", self._result_consolidation_node)
        workflow.add_node("memory_save", self._memory_save_node)
        
    # Define workflow with interrupts
        workflow.set_entry_point("memory_load")
        workflow.add_edge("memory_load", "workflow_planning")
        workflow.add_edge("workflow_planning", "human_approval")
        workflow.add_edge("human_approval", "prompt_generation")
        workflow.add_edge("prompt_generation", "step_execution")
        
        # Add human review after execution
        workflow.add_edge("step_execution", "execution_review")
        workflow.add_edge("execution_review", "result_consolidation")
        workflow.add_edge("result_consolidation", "memory_save")
        workflow.add_edge("memory_save", END)
    
        # Add interrupts for human review
        return workflow.compile(interrupt_after=["step_execution"])


    def _memory_load_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Load conversation memory"""
        print("--- 🧠 LOADING MEMORY ---")
        
        session_id = state.get("session_id")
        user_id = state.get("user_id")
        
        # Load existing memory or create new
        memory = self.memory_manager.load_conversation(session_id)
        if not memory:
            memory = ConversationMemory(
                session_id=session_id,
                user_id=user_id
            )
        
        # Add current query to conversation history
        memory.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": state["user_query"],
            "type": "user_input"
        })
        
        return {"memory": memory}
    
    def _workflow_planning_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Plan the workflow based on user query and memory"""
        print("--- 📋 WORKFLOW PLANNING ---")
        
        user_query = state["user_query"]
        memory = state["memory"]
        
        # Extract context from memory
        recent_context = ""
        if memory.conversation_history:
            recent_messages = memory.conversation_history[-5:]  # Last 5 interactions
            recent_context = "\n".join([
                f"- {msg.get('type', 'unknown')}: {str(msg.get('user_message', msg.get('bot_response', '')))[:100]}"
                for msg in recent_messages
            ])
        
        # Available agents and their capabilities
        agent_capabilities = {
            "chatbot": "General conversation, Q&A, explanations",
            "web_scraper": "Web research, data collection from URLs, real-time information",
            "database_orchestrator": "Query databases (SQL/flat files), data analysis",
            "knowledge_agent": "Query knowledge base, contextual prompt generation"
        }
        
        class WorkflowPlanOutput(BaseModel):
            workflow_type: str = Field(description="Type of workflow: simple, complex, research, analysis")
            estimated_duration: str = Field(description="Estimated time to complete")
            risk_level: str = Field(description="Risk level: low, medium, high")
            requires_approval: bool = Field(description="Whether human approval is needed")
            steps: List[Dict[str, Any]] = Field(description="List of workflow steps")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a workflow planning expert. Analyze the user query and create an optimal workflow.

Available Agents:
{{json.dumps(agent_capabilities, indent=2)}}

Recent Context:
{{recent_context}}

Rules:
1. Choose the most appropriate agents for the task
2. Consider dependencies between steps
3. Mark steps that might need human approval (data modification, external actions)
4. Estimate realistic timeframes
5. Assess risk levels based on actions involved

For each step, specify:
- step_id: unique identifier
- agent_name: which agent to use
- action: what action to perform
- inputs: required inputs
- requires_human_approval: boolean
- description: human-readable description"""),
            ("user", "User Query: {query}")
        ])
        
        try:
            planner = prompt_template | self.llm.with_structured_output(WorkflowPlanOutput)
            plan_output = planner.invoke({"query": user_query})
            
            # Create WorkflowStep objects
            workflow_steps = []
            for i, step_data in enumerate(plan_output.steps):
                step = WorkflowStep(
                    step_id=step_data.get("step_id", f"step_{i}"),
                    agent_name=step_data.get("agent_name", "chatbot"),
                    action=step_data.get("action", "process"),
                    inputs=step_data.get("inputs", {}),
                    requires_human_approval=step_data.get("requires_human_approval", False)
                )
                workflow_steps.append(step)
            
            workflow_plan = WorkflowPlan(
                workflow_id=str(uuid.uuid4()),
                user_query=user_query,
                workflow_type=plan_output.workflow_type,
                steps=workflow_steps,
                estimated_duration=plan_output.estimated_duration,
                risk_level=plan_output.risk_level,
                requires_approval=plan_output.requires_approval
            )
            
        except Exception as e:
            logging.error(f"Workflow planning failed: {e}")
            # Fallback to simple chatbot workflow
            workflow_plan = WorkflowPlan(
                workflow_id=str(uuid.uuid4()),
                user_query=user_query,
                workflow_type="simple",
                steps=[WorkflowStep(
                    step_id="fallback_step",
                    agent_name="chatbot",
                    action="chat",
                    inputs={"query": user_query}
                )],
                estimated_duration="30 seconds",
                risk_level="low",
                requires_approval=False
            )
        
        return {
            "workflow_plan": workflow_plan,
            "current_step": 0,
            "step_outputs": {}
        }
    
    def _human_approval_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Handle human approval if required"""
        print("--- 👤 HUMAN APPROVAL ---")
        
        workflow_plan = state["workflow_plan"]
        
        if not workflow_plan.requires_approval:
            return {"human_interactions": []}
        
        # Create human interaction for workflow approval
        interaction = HumanInteraction(
            interaction_id=str(uuid.uuid4()),
            interaction_type="approval",
            message="""
Workflow Plan Approval Required:

Query: {workflow_plan.user_query}
Type: {workflow_plan.workflow_type}
Risk Level: {workflow_plan.risk_level}
Estimated Duration: {workflow_plan.estimated_duration}

Steps:
{chr(10).join([f"{i+1}. {step.agent_name}: {step.action}" for i, step in enumerate(workflow_plan.steps)])}

Do you approve this workflow?
            """.strip(),
            options=["approve", "reject", "modify"]
        )
        
        # Get human response
        response = self.human_interaction_handler(interaction)
        interaction.user_response = response
        
        if response != "approve":
            return {
                "human_interactions": [interaction],
                "error_state": f"Workflow {response}d by user"
            }
        
        return {"human_interactions": [interaction]}
    
    def _prompt_generation_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Generate optimized prompts for each workflow step"""
        print("--- ✨ PROMPT GENERATION ---")
        
        workflow_plan = state["workflow_plan"]
        memory = state["memory"]
        
        # Generate contextual prompts for each step
        updated_steps = []
        
        for step in workflow_plan.steps:
            try:
                if step.agent_name == "knowledge_agent" and hasattr(self.knowledge_agent, 'generate_contextual_prompt'):
                    # Use knowledge agent's contextual prompt generation
                    context_result = self.knowledge_agent.generate_contextual_prompt(
                        user_query=f"{step.action}: {step.inputs.get('query', workflow_plan.user_query)}",
                        task_type=step.action,
                        context_limit=3
                    )
                    step.generated_prompt = context_result["optimized_prompt"]
                else:
                    # Generate prompt using LLM
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", """You are a prompt engineer. Create an optimized prompt for the {step.agent_name} agent.

Agent Capabilities:
- chatbot: General conversation, explanations
- web_scraper: Web research, data extraction
- database_orchestrator: Database queries, data analysis
- knowledge_agent: Knowledge retrieval, contextual information

Context from memory: {json.dumps(memory.conversation_history[-3:], indent=2) if memory.conversation_history else "No previous context"}

Generate a clear, specific prompt that will help the agent complete this task effectively."""),
                        ("user", f"Task: {step.action}\nInputs: {json.dumps(step.inputs)}\nOriginal Query: {workflow_plan.user_query}")
                    ])
                    
                    response = self.llm.invoke(prompt_template.format_messages())
                    step.generated_prompt = response.content
                    
            except Exception as e:
                logging.warning(f"Failed to generate prompt for step {step.step_id}: {e}")
                step.generated_prompt = f"Execute {step.action} with inputs: {json.dumps(step.inputs)}"
            
            updated_steps.append(step)
        
        # Update workflow plan with generated prompts
        workflow_plan.steps = updated_steps
        
        return {"workflow_plan": workflow_plan}
    
    def _step_execution_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Execute workflow steps sequentially"""
        print("--- ⚡ STEP EXECUTION ---")
        
        workflow_plan = state["workflow_plan"]
        current_step = state.get("current_step", 0)
        step_outputs = state.get("step_outputs", {})
        
        # Execute each step
        for i, step in enumerate(workflow_plan.steps[current_step:], current_step):
            print(f"Executing Step {i+1}: {step.agent_name} - {step.action}")
            
            # Check if human approval is needed for this step
            if step.requires_human_approval:
                interaction = HumanInteraction(
                    interaction_id=str(uuid.uuid4()),
                    interaction_type="approval",
                    message=f"Approve execution of: {step.action} using {step.agent_name}?",
                    options=["approve", "skip", "abort"]
                )
                
                response = self.human_interaction_handler(interaction)
                if response != "approve":
                    step.status = "skipped" if response == "skip" else "rejected"
                    continue
            
            # Execute the step
            try:
                start_time = datetime.now()
                result = self._execute_single_step(step, step_outputs)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                step.outputs = result
                step.status = "completed"
                step.execution_time = execution_time
                step_outputs[step.step_id] = result
                
            except Exception as e:
                step.status = "failed"
                step.error_message = str(e)
                print(f"Step {i+1} failed: {e}")
                logging.error(f"Step execution failed: {e}")
        
        return {
            "step_outputs": step_outputs,
            "current_step": len(workflow_plan.steps),
            "workflow_plan": workflow_plan
        }
    
    def _execute_single_step(self, step: WorkflowStep, previous_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        agent_name = step.agent_name
        action = step.action
        inputs = step.inputs.copy()
        
        # Add generated prompt to inputs
        if step.generated_prompt:
            inputs["generated_prompt"] = step.generated_prompt
        
        # Add previous step outputs if referenced
        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to previous step output
                ref_step_id = value[1:]  # Remove $
                if ref_step_id in previous_outputs:
                    inputs[key] = previous_outputs[ref_step_id]
        
        # Route to appropriate agent
        try:
            if agent_name == "chatbot" and self.chatbot:
                query = step.generated_prompt or inputs.get("query", inputs.get("message", ""))
                result = self.chatbot.chat(query)
                return {"response": result, "type": "chat_response"}
                
            elif agent_name == "web_scraper" and self.web_scraper:
                prompt = step.generated_prompt or inputs.get("prompt", inputs.get("query", ""))
                result = self.web_scraper.run(prompt)
                return {"data": result, "type": "scraped_data"}
                
            elif agent_name == "database_orchestrator" and self.database_orchestrator:
                query = step.generated_prompt or inputs.get("query", "")
                result = self.database_orchestrator.query(query)
                return {"data": result, "type": "database_result"}
                
            elif agent_name == "knowledge_agent" and self.knowledge_agent:
                if action == "query":
                    query = inputs.get("query", "")
                    result = self.knowledge_agent.query_knowledge(query)
                    return {"data": result, "type": "knowledge_result"}
                elif action == "add_knowledge":
                    content = inputs.get("content", "")
                    metadata = inputs.get("metadata", {})
                    result = self.knowledge_agent.add_knowledge(content, metadata)
                    return {"status": result, "type": "knowledge_added"}
            
            else:
                # Fallback to basic LLM call
                prompt = step.generated_prompt or f"Task: {action}\nInputs: {json.dumps(inputs)}"
                result = self.llm.invoke(prompt)
                return {"response": result.content, "type": "llm_response"}
                
        except Exception as e:
            logging.error(f"Agent execution failed: {e}")
            return {"error": str(e), "type": "error"}
    
    def _execution_review(self, state: OrchestratorState) -> Dict[str, Any]:
        """Analyze results from all steps"""
        print("--- 📊 RESULT ANALYSIS ---")
        



    def _result_consolidation_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Consolidate results from all steps"""
        print("--- 📊 RESULT CONSOLIDATION ---")
        
        workflow_plan = state["workflow_plan"]
        step_outputs = state["step_outputs"]
        
        # Analyze all step outputs and create final result
        successful_steps = [step for step in workflow_plan.steps if step.status == "completed"]
        failed_steps = [step for step in workflow_plan.steps if step.status == "failed"]
        
        # Generate final summary
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a result analyst. Consolidate the workflow execution results into a comprehensive response.

Create a final answer that:
1. Directly addresses the user's original query
2. Incorporates insights from all successful steps
3. Mentions any limitations from failed steps
4. Provides actionable conclusions

Be concise but comprehensive."""),
                ("user", f"Original Query: {workflow_plan.user_query}\n\nStep Results: {json.dumps(step_outputs, indent=2)}")
            ])
            
            consolidation_result = self.llm.invoke(prompt_template.format_messages())
            summary = consolidation_result.content
            
        except Exception as e:
            logging.error(f"Result consolidation failed: {e}")
            summary = f"Workflow completed with {len(successful_steps)} successful steps and {len(failed_steps)} failed steps."
        
        final_result = {
            "query": workflow_plan.user_query,
            "workflow_id": workflow_plan.workflow_id,
            "summary": summary,
            "steps_completed": len(successful_steps),
            "steps_failed": len(failed_steps),
            "execution_details": {
                "workflow_type": workflow_plan.workflow_type,
                "risk_level": workflow_plan.risk_level,
                "total_execution_time": sum(step.execution_time or 0 for step in workflow_plan.steps)
            },
            "raw_outputs": step_outputs,
            "step_details": [
                {
                    "step_id": step.step_id,
                    "agent": step.agent_name,
                    "action": step.action,
                    "status": step.status,
                    "execution_time": step.execution_time
                }
                for step in workflow_plan.steps
            ]
        }
        
        return {"final_result": final_result}
    
    def _memory_save_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Save results and conversation to memory"""
        print("--- 💾 SAVING MEMORY ---")
        
        memory = state["memory"]
        final_result = state["final_result"]
        workflow_plan = state["workflow_plan"]
        
        try:
            # Add workflow result to conversation history
            memory.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "bot_response": final_result["summary"],
                "type": "workflow_result",
                "workflow_id": workflow_plan.workflow_id
            })
            
            # Add workflow to workflow history
            memory.workflow_history.append({
                "workflow_id": workflow_plan.workflow_id,
                "query": workflow_plan.user_query,
                "type": workflow_plan.workflow_type,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
            
            # Update timestamp
            memory.last_updated = datetime.now().isoformat()
            
            # Save to database
            self.memory_manager.save_conversation(memory)
            self.memory_manager.save_workflow(workflow_plan, memory.session_id, "completed")
            
        except Exception as e:
            logging.error(f"Memory save failed: {e}")
        
        return {"memory": memory}
    
    def _default_human_handler(self, interaction: HumanInteraction) -> str:
        """Default human interaction handler (console-based)"""
        print(f"\n{'='*50}")
        print(f"HUMAN INPUT REQUIRED")
        print(f"{'='*50}")
        print(interaction.message)
        
        if interaction.options:
            print(f"\nOptions: {', '.join(interaction.options)}")
            while True:
                response = input("\nYour choice: ").strip().lower()
                if response in [opt.lower() for opt in interaction.options]:
                    return response
                print("Invalid option. Please choose from the available options.")
        else:
            return input("\nYour response: ").strip()
    
    def run_workflow(
        self,
        user_query: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main entry point to run the orchestration workflow"""
        if not session_id:
            session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        
        inputs = {
            "user_query": user_query,
            "user_id": user_id,
            "session_id": session_id
        }
        
        try:
            print("--- 🚀 STARTING ORCHESTRATION WORKFLOW ---")
            final_state = self.app.invoke(inputs)
            print("--- 🏁 ORCHESTRATION COMPLETED ---")
            
            return final_state.get("final_result", {"error": "No final result generated"})
            
        except Exception as e:
            logging.error(f"Orchestration workflow failed: {e}")
            return {"error": f"Workflow execution failed: {str(e)}"}
    
    def run_workflow_with_streaming(
        self,
        user_query: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run workflow with step-by-step streaming output"""
        if not session_id:
            session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        
        inputs = {
            "user_query": user_query,
            "user_id": user_id,
            "session_id": session_id
        }
        
        final_state = {}
        
        print("--- 🚀 STARTING ORCHESTRATION WORKFLOW ---")
        try:
            for output in self.app.stream(inputs, {"recursion_limit": 15}):
                for key, value in output.items():
                    print(f"\n--- ✅ Output from node: {key} ---")
                    if key in ["workflow_plan", "final_result"]:
                        print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
                    else:
                        print(f"Updated state: {key}")
                    final_state.update(value)
            
            print("\n--- 🏁 ORCHESTRATION COMPLETED ---")
            return final_state.get("final_result", {"error": "No final result generated"})
            
        except Exception as e:
            logging.error(f"Orchestration workflow failed: {e}")
            return {"error": f"Workflow execution failed: {str(e)}"}

# Flask Integration
def create_orchestration_flask_app():
    """Create Flask app with orchestration agent"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    # Initialize orchestration agent
    orchestrator = OrchestrationAgent(
        api_key=os.getenv("GEMINI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    @app.route("/", methods=["GET"])
    def hello():
        return "<p>Orchestration Agent API</p>"
    
    @app.route("/orchestrate", methods=["POST"])
    def orchestrate():
        data = request.json
        user_query = data.get("query", "")
        user_id = data.get("user_id", "anonymous")
        session_id = data.get("session_id")
        
        if not user_query:
            return jsonify({"error": "No query provided"}), 400
        
        try:
            result = orchestrator.run_workflow(user_query, user_id, session_id)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/memory/<session_id>", methods=["GET"])
    def get_memory(session_id):
        try:
            memory = orchestrator.memory_manager.load_conversation(session_id)
            if memory:
                return jsonify(memory.dict())
            else:
                return jsonify({"error": "Session not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app




# Example usage
if __name__ == "__main__":
    # Test with environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment variables")
        print("Set it with: export GEMINI_API_KEY='your-api-key'")
    
    try:
        # Initialize the orchestration agent
        orchestrator = OrchestrationAgent(
            api_key=api_key,
            tavily_api_key=os.getenv("TAVILY_API_KEY")
        )
        
        # Example workflow execution
        result = orchestrator.run_workflow_with_streaming(
            user_query="What is the weather like today?",
            user_id="test_user",
            session_id="test_session_1"
        )
        
        print("\nFinal Result:")
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"Failed to run orchestration: {e}")
        logging.error(f"Main execution failed: {e}")

