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




# class WorkflowStep(BaseModel):
#     """Individual step in a workflow"""
#     step_id: str
#     agent_type: AgentType
#     action: str
#     parameters: Dict[str, Any]
#     requires_human_approval: bool = False
#     depends_on: List[str] = []
    
# class WorkflowPlan(BaseModel):
#     """Complete workflow execution plan"""
#     plan_id: str
#     user_query: str
#     steps: List[WorkflowStep]
#     estimated_duration: str
#     requires_human_approval: bool
#     reasoning: str

# @dataclass
# class ConversationMemory:
#     """Conversation memory structure"""
#     user_id: str
#     messages: List[Dict[str, Any]]
#     context: Dict[str, Any]
#     last_workflow: Optional[str]
#     created_at: datetime
#     updated_at: datetime

# class SuperAgentState(TypedDict):
#     """State for SuperAgent workflow"""
#     user_id: str
#     user_query: str
#     conversation_memory: Optional[ConversationMemory]
#     vector_context: str
#     workflow_plan: Optional[WorkflowPlan]
#     current_step: int
#     step_results: Dict[str, Any]
#     requires_human_approval: bool
#     human_approval_received: bool
#     final_response: str
#     error_message: str

# # Updated SuperAgent class with intelligent routing
# class SuperAgent:
#     """
#     Ultimate AI Assistant with intelligent query routing
#     """
    
#     def __init__(
#         self,
#         model: str = "gemini-2.5-pro",
#         model_provider: str = "google_genai", 
#         temperature: float = 0.1,
#         api_key: Optional[str] = None,
#         tavily_api_key: Optional[str] = None,
#         database_directory: str = "./Databases",
#         knowledge_base_path: str = "./knowledge_base",
#         memory_db_path: str = "./conversation_memory.db"
#     ):
#         self.model = model
#         self.model_provider = model_provider
#         self.temperature = temperature
#         self.api_key = api_key or os.getenv("LLM_API_KEY")
#         self.memory_db_path = memory_db_path
        
#         if not self.api_key:
#             raise ValueError("API key is required.")
        
#         # Initialize LLM
#         self.llm = init_chat_model(
#             model=self.model,
#             model_provider=self.model_provider,
#             temperature=self.temperature,
#             api_key=self.api_key
#         )
        
#         # Initialize all specialized agents
#         self.agents = self._initialize_agents(tavily_api_key, database_directory, knowledge_base_path)
        
#         # Initialize intelligent router
#         self.router = IntelligentRouterAgent(self.llm, self.agents)
        
#         # Initialize conversation memory database
#         self._init_memory_db()
        
#         logger.info("SuperAgent initialized with intelligent routing")
    
#     def _initialize_agents(self, tavily_api_key, database_directory, knowledge_base_path):
#         """Initialize all specialized agents"""
#         agents = {}
        
#         try:
#             # Basic Chatbot
#             agents[AgentType.CHATBOT] = Chatbot(
#                 model=self.model,
#                 temperature=self.temperature,
#                 api_key=self.api_key
#             )
            
#             # Web Scraping Agent
#             agents[AgentType.WEB_SCRAPING] = WebScrapingAgent(
#                 model=self.model,
#                 temperature=self.temperature,
#                 api_key=self.api_key,
#                 tavily_api_key=tavily_api_key
#             )
            
#             # Database Query Orchestrator
#             agents[AgentType.DATABASE_QUERY] = DatabaseQueryOrchestrator(
#                 model=self.model,
#                 temperature=self.temperature,
#                 api_key=self.api_key,
#                 database_directory=database_directory
#             )
            
#             # Vector Knowledge Agent
#             agents[AgentType.VECTOR_KNOWLEDGE] = VectorKnowledgeAgent(
#                 model=self.model,
#                 temperature=self.temperature,
#                 api_key=self.api_key,
#                 index_path=knowledge_base_path
#             )
            
#             logger.info("All agents initialized successfully")
#             return agents
            
#         except Exception as e:
#             logger.error(f"Error initializing agents: {e}")
#             raise
    
#     def _init_memory_db(self):
#         """Initialize SQLite database for conversation memory"""
#         try:
#             conn = sqlite3.connect(self.memory_db_path)
#             cursor = conn.cursor()
            
#             cursor.execute('''
#                 CREATE TABLE IF NOT EXISTS conversations (
#                     user_id TEXT PRIMARY KEY,
#                     messages TEXT,
#                     context TEXT,
#                     last_workflow TEXT,
#                     created_at TIMESTAMP,
#                     updated_at TIMESTAMP
#                 )
#             ''')
            
#             conn.commit()
#             conn.close()
#             logger.info("Memory database initialized")
            
#         except Exception as e:
#             logger.error(f"Error initializing memory database: {e}")
    
#     def _load_conversation_memory(self, user_id: str) -> Optional[ConversationMemory]:
#         """Load conversation memory for a user"""
#         try:
#             conn = sqlite3.connect(self.memory_db_path)
#             cursor = conn.cursor()
            
#             cursor.execute(
#                 "SELECT * FROM conversations WHERE user_id = ?", 
#                 (user_id,)
#             )
#             row = cursor.fetchone()
#             conn.close()
            
#             if row:
#                 return ConversationMemory(
#                     user_id=row[0],
#                     messages=json.loads(row[1]),
#                     context=json.loads(row[2]),
#                     last_workflow=row[3],
#                     created_at=datetime.fromisoformat(row[4]),
#                     updated_at=datetime.fromisoformat(row[5])
#                 )
#             return None
            
#         except Exception as e:
#             logger.error(f"Error loading conversation memory: {e}")
#             return None
    
#     def _save_conversation_memory(self, memory: ConversationMemory):
#         """Save conversation memory for a user"""
#         try:
#             conn = sqlite3.connect(self.memory_db_path)
#             cursor = conn.cursor()
            
#             memory.updated_at = datetime.now()
            
#             cursor.execute('''
#                 INSERT OR REPLACE INTO conversations 
#                 (user_id, messages, context, last_workflow, created_at, updated_at)
#                 VALUES (?, ?, ?, ?, ?, ?)
#             ''', (
#                 memory.user_id,
#                 json.dumps(memory.messages),
#                 json.dumps(memory.context),
#                 memory.last_workflow,
#                 memory.created_at.isoformat(),
#                 memory.updated_at.isoformat()
#             ))
            
#             conn.commit()
#             conn.close()
#             logger.info(f"Memory saved for user {memory.user_id}")
            
#         except Exception as e:
#             logger.error(f"Error saving conversation memory: {e}")
    
#     def _build_workflow(self):
#         """Build the SuperAgent LangGraph workflow"""
#         workflow = StateGraph(SuperAgentState)
        
#         # Add workflow nodes
#         workflow.add_node("load_memory", self._load_memory_node)
#         workflow.add_node("analyze_query", self._analyze_query_node)
#         workflow.add_node("create_plan", self._create_plan_node)
#         workflow.add_node("human_approval", self._human_approval_node)
#         workflow.add_node("execute_workflow", self._execute_workflow_node)
#         workflow.add_node("finalize_response", self._finalize_response_node)
#         workflow.add_node("save_memory", self._save_memory_node)
        
#         # Define workflow edges
#         workflow.set_entry_point("load_memory")
#         workflow.add_edge("load_memory", "analyze_query")
#         workflow.add_edge("analyze_query", "create_plan")
        
#         # Conditional edge for human approval
#         workflow.add_conditional_edges(
#             "create_plan",
#             lambda state: "human_approval" if state.get("requires_human_approval") else "execute_workflow"
#         )
        
#         workflow.add_edge("human_approval", "execute_workflow")
#         workflow.add_edge("execute_workflow", "finalize_response")
#         workflow.add_edge("finalize_response", "save_memory")
#         workflow.add_edge("save_memory", END)
        
#         return workflow.compile()
    
#     def _load_memory_node(self, state: SuperAgentState) -> Dict[str, Any]:
#         """Load conversation memory for the user"""
#         logger.info("Loading conversation memory")
        
#         memory = self._load_conversation_memory(state["user_id"])
        
#         return {
#             "conversation_memory": memory,
#             "current_step": 0,
#             "step_results": {},
#             "requires_human_approval": False,
#             "human_approval_received": False
#         }
    
#     def _analyze_query_node(self, state: SuperAgentState) -> Dict[str, Any]:
#         """Analyze user query and retrieve relevant context"""
#         logger.info("Analyzing user query")
        
#         user_query = state["user_query"]
        
#         # Get context from vector knowledge base
#         vector_context = ""
#         try:
#             vector_agent = self.agents[AgentType.VECTOR_KNOWLEDGE]
#             context_result = vector_agent.query_knowledge(user_query, k=3)
#             vector_context = context_result.retrieved_knowledge
#         except Exception as e:
#             logger.error(f"Error getting vector context: {e}")
        
#         return {
#             "vector_context": vector_context
#         }
    
#     def _create_plan_node(self, state: SuperAgentState) -> Dict[str, Any]:
#         """Create dynamic workflow plan based on query analysis"""
#         logger.info("Creating workflow plan")
        
#         user_query = state["user_query"]
#         vector_context = state.get("vector_context", "")
#         conversation_memory = state.get("conversation_memory")
        
#         # Build context for planning
#         context = f"User Query: {user_query}\n"
#         if vector_context:
#             context += f"Relevant Knowledge: {vector_context}\n"
#         if conversation_memory:
#             recent_messages = conversation_memory.messages[-3:] if conversation_memory.messages else []
#             context += f"Recent Conversation: {json.dumps(recent_messages)}\n"
        
#         # Use LLM to create workflow plan
#         planning_prompt = ChatPromptTemplate.from_messages([
#             ("system", """
# You are a workflow planning expert. Based on the user query and available context, create a dynamic workflow plan.

# Available Agents:
# - CHATBOT: General conversation and simple Q&A
# - WEB_SCRAPING: Research information from websites
# - DATABASE_QUERY: Query SQL databases or flat files
# - VECTOR_KNOWLEDGE: Query vector knowledge base

# Create a workflow plan with specific steps. For complex or potentially harmful operations, require human approval.
#             """),
#             ("user", "Context: {context}\n\nCreate a workflow plan.")
#         ])
        
#         class PlanGenerator(BaseModel):
#             steps: List[Dict[str, Any]] = Field(description="List of workflow steps")
#             requires_human_approval: bool = Field(description="Whether human approval is needed")
#             reasoning: str = Field(description="Reasoning for the plan")
#             estimated_duration: str = Field(description="Estimated completion time")
        
#         planner = planning_prompt | self.llm.with_structured_output(PlanGenerator)
        
#         try:
#             plan_result = planner.invoke({"context": context})
            
#             # Convert to WorkflowPlan
#             steps = []
#             for i, step_data in enumerate(plan_result.steps):
#                 step = WorkflowStep(
#                     step_id=f"step_{i}",
#                     agent_type=AgentType(step_data.get("agent_type", "chatbot")),
#                     action=step_data.get("action", "process"),
#                     parameters=step_data.get("parameters", {}),
#                     requires_human_approval=step_data.get("requires_human_approval", False)
#                 )
#                 steps.append(step)
            
#             workflow_plan = WorkflowPlan(
#                 plan_id=f"plan_{datetime.now().timestamp()}",
#                 user_query=user_query,
#                 steps=steps,
#                 estimated_duration=plan_result.estimated_duration,
#                 requires_human_approval=plan_result.requires_human_approval,
#                 reasoning=plan_result.reasoning
#             )
            
#             return {
#                 "workflow_plan": workflow_plan,
#                 "requires_human_approval": plan_result.requires_human_approval
#             }
            
#         except Exception as e:
#             logger.error(f"Error creating workflow plan: {e}")
#             # Fallback to simple chatbot
#             fallback_plan = WorkflowPlan(
#                 plan_id=f"fallback_{datetime.now().timestamp()}",
#                 user_query=user_query,
#                 steps=[
#                     WorkflowStep(
#                         step_id="fallback_step",
#                         agent_type=AgentType.CHATBOT,
#                         action="chat",
#                         parameters={"message": user_query}
#                     )
#                 ],
#                 estimated_duration="< 1 minute",
#                 requires_human_approval=False,
#                 reasoning="Fallback to simple chatbot due to planning error"
#             )
            
#             return {
#                 "workflow_plan": fallback_plan,
#                 "requires_human_approval": False
#             }
    
#     def _human_approval_node(self, state: SuperAgentState) -> Dict[str, Any]:
#         """Handle human-in-the-loop approval"""
#         logger.info("Requesting human approval")
        
#         # In a real implementation, this would integrate with your Telegram bot
#         # to send an approval request and wait for response
        
#         workflow_plan = state.get("workflow_plan")
#         if workflow_plan:
#             approval_message = f"""
# 🤖 **Workflow Approval Required**

# **Query:** {workflow_plan.user_query}
# **Plan:** {workflow_plan.reasoning}
# **Steps:** {len(workflow_plan.steps)}
# **Estimated Duration:** {workflow_plan.estimated_duration}

# Reply with 'APPROVE' to proceed or 'DENY' to cancel.
#             """
            
#             # For now, we'll auto-approve (in production, implement actual approval mechanism)
#             logger.info("Auto-approving workflow (implement actual approval in production)")
            
#             return {
#                 "human_approval_received": True
#             }
        
#         return {
#             "human_approval_received": False,
#             "error_message": "No workflow plan available for approval"
#         }
    
#     def _execute_workflow_node(self, state: SuperAgentState) -> Dict[str, Any]:
#         """Execute the planned workflow steps"""
#         logger.info("Executing workflow steps")
        
#         workflow_plan = state.get("workflow_plan")
#         if not workflow_plan:
#             return {
#                 "error_message": "No workflow plan available",
#                 "final_response": "Sorry, I couldn't create a plan to handle your request."
#             }
        
#         step_results = {}
        
#         try:
#             for step in workflow_plan.steps:
#                 logger.info(f"Executing step {step.step_id}: {step.action}")
                
#                 agent = self.agents.get(step.agent_type)
#                 if not agent:
#                     step_results[step.step_id] = {
#                         "error": f"Agent {step.agent_type} not available"
#                     }
#                     continue
                
#                 # Execute step based on agent type
#                 if step.agent_type == AgentType.CHATBOT:
#                     result = agent.chat(step.parameters.get("message", ""))
#                     step_results[step.step_id] = {"response": result}
                
#                 elif step.agent_type == AgentType.WEB_SCRAPING:
#                     query = step.parameters.get("query", workflow_plan.user_query)
#                     result = agent.run(query)
#                     step_results[step.step_id] = result
                
#                 elif step.agent_type == AgentType.DATABASE_QUERY:
#                     query = step.parameters.get("query", workflow_plan.user_query)
#                     result = agent.query(query)
#                     step_results[step.step_id] = result
                
#                 elif step.agent_type == AgentType.VECTOR_KNOWLEDGE:
#                     query = step.parameters.get("query", workflow_plan.user_query)
#                     result = agent.query_knowledge(query)
#                     step_results[step.step_id] = {
#                         "contexts": result.similar_contexts,
#                         "knowledge": result.retrieved_knowledge
#                     }
                
#                 logger.info(f"Step {step.step_id} completed successfully")
        
#         except Exception as e:
#             logger.error(f"Error executing workflow: {e}")
#             return {
#                 "error_message": f"Workflow execution failed: {str(e)}",
#                 "step_results": step_results
#             }
        
#         return {
#             "step_results": step_results
#         }
    
#     def _finalize_response_node(self, state: SuperAgentState) -> Dict[str, Any]:
#         """Combine workflow results into final response"""
#         logger.info("Finalizing response")
        
#         user_query = state["user_query"]
#         step_results = state.get("step_results", {})
#         error_message = state.get("error_message", "")
        
#         if error_message:
#             return {
#                 "final_response": f"I encountered an error: {error_message}"
#             }
        
#         # Use LLM to combine results into coherent response
#         response_prompt = ChatPromptTemplate.from_messages([
#             ("system", """
# You are an AI assistant. Combine the workflow results into a coherent, helpful response to the user's query.
# Make the response natural and conversational while including all relevant information from the results.
#             """),
#             ("user", "User Query: {query}\n\nWorkflow Results: {results}\n\nProvide a comprehensive response.")
#         ])
        
#         try:
#             response_chain = response_prompt | self.llm
#             final_response = response_chain.invoke({
#                 "query": user_query,
#                 "results": json.dumps(step_results, indent=2)
#             })
            
#             return {
#                 "final_response": final_response.content if hasattr(final_response, 'content') else str(final_response)
#             }
            
#         except Exception as e:
#             logger.error(f"Error finalizing response: {e}")
#             # Fallback response
#             return {
#                 "final_response": "I processed your request but encountered an issue formatting the response. Here are the raw results: " + json.dumps(step_results, indent=2)
#             }
    
#     def _save_memory_node(self, state: SuperAgentState) -> Dict[str, Any]:
#         """Save conversation to memory"""
#         logger.info("Saving conversation memory")
        
#         user_id = state["user_id"]
#         user_query = state["user_query"]
#         final_response = state.get("final_response", "")
#         workflow_plan = state.get("workflow_plan")
        
#         # Load or create memory
#         memory = self._load_conversation_memory(user_id)
#         if not memory:
#             memory = ConversationMemory(
#                 user_id=user_id,
#                 messages=[],
#                 context={},
#                 last_workflow=None,
#                 created_at=datetime.now(),
#                 updated_at=datetime.now()
#             )
        
#         # Add new messages
#         memory.messages.append({
#             "role": "user",
#             "content": user_query,
#             "timestamp": datetime.now().isoformat()
#         })
        
#         memory.messages.append({
#             "role": "assistant",
#             "content": final_response,
#             "timestamp": datetime.now().isoformat()
#         })
        
#         # Update context
#         if workflow_plan:
#             memory.last_workflow = workflow_plan.plan_id
#             memory.context["last_plan"] = workflow_plan.dict()
        
#         # Keep only last 20 messages to prevent database bloat
#         if len(memory.messages) > 20:
#             memory.messages = memory.messages[-20:]
        
#         # Save memory
#         self._save_conversation_memory(memory)
        
#         return {}
    
#     async def process_message(self, user_id: str, message: str) -> str:
#         """
#         Main entry point - uses intelligent routing instead of workflow planning
#         """
#         try:
#             logger.info(f"Processing message from user {user_id}: {message}")
            
#             # Load conversation context
#             memory = self._load_conversation_memory(user_id)
#             context = None
#             if memory and memory.messages:
#                 # Get recent conversation for context
#                 recent_messages = memory.messages[-3:]
#                 context = " ".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
            
#             # Route and execute with intelligent router
#             result = self.router.execute_routing(message, context)
            
#             # Extract the response
#             primary_response = result.get("primary_response", "I couldn't process your request.")
#             routing_info = result.get("routing_decision", {})
            
#             # Log routing decision for debugging
#             logger.info(f"Routed to {routing_info.get('primary_agent')} with confidence {routing_info.get('confidence')}")
            
#             # Handle multiple agent responses if needed
#             if result.get("requires_multiple_agents") and result.get("secondary_responses"):
#                 # Combine responses intelligently
#                 combined_response = self._combine_responses(primary_response, result["secondary_responses"])
#                 final_response = combined_response
#             else:
#                 final_response = primary_response
            
#             # Save conversation memory
#             self._save_conversation_message(user_id, message, final_response)
            
#             logger.info(f"Response generated for user {user_id}")
#             return final_response
            
#         except Exception as e:
#             logger.error(f"Error processing message: {e}")
#             return f"I encountered an error processing your request: {str(e)}"
    
#     def _combine_responses(self, primary_response: str, secondary_responses: Dict[str, str]) -> str:
#         """Combine multiple agent responses intelligently"""
#         try:
#             # Use LLM to combine responses coherently
#             combine_prompt = ChatPromptTemplate.from_messages([
#                 ("system", """
# You are an expert at combining multiple AI agent responses into a single coherent answer.
# Combine the responses naturally, avoiding redundancy while preserving all important information.
# Make it sound like a single, well-structured response.
#                 """),
#                 ("user", """
# Primary Response: {primary}

# Secondary Responses: {secondary}

# Combine these into a single, coherent response.
#                 """)
#             ])
            
#             combiner = combine_prompt | self.llm
#             result = combiner.invoke({
#                 "primary": primary_response,
#                 "secondary": str(secondary_responses)
#             })
            
#             return result.content if hasattr(result, 'content') else str(result)
            
#         except Exception as e:
#             logger.error(f"Error combining responses: {e}")
#             # Fallback to simple concatenation
#             combined = primary_response
#             for agent_type, response in secondary_responses.items():
#                 combined += f"\n\nAdditionally from {agent_type}: {response}"
#             return combined
    
#     def _save_conversation_message(self, user_id: str, user_message: str, bot_response: str):
#         """Save conversation message to memory"""
#         try:
#             # Load or create memory
#             memory = self._load_conversation_memory(user_id)
#             if not memory:
#                 memory = ConversationMemory(
#                     user_id=user_id,
#                     messages=[],
#                     context={},
#                     last_workflow=None,
#                     created_at=datetime.now(),
#                     updated_at=datetime.now()
#                 )
            
#             # Add messages
#             memory.messages.extend([
#                 {
#                     "role": "user", 
#                     "content": user_message,
#                     "timestamp": datetime.now().isoformat()
#                 },
#                 {
#                     "role": "assistant",
#                     "content": bot_response, 
#                     "timestamp": datetime.now().isoformat()
#                 }
#             ])
            
#             # Keep only last 20 messages
#             if len(memory.messages) > 20:
#                 memory.messages = memory.messages[-20:]
            
#             # Save memory
#             self._save_conversation_memory(memory)
            
#         except Exception as e:
#             logger.error(f"Error saving conversation: {e}")
    
#     def get_routing_stats(self) -> Dict[str, Any]:
#         """Get statistics about routing decisions"""
#         # This could be implemented to track routing patterns
#         return {
#             "message": "Routing statistics not implemented yet",
#             "available_agents": list(self.agents.keys())
#         }
    
#     def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
#         """Get conversation history for a user"""
#         memory = self._load_conversation_memory(user_id)
#         if memory and memory.messages:
#             return memory.messages[-limit:]
#         return []
    
#     def clear_conversation_memory(self, user_id: str) -> bool:
#         """Clear conversation memory for a user"""
#         try:
#             conn = sqlite3.connect(self.memory_db_path)
#             cursor = conn.cursor()
#             cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
#             conn.commit()
#             conn.close()
#             logger.info(f"Memory cleared for user {user_id}")
#             return True
#         except Exception as e:
#             logger.error(f"Error clearing memory: {e}")
#             return False


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



import os
import json
import requests
from typing import Dict, List, Any, Optional, Sequence
from pydantic import BaseModel, Field, create_model
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
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
